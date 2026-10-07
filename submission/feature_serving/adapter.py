"""CPU-only causal adapter for the frozen 422+144 feature detector.

Resources contain read-only models shared across series. Every State owns its
AR filter, V4 state, calibration references, cumulative buffers and tick count.
No online length, labels, peer values or training-workspace files are read.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import threading
from typing import Mapping

import lightgbm as lgb
import numpy as np
from scipy.linalg import solve
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits
import torch

from .constants import INFER_STRIDE, ORDER
from .features import WINDOWS, channels
from .predictor import Predictor, inputs
from .legacy import v2_candidate, v3_candidate, v5_sequential_evidence
from .legacy.v817_exact_pairwise_v2_pit_transport import ExactPairwisePitV4Seq18State, _install


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, expected in manifest['runtime_sources_sha256'].items():
        path = root / name
        if file_sha256(path) != expected:
            raise RuntimeError(f'Serving source identity changed: {name}')
    for name, key in [('vendor_manifest.json', 'vendor_manifest_sha256'),
                      ('requirements.txt', 'requirements_sha256')]:
        if file_sha256(root / name) != manifest[key]:
            raise RuntimeError(f'Serving package metadata identity changed: {name}')
    return dict(manifest['runtime_sources_sha256'])


class _FeatureModules:
    """Compatibility registry; modules contain code, never per-series state."""
    def _import_experiment(self, name):
        return {'v2_candidate': v2_candidate, 'v3_candidate': v3_candidate,
                'v5_sequential_evidence': v5_sequential_evidence}[name]


class Resources:
    """Load models from a supplied directory; no paths are inferred externally.

    Required: predictor0/latest.pt, predictor1/latest.pt, augmented_seed0.txt,
    augmented_seed1.txt, augmented_seed2.txt. All three control_seed*.txt files
    may additionally be supplied to enable paired control/augmented outputs.
    """
    def __init__(self, model_directory: str | Path,
                 expected_hashes: Mapping[str, str] | None = None):
        self.source_hashes = verify_sources()
        self.model_directory = Path(model_directory).resolve()
        self.model_hashes: dict[str, str] = {}
        self._prediction_lock = threading.Lock()
        self._features = _FeatureModules()
        # Fix this pure helper before any series is initialized, so first-use
        # import order cannot choose a different implementation for one series.
        _install(v2_candidate)
        torch.set_num_threads(1)
        self.predictors = []
        fit_ids = []
        for inner in (0, 1):
            name = f'predictor{inner}/latest.pt'
            path = self._authenticate(name, expected_hashes)
            # These checkpoints are trusted artifacts from the accompanying
            # training code. CPU mapping prevents GPU allocation at load time.
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint.get('step') != 800 or 'model' not in checkpoint:
                raise ValueError(f'Incomplete or incompatible predictor: {name}')
            with torch.random.fork_rng(devices=[]):
                predictor = Predictor().cpu().eval()
            predictor.load_state_dict(checkpoint['model'], strict=True)
            predictor.requires_grad_(False)
            self.predictors.append(predictor)
            fit_ids.append(np.asarray(checkpoint['fit_ids'], dtype=np.int64))
        if len(fit_ids[0]) != len(fit_ids[1]) or np.intersect1d(*fit_ids).size:
            raise ValueError('Predictor training partitions must be disjoint and equally sized.')
        self.predictor_training_series = tuple(len(ids) for ids in fit_ids)
        self.augmented_heads = self._load_heads('augmented', 566, expected_hashes)
        controls = [(self.model_directory / f'control_seed{seed}.txt').is_file() for seed in (0, 1, 2)]
        if any(controls) and not all(controls):
            raise ValueError('Supply all three control heads, or none.')
        self.control_heads = self._load_heads('control', 422, expected_hashes) if all(controls) else ()

    def _authenticate(self, name, expected_hashes):
        path = self.model_directory / name
        digest = file_sha256(path)
        if expected_hashes is not None and expected_hashes.get(name) != digest:
            raise RuntimeError(f'Model identity changed or missing from expected hashes: {name}')
        self.model_hashes[name] = digest
        return path

    def _load_heads(self, arm, columns, expected_hashes):
        heads = []
        for seed in (0, 1, 2):
            path = self._authenticate(f'{arm}_seed{seed}.txt', expected_hashes)
            head = lgb.Booster(model_file=str(path))
            if head.num_feature() != columns or head.num_trees() != 500:
                raise ValueError(f'Unexpected {arm} head shape: {path.name}')
            heads.append(head)
        return tuple(heads)

    def predict(self, base, neural) -> tuple[float | None, float]:
        matrix = np.stack([np.concatenate([base, row]) for row in neural])
        if matrix.shape != (2, 566) or not np.isfinite(matrix).all():
            raise ValueError('Invalid integrated feature row.')
        with self._prediction_lock:
            # Average each view's seed probabilities first, matching CV.
            view_probability = np.mean(
                [head.predict(matrix, num_threads=1) for head in self.augmented_heads], axis=0)
            augmented = float(.5 * view_probability[0] + .5 * view_probability[1])
            control = (float(np.mean([head.predict(base[None], num_threads=1)[0]
                                      for head in self.control_heads])) if self.control_heads else None)
        if not np.isfinite(augmented) or not 0 <= augmented <= 1:
            raise RuntimeError('Nonfinite or invalid augmented probability.')
        if control is not None and (not np.isfinite(control) or not 0 <= control <= 1):
            raise RuntimeError('Nonfinite or invalid control probability.')
        return control, augmented

    def describe(self):
        return {'cpu_only': True, 'torch_threads': torch.get_num_threads(),
                'lightgbm_prediction_threads': 1, 'infer_stride': INFER_STRIDE,
                'control_available': bool(self.control_heads),
                'predictor_training_series': list(self.predictor_training_series),
                'model_sha256': dict(self.model_hashes), 'source_sha256': dict(self.source_hashes),
                'versions': {'torch': torch.__version__, 'lightgbm': lgb.__version__, 'numpy': np.__version__}}


class _ARState:
    def __init__(self, history):
        h = len(history)
        cut = h // 2
        self.mu = float(np.mean(history[:cut]))
        self.sd = max(float(np.std(history[:cut])), 1e-5)
        z = (history-self.mu) / self.sd
        design = np.column_stack([z[ORDER-k-1:cut-k-1] for k in range(ORDER)])
        gram = design.T @ design
        ridge = .01 * np.trace(gram) / ORDER + 1e-5
        coefficients = solve(gram+ridge*np.eye(ORDER), design.T @ z[ORDER:cut], assume_a='pos')
        self.filter = np.r_[1., -coefficients]
        residual, self.filter_state = lfilter(self.filter, [1.], z, zi=np.zeros(ORDER))
        self.residual_scale = max(float(np.sqrt(np.mean(residual[ORDER:cut]**2))), .02)
        self.history_values = z.astype('f4')
        self.history_residuals = (residual/self.residual_scale).astype('f4')
        self.static = np.r_[coefficients, np.log(self.residual_scale), np.log(self.sd),
                            self.mu, np.log(h)/9].astype('f4')

    def update(self, value):
        z = (value-self.mu)/self.sd
        residual, self.filter_state = lfilter(self.filter, [1.], [z], zi=self.filter_state)
        return np.float32(z), np.float32(residual[0]/self.residual_scale)


class _NeuralState:
    def __init__(self, predictor, values, residuals, static):
        self.predictor = predictor
        self.static = torch.from_numpy(static[None])
        cal = min(512, len(values)//2-64)
        v, r = values[-cal-64:], residuals[-cal-64:]
        with torch.inference_mode():
            quantiles = predictor(torch.from_numpy(inputs(v[:-1], r[:-1])[None]),
                                  self.static)[0, :, 63:].T.numpy().copy()
        if len(quantiles) != cal:
            raise RuntimeError('Historical prediction alignment differs.')
        _, learned = channels(v[64:], r[64:], quantiles)
        reference = learned[1:].astype('f8')
        self.reference_mean = reference.mean(axis=0)
        self.reference_scale = np.maximum(reference.std(axis=0), .05)
        self.reference_count = len(reference)
        self.values = deque(values[-64:], maxlen=64)
        self.residuals = deque(residuals[-64:], maxlen=64)
        self.last = (values[-1], residuals[-1], quantiles[-1])
        self.cumulative = np.zeros(12, dtype='f8')
        self.previous_cumulative = deque([self.cumulative.copy()], maxlen=257)
        self.count = 0

    def update(self, value, residual):
        # Predictor receptive field is63 inputs. The64th prior observation is
        # required to form the difference channel of the first retained input.
        x = inputs(np.asarray(self.values), np.asarray(self.residuals))[:, -63:]
        with torch.inference_mode():
            quantiles = self.predictor(torch.from_numpy(x[None]), self.static)[0, :, -1].numpy().copy()
        old_value, old_residual, old_quantiles = self.last
        _, learned = channels(np.array([old_value, value]), np.array([old_residual, residual]),
                              np.stack([old_quantiles, quantiles]))
        standardized = np.clip((learned[-1].astype('f8')-self.reference_mean)/self.reference_scale, -30, 30)
        self.cumulative = self.cumulative + standardized
        self.previous_cumulative.append(self.cumulative.copy())
        self.count += 1
        row = []
        for window in WINDOWS:
            n = min(window, self.count) if window else self.count
            before = self.previous_cumulative[-(n+1)] if window and self.count > window else np.zeros(12)
            delta = (self.cumulative-before)/n
            row.extend([delta, delta*np.sqrt(n/(1+n/self.reference_count))])
        row.extend([np.log1p(self.reference_scale), np.clip(self.reference_mean, -20, 20)])
        self.last = (value, residual, quantiles)
        self.values.append(value)
        self.residuals.append(residual)
        return np.concatenate(row).astype('f4')


class State:
    """One series. update returns augmented p; update_pair also returns control.

    Each call to either method consumes exactly one new observation. Choose one
    method per observation. Blend with the current incumbent output externally;
    holding these detector outputs does not require holding the incumbent.
    """
    def __init__(self, history, resources: Resources):
        historical = np.asarray(history, dtype=np.float64)
        if historical.ndim != 1 or len(historical) < 132 or not np.isfinite(historical).all():
            raise ValueError('History must be a finite one-dimensional array of at least132 observations.')
        self.resources = resources
        with threadpool_limits(limits=1):
            self._ar = _ARState(historical)
            self._v4 = ExactPairwisePitV4Seq18State(historical, ('v4_seq18',), resources._features)
        self._neural = [_NeuralState(model, self._ar.history_values, self._ar.history_residuals,
                                     self._ar.static) for model in resources.predictors]
        # Initialization arrays are not needed by online updates.
        del self._ar.history_values, self._ar.history_residuals
        self.observations_seen = 0
        self.last_pair: tuple[float | None, float | None] = (None, None)
        self.last_base_features = None
        self.last_neural_features = None
        self.last_normalized_value = None
        self.last_innovation = None

    def update_pair(self, value: float) -> tuple[float | None, float]:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError('Observation must be finite.')
        base = self._v4.update(value)['v4_seq18'][:422]
        normalized, innovation = self._ar.update(value)
        neural = [state.update(normalized, innovation) for state in self._neural]
        if self.observations_seen % INFER_STRIDE == 0:
            self.last_pair = self.resources.predict(base, neural)
        self.observations_seen += 1
        self.last_base_features = base
        self.last_neural_features = tuple(neural)
        self.last_normalized_value = normalized
        self.last_innovation = innovation
        return self.last_pair

    def update(self, value: float) -> float:
        return self.update_pair(value)[1]
