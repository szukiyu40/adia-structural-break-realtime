"""Causal NumPy-only v31 + v32 postprocessor for the ADIA v13 stack."""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COMPETITION = "ADIA Structural Break Challenge: Real-Time Edition"
CANDIDATE_ID = "v35_scaled_detector_postprocessed_v13"
MANIFEST_FILENAME = "postprocess_manifest.json"
CALIBRATION_FILENAME = "postprocess_calibration.npz"
ARRAY_NAMES = (
    "v31_moors_reference",
    "v31_tail_reference",
    "v32_hankel_reference",
    "v32_delay_reference",
)
SOURCE_PATHS = frozenset({"main.py", "postprocess_runtime.py"})
V31_ALPHA = 0.035
V32_ALPHA = 0.070
WINDOW = 128
CHECKPOINTS = (127, 255, 511)
EMBED_DIMENSION = 4
HISTORY_CLOUD_CAP = 128
SPD_EPSILON = 1e-4
CLIP_Z = 8.0
PROBABILITY_CLIP = 1e-7


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _source_path(relative_name: str) -> Path:
    root = Path(__file__).resolve().parent
    path = root / relative_name
    if not path.is_file():
        raise FileNotFoundError(f"missing postprocess source: {relative_name}")
    return path


@dataclass(frozen=True)
class PostprocessResources:
    v31_moors_reference: np.ndarray
    v31_tail_reference: np.ndarray
    v32_hankel_reference: np.ndarray
    v32_delay_reference: np.ndarray


def validate_resources(model_directory_path: str | Path) -> PostprocessResources:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing postprocess manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "status", "competition", "candidate_id",
        "base_manifest_sha256", "calibration_file", "calibration_sha256",
        "calibration_size_bytes", "arrays", "contract", "source_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("postprocess manifest field inventory changed")
    for key, expected in (
        ("schema_version", 1), ("status", "complete"),
        ("competition", COMPETITION), ("candidate_id", CANDIDATE_ID),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"postprocess manifest {key} changed")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(SOURCE_PATHS):
        raise ValueError("postprocess source inventory changed")
    for relative_name in sorted(SOURCE_PATHS):
        digest = source_hashes.get(relative_name)
        if not _is_sha256(digest) or _sha256_file(_source_path(relative_name)) != digest:
            raise RuntimeError(f"postprocess source hash mismatch: {relative_name}")
    base_digest = manifest.get("base_manifest_sha256")
    base_manifest = root / "production_manifest.json"
    if not _is_sha256(base_digest) or _sha256_file(base_manifest) != base_digest:
        raise RuntimeError("base v13 manifest hash mismatch")
    if manifest.get("calibration_file") != CALIBRATION_FILENAME:
        raise ValueError("postprocess calibration filename changed")
    calibration_path = root / CALIBRATION_FILENAME
    expected_digest = manifest.get("calibration_sha256")
    if not _is_sha256(expected_digest) or _sha256_file(calibration_path) != expected_digest:
        raise RuntimeError("postprocess calibration hash mismatch")
    if manifest.get("calibration_size_bytes") != calibration_path.stat().st_size:
        raise RuntimeError("postprocess calibration size mismatch")
    contract = manifest.get("contract")
    expected_contract = {
        "v31_alpha": V31_ALPHA,
        "v32_alpha": V32_ALPHA,
        "v31_window": WINDOW,
        "v32_checkpoints": list(CHECKPOINTS),
        "future_ticks_used": False,
        "other_series_current_values_used": False,
        "supervised_online_updates": False,
    }
    if contract != expected_contract:
        raise ValueError("postprocess runtime contract changed")
    records = manifest.get("arrays")
    if not isinstance(records, Mapping) or set(records) != set(ARRAY_NAMES):
        raise ValueError("postprocess calibration array inventory changed")
    loaded: dict[str, np.ndarray] = {}
    with np.load(calibration_path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_NAMES):
            raise ValueError("postprocess NPZ array inventory changed")
        for name in ARRAY_NAMES:
            values = np.asarray(archive[name])
            record = records[name]
            if not isinstance(record, Mapping):
                raise ValueError(f"invalid postprocess array record: {name}")
            if record.get("dtype") != "<f8" or values.dtype.str != "<f8":
                raise ValueError(f"postprocess array dtype changed: {name}")
            if record.get("shape") != list(values.shape) or values.ndim != 1 or values.size == 0:
                raise ValueError(f"postprocess array shape changed: {name}")
            if not np.isfinite(values).all() or np.any(values[1:] < values[:-1]):
                raise RuntimeError(f"postprocess array is invalid: {name}")
            digest = record.get("sha256")
            if not _is_sha256(digest) or _sha256_array(values) != digest:
                raise RuntimeError(f"postprocess array hash mismatch: {name}")
            frozen = values.copy()
            frozen.flags.writeable = False
            loaded[name] = frozen
    return PostprocessResources(**loaded)


def _clean_history(values: Sequence[float] | np.ndarray) -> np.ndarray:
    history = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(history)
    replacement = float(np.median(history[finite])) if finite.any() else 0.0
    history[~finite] = replacement
    if history.size < 512:
        raise ValueError("v31/v32 postprocessor requires at least 512 historical ticks")
    return history


def _history_scale(history: np.ndarray) -> tuple[float, float]:
    median = float(np.median(history))
    q25, q75 = np.quantile(history, (0.25, 0.75), method="linear")
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1e-12:
        mad = float(np.median(np.abs(history - median)))
        scale = float(mad / 0.6744897501960817) if mad > 0.0 else 1.0
    return median, scale


def _quantile_sorted(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(values[lower] + fraction * (values[upper] - values[lower]))


def _moors(values: np.ndarray) -> float:
    q = np.quantile(values, np.arange(1, 8, dtype=np.float64) / 8.0, method="linear")
    return float((q[6] - q[4] + q[2] - q[0]) / max(float(q[5] - q[1]), 1e-12))


def _tail_spread(values: np.ndarray) -> float:
    q05, q25, _, q75, q95 = np.quantile(values, (0.05, 0.25, 0.50, 0.75, 0.95), method="linear")
    return max(float((q95 - q05) / max(float(q75 - q25), 1e-12)), 1e-12)


def _delay_embed(values: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(values, EMBED_DIMENSION).copy()


def _corr_spd(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(vectors.shape[0] - 1, 1)
    scales = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(scales, scales)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation + SPD_EPSILON * np.eye(EMBED_DIMENSION, dtype=np.float64)


def _spd_log(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.log(np.maximum(eigenvalues, 1e-12))) @ eigenvectors.T


def _hankel_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = _spd_log(left) - _spd_log(right)
    return float(np.sqrt(np.sum(delta * delta)))


def _pairwise_sq(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    return np.sum(delta * delta, axis=2)


def _capped_history(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] <= HISTORY_CLOUD_CAP:
        return vectors.copy()
    indices = np.floor(np.arange(HISTORY_CLOUD_CAP, dtype=np.float64) * (vectors.shape[0] - 1) / (HISTORY_CLOUD_CAP - 1)).astype(np.int64)
    return vectors[indices].copy()


def _centered_ecdf(reference: np.ndarray, value: float) -> float:
    return float((np.searchsorted(reference, value, side="right") + 0.5) / (reference.size + 1.0) - 0.5)


@dataclass(frozen=True)
class PostprocessSignals:
    v31_correction: float
    v32_correction: float


class CausalPostprocessState:
    """One-series state; consumes exactly one new value per update."""

    def __init__(self, historical: Sequence[float] | np.ndarray, resources: PostprocessResources):
        history = _clean_history(historical)
        self.resources = resources
        self.median, self.scale = _history_scale(history)
        history_z = np.clip((history - self.median) / self.scale, -CLIP_Z, CLIP_Z)
        self.history_moors = _moors(history_z)
        self.history_tail_spread = _tail_spread(history_z)
        history_vectors = _delay_embed(history_z)
        self.history_spd = _corr_spd(history_vectors)
        self.history_cloud = _capped_history(history_vectors)
        history_sq = _pairwise_sq(self.history_cloud, self.history_cloud)
        triangle = np.sqrt(history_sq[np.triu_indices(self.history_cloud.shape[0], k=1)])
        sigma = float(np.median(triangle)) if triangle.size else 1.0
        if not math.isfinite(sigma) or sigma < 1e-12:
            sigma = 1.0
        self.gamma = 1.0 / (2.0 * sigma * sigma)
        self.history_kernel_mean = float(np.exp(-self.gamma * history_sq).mean())
        self.rolling: deque[float] = deque()
        self.sorted_window: list[float] = []
        self.recent: deque[float] = deque(maxlen=67)
        self.time_online = -1
        self.hankel_runmax = -math.inf
        self.v32_correction = 0.0
        self.last_moors_feature = math.nan
        self.last_tail_feature = math.nan
        self.last_hankel_feature = math.nan
        self.last_delay_feature = math.nan

    def _v31(self) -> float:
        values = self.sorted_window
        q05 = _quantile_sorted(values, 0.05)
        q125 = _quantile_sorted(values, 0.125)
        q25 = _quantile_sorted(values, 0.25)
        q375 = _quantile_sorted(values, 0.375)
        q50 = _quantile_sorted(values, 0.50)
        q625 = _quantile_sorted(values, 0.625)
        q75 = _quantile_sorted(values, 0.75)
        q875 = _quantile_sorted(values, 0.875)
        q95 = _quantile_sorted(values, 0.95)
        moors = (q875 - q625 + q375 - q125) / max(q75 - q25, 1e-12)
        tail_spread = max((q95 - q05) / max(q75 - q25, 1e-12), 1e-12)
        self.last_moors_feature = abs(float(moors) - self.history_moors)
        self.last_tail_feature = abs(math.log(tail_spread) - math.log(self.history_tail_spread))
        moors_rank = _centered_ecdf(self.resources.v31_moors_reference, self.last_moors_feature)
        tail_rank = _centered_ecdf(self.resources.v31_tail_reference, self.last_tail_feature)
        return 0.5 * (moors_rank + tail_rank)

    def _update_v32(self) -> None:
        recent = np.asarray(self.recent, dtype=np.float64)
        vectors = _delay_embed(recent)
        cloud = vectors[-32:]
        current_spd = _corr_spd(cloud)
        hankel = _hankel_distance(self.history_spd, current_spd)
        self.hankel_runmax = max(self.hankel_runmax, hankel)
        cross_sq = _pairwise_sq(self.history_cloud, cloud)
        online_sq = _pairwise_sq(cloud, cloud)
        delay = self.history_kernel_mean + float(np.exp(-self.gamma * online_sq).mean()) - 2.0 * float(np.exp(-self.gamma * cross_sq).mean())
        self.last_hankel_feature = self.hankel_runmax
        self.last_delay_feature = delay
        hankel_rank = _centered_ecdf(self.resources.v32_hankel_reference, hankel)
        delay_rank = _centered_ecdf(self.resources.v32_delay_reference, delay)
        self.v32_correction = hankel_rank - delay_rank

    def update(self, raw_value: float) -> PostprocessSignals:
        value = float(raw_value)
        if not math.isfinite(value):
            value = self.median
        z = float(np.clip((value - self.median) / self.scale, -CLIP_Z, CLIP_Z))
        self.time_online += 1
        if len(self.rolling) == WINDOW:
            outgoing = self.rolling.popleft()
            index = bisect.bisect_left(self.sorted_window, outgoing)
            if index >= len(self.sorted_window) or self.sorted_window[index] != outgoing:
                raise RuntimeError("rolling quantile state lost an outgoing value")
            self.sorted_window.pop(index)
        self.rolling.append(z)
        bisect.insort(self.sorted_window, z)
        self.recent.append(z)
        v31_correction = self._v31() if len(self.rolling) == WINDOW else 0.0
        if self.time_online in CHECKPOINTS:
            if len(self.recent) < 35:
                raise RuntimeError("v32 checkpoint lacks causal delay vectors")
            self._update_v32()
        return PostprocessSignals(v31_correction=v31_correction, v32_correction=self.v32_correction)


def apply_postprocess(probability: float, signals: PostprocessSignals) -> float:
    value = min(max(float(probability), PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)
    score = math.log(value) - math.log1p(-value)
    score += V31_ALPHA * float(signals.v31_correction)
    score += V32_ALPHA * float(signals.v32_correction)
    if score >= 0.0:
        output = 1.0 / (1.0 + math.exp(-score))
    else:
        exp_score = math.exp(score)
        output = exp_score / (1.0 + exp_score)
    return min(max(output, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)
