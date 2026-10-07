"""Deterministic causal runtime for V344/V360/V341/V377 additions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from history_process_features_runtime import extract_history_process_features
import v363_patch


EPSILON = 1e-7
FEATURE_CLIP = 30.0
WINDOWS = (8, 16, 32, 64, 128)
VALUE_BINS = 8
TRANSITION_BINS = 4
PRIOR_FLOOR = 0.5
PRIOR_STRENGTH = 4.0
ABSORPTION_FEATURE_COUNT = 96
ABSORPTION_DESIGN_COUNT = 4 * ABSORPTION_FEATURE_COUNT
ABSORPTION_START_TICK = 128
NEAR_IID_CODE = 7
MODERATE_CODE = 6
EXTREME_CODE = 0
LEGACY_SHARE = 0.04
POSITIONAL_SHARE = 0.08


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _logit(value: float) -> float:
    clipped = float(np.clip(value, EPSILON, 1.0 - EPSILON))
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponent = math.exp(value)
    return float(exponent / (1.0 + exponent))


def _finite(values: Sequence[float] | np.ndarray, replacement: float | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(result)
    if replacement is None:
        replacement = float(np.median(result[finite])) if finite.any() else 0.0
    result[~finite] = replacement
    return result


@dataclass(frozen=True)
class Resources:
    positional: Any
    legacy: tuple[Any, Any]
    process_lower: np.ndarray
    process_upper: np.ndarray
    process_center: np.ndarray
    process_scale: np.ndarray
    process_centroids: np.ndarray
    semantic_code_by_raw: np.ndarray
    process_temperature: float
    uc_beta: np.ndarray
    absorption_beta: np.ndarray
    absorption_center: np.ndarray
    absorption_scale: np.ndarray


def _validated_payload(
    model_directory_path: str | Path,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Authenticate resources without importing LightGBM before worker fork."""

    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / "v377_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("candidate_id") != "v377_mature_absorption_gate"
        or manifest.get("future_ticks_used") is not False
        or manifest.get("peer_test_series_used") is not False
        or manifest.get("deterministic") is not True
    ):
        raise RuntimeError("V377 manifest contract changed")
    for name, digest in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V377 resource changed: {name}")
    with np.load(root / "v377_parameters.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_shapes = {
        "process_lower": (67,), "process_upper": (67,), "process_center": (67,),
        "process_scale": (67,), "process_centroids": (8, 67),
        "semantic_code_by_raw": (8,), "uc_beta": (2, 120),
        "absorption_beta": (2, ABSORPTION_DESIGN_COUNT),
        "absorption_center": (2, ABSORPTION_FEATURE_COUNT),
        "absorption_scale": (2, ABSORPTION_FEATURE_COUNT),
    }
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"V377 parameter schema changed: {name}")
    if arrays.get("process_temperature", np.empty(0)).shape != (1,):
        raise RuntimeError("V377 process-temperature schema changed")
    return root, arrays


def validate_resources(model_directory_path: str | Path) -> None:
    """Pre-fork-safe hash and array validation used by ``train``."""

    _validated_payload(model_directory_path)


def load_resources(model_directory_path: str | Path) -> Resources:
    """Load models only after Crunch has forked its inference workers."""

    root, arrays = _validated_payload(model_directory_path)
    import lightgbm as lgb

    positional = lgb.Booster(model_file=str(root / "v377_positional_model.txt"))
    legacy = (
        lgb.Booster(model_file=str(root / "v377_legacy_pilot.txt")),
        lgb.Booster(model_file=str(root / "v377_legacy_full.txt")),
    )
    if positional.num_feature() != 94 or any(model.num_feature() != 422 for model in legacy):
        raise RuntimeError("V377 booster schema changed")
    return Resources(
        positional=positional,
        legacy=legacy,
        process_lower=arrays["process_lower"].astype(np.float64),
        process_upper=arrays["process_upper"].astype(np.float64),
        process_center=arrays["process_center"].astype(np.float64),
        process_scale=arrays["process_scale"].astype(np.float64),
        process_centroids=arrays["process_centroids"].astype(np.float64),
        semantic_code_by_raw=arrays["semantic_code_by_raw"].astype(np.int8),
        process_temperature=float(arrays["process_temperature"][0]),
        uc_beta=arrays["uc_beta"].astype(np.float64),
        absorption_beta=arrays["absorption_beta"].astype(np.float64),
        absorption_center=arrays["absorption_center"].astype(np.float64),
        absorption_scale=arrays["absorption_scale"].astype(np.float64),
    )


def _process_responsibility(history: np.ndarray, resources: Resources) -> np.ndarray:
    feature = extract_history_process_features(history)
    clipped = np.clip(feature, resources.process_lower, resources.process_upper)
    standardized = (clipped - resources.process_center) / resources.process_scale
    squared = np.sum((standardized[None, :] - resources.process_centroids) ** 2, axis=1)
    logits = -(squared - float(np.min(squared))) / resources.process_temperature
    logits -= float(np.max(logits))
    raw = np.exp(logits)
    raw /= float(np.sum(raw))
    semantic = np.zeros(8, dtype=np.float64)
    for raw_code, semantic_code in enumerate(resources.semantic_code_by_raw):
        semantic[int(semantic_code)] = raw[raw_code]
    return semantic


class PositionalState:
    STREAMS = ("level", "absolute", "square", "difference")

    def __init__(self, historical: np.ndarray):
        history = _finite(historical)
        self.history_size = int(history.size)
        self.center = float(np.mean(history))
        self.scale = max(float(np.std(history, ddof=0)), 1e-8)
        z = np.clip((history - self.center) / self.scale, -12.0, 12.0)
        self.history_abs_mean = float(np.mean(np.abs(z)))
        self.history_square_mean = float(np.mean(z * z))
        difference = np.diff(z)
        self.difference_center = float(np.mean(difference))
        self.difference_scale = max(float(np.std(difference, ddof=0)), 1e-8)
        self.previous_z = float(z[-1])
        self.count = 0
        self.cumulative = {name: [0.0] for name in self.STREAMS}
        self.cumulative_square = {name: [0.0] for name in self.STREAMS}

    def _moments(self, name: str, bins: int) -> tuple[np.ndarray, np.ndarray]:
        edges = np.floor(self.count * np.arange(bins + 1, dtype=np.float64) / bins).astype(np.int64)
        means = np.zeros(bins, dtype=np.float64)
        variances = np.zeros(bins, dtype=np.float64)
        cumulative = self.cumulative[name]
        square = self.cumulative_square[name]
        for index in range(bins):
            left, right = int(edges[index]), int(edges[index + 1])
            width = right - left
            if width:
                total = cumulative[right] - cumulative[left]
                mean = total / width
                means[index] = mean
                variances[index] = max((square[right] - square[left]) / width - mean * mean, 0.0)
        return means, variances

    @staticmethod
    def _aggregate(values: np.ndarray) -> np.ndarray:
        differences = np.diff(values)
        path = float(np.sum(np.abs(differences)))
        endpoint = abs(float(values[-1] - values[0]))
        curvature = float(np.max(np.abs(np.diff(values, n=2)))) if values.size >= 3 else 0.0
        return np.asarray((float(np.max(values) - np.min(values)), path, endpoint, max(path - endpoint, 0.0), curvature))

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.center
        z = float(np.clip((clean - self.center) / self.scale, -12.0, 12.0))
        raw_difference = z - self.previous_z
        difference = float(np.clip((raw_difference - self.difference_center) / self.difference_scale, -12.0, 12.0))
        self.previous_z = z
        values = {
            "level": z,
            "absolute": abs(z) - self.history_abs_mean,
            "square": z * z - self.history_square_mean,
            "difference": difference,
        }
        self.count += 1
        for name, current in values.items():
            self.cumulative[name].append(self.cumulative[name][-1] + current)
            self.cumulative_square[name].append(self.cumulative_square[name][-1] + current * current)
        moments: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        parts: list[np.ndarray] = []
        for name in self.STREAMS:
            for bins in (4, 8):
                moments[(name, bins)] = self._moments(name, bins)
                parts.append(moments[(name, bins)][0])
        for name in ("level", "difference"):
            for bins in (4, 8):
                parts.append(np.log(moments[(name, bins)][1] + 1e-8))
        for name in self.STREAMS:
            parts.append(self._aggregate(moments[(name, 8)][0]))
        parts.append(np.asarray((math.log1p(self.count), math.log1p(self.history_size))))
        output = np.concatenate(parts)
        np.nan_to_num(output, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(output, -20.0, 20.0, out=output)
        if output.shape != (94,):
            raise RuntimeError("V377 positional row changed")
        return output.astype(np.float32)


def _categories(reference: np.ndarray, values: np.ndarray, bins: int) -> np.ndarray:
    ordered = np.sort(reference, kind="stable")
    ranks = np.searchsorted(ordered, values, side="right")
    return np.clip(np.floor(ranks * float(bins) / float(ordered.size + 1)).astype(np.int16), 0, bins - 1)


class UniversalCodeState:
    def __init__(self, historical: np.ndarray):
        history = _finite(historical)
        self.replacement = float(np.median(history))
        self.history = history
        self.value_order = np.sort(history, kind="stable")
        increments = np.diff(history)
        self.increment_order = np.sort(increments, kind="stable")
        history4 = _categories(history, history, TRANSITION_BINS)
        transition = (history4[:-1] * TRANSITION_BINS + history4[1:]).astype(np.int16)
        ordinal = self._ordinal_series(history)
        references = (
            _categories(history, history, VALUE_BINS),
            _categories(increments, increments, VALUE_BINS),
            transition,
            ordinal,
        )
        category_counts = (8, 8, 16, 6)
        self.prior = []
        self.log_null = []
        self.tables: list[dict[int, tuple[float, np.ndarray]]] = []
        for reference, count in zip(references, category_counts, strict=True):
            counts = np.bincount(reference, minlength=count).astype(np.float64)
            null = (counts + PRIOR_FLOOR) / (float(np.sum(counts)) + PRIOR_FLOOR * count)
            prior = PRIOR_FLOOR + PRIOR_STRENGTH * null
            self.prior.append(prior)
            self.log_null.append(np.log(np.maximum(null, 1e-12)))
            by_window: dict[int, tuple[float, np.ndarray]] = {}
            for window in WINDOWS:
                constant = math.lgamma(float(np.sum(prior))) - math.lgamma(float(np.sum(prior)) + window)
                table = np.empty((count, window + 1), dtype=np.float64)
                for category in range(count):
                    base = math.lgamma(float(prior[category]))
                    table[category] = [math.lgamma(value + float(prior[category])) - base for value in range(window + 1)]
                by_window[window] = (constant, table)
            self.tables.append(by_window)
        self.cumulative = [[np.zeros(count, dtype=np.int32)] for count in category_counts]
        self.previous_value = float(history[-1])
        self.previous_value4 = int(_categories(history, np.asarray((history[-1],)), 4)[0])
        self.last_two = [float(history[-2]), float(history[-1])]
        self.count = 0

    @staticmethod
    def _ordinal_code(values: Sequence[float]) -> int:
        order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
        code = int(order[0] * 9 + order[1] * 3 + order[2])
        return {5: 0, 7: 1, 11: 2, 15: 3, 19: 4, 21: 5}[code]

    @classmethod
    def _ordinal_series(cls, values: np.ndarray) -> np.ndarray:
        return np.asarray([cls._ordinal_code(values[index:index + 3]) for index in range(values.size - 2)], dtype=np.int16)

    @staticmethod
    def _one_category(ordered: np.ndarray, value: float, bins: int) -> int:
        rank = int(np.searchsorted(ordered, value, side="right"))
        return int(np.clip(math.floor(rank * bins / (ordered.size + 1)), 0, bins - 1))

    def _logbf(self, stream: int, window: int) -> float:
        if self.count < window:
            return 0.0
        counts = self.cumulative[stream][self.count] - self.cumulative[stream][self.count - window]
        constant, table = self.tables[stream][window]
        alternative = constant + sum(table[index, int(value)] for index, value in enumerate(counts))
        null = float(np.dot(counts, self.log_null[stream]))
        return float(np.clip(alternative - null, -FEATURE_CLIP, FEATURE_CLIP))

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.replacement
        value8 = self._one_category(self.value_order, clean, 8)
        increment = clean - self.previous_value
        increment8 = self._one_category(self.increment_order, increment, 8)
        value4 = self._one_category(self.value_order, clean, 4)
        transition = self.previous_value4 * 4 + value4
        ordinal = self._ordinal_code((self.last_two[0], self.last_two[1], clean))
        self.previous_value = clean
        self.previous_value4 = value4
        self.last_two = [self.last_two[1], clean]
        categories = (value8, increment8, transition, ordinal)
        self.count += 1
        for stream, category in enumerate(categories):
            current = self.cumulative[stream][-1].copy()
            current[category] += 1
            self.cumulative[stream].append(current)
        output: list[float] = []
        aggregates = []
        for stream in range(4):
            window_values = [self._logbf(stream, window) for window in WINDOWS]
            output.extend(window_values)
            available = np.asarray([
                value if self.count >= window else -FEATURE_CLIP
                for value, window in zip(window_values, WINDOWS, strict=True)
            ])
            maximum = float(np.max(available))
            mixture = float(np.clip(maximum + math.log(float(np.mean(np.exp(available - maximum)))), -FEATURE_CLIP, FEATURE_CLIP))
            output.extend((mixture, maximum))
            aggregates.append(mixture)
        maximum = float(max(aggregates))
        output.extend((float(np.clip(maximum + math.log(float(np.mean(np.exp(np.asarray(aggregates) - maximum)))), -FEATURE_CLIP, FEATURE_CLIP)), maximum))
        result = np.asarray(output, dtype=np.float32)
        if result.shape != (30,) or not np.isfinite(result).all():
            raise RuntimeError("V377 universal-code row changed")
        return result


class PatchState:
    def __init__(self, historical: np.ndarray, selected: np.ndarray):
        history = _finite(historical)
        self.history_size = int(history.size)
        self.replacement = float(np.median(history))
        center = float(np.median(history))
        scale = max(float(1.4826 * np.median(np.abs(history - center))), float(np.std(history)), 1e-8)
        history_z = np.clip((history - center) / scale, -12.0, 12.0)
        reference = v363_patch._patch_features(v363_patch._reference_windows(history_z))  # noqa: SLF001
        self.reference_center = np.median(reference, axis=0)
        reference_scale = 1.4826 * np.median(np.abs(reference - self.reference_center), axis=0)
        fallback = np.std(reference, axis=0)
        self.reference_scale = np.maximum(np.where(reference_scale > 1e-6, reference_scale, fallback), 1e-4)
        self.center = center
        self.scale = scale
        self.window = list(history_z[-63:])
        self.running = np.zeros(44, dtype=np.float64)
        self.decayed = np.zeros(44, dtype=np.float64)
        self.count = 0
        self.selected = selected

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.replacement
        z = float(np.clip((clean - self.center) / self.scale, -12.0, 12.0))
        self.window.append(z)
        if len(self.window) > 64:
            self.window.pop(0)
        raw = v363_patch._patch_features(np.asarray(self.window, dtype=np.float64)[None, :])[0]  # noqa: SLF001
        signed = np.clip((raw - self.reference_center) / self.reference_scale, -20.0, 20.0)
        absolute = np.abs(signed)
        self.running = np.maximum(self.running, absolute)
        self.decayed = np.maximum(0.99 * self.decayed, absolute)
        self.count += 1
        full = np.concatenate((signed, absolute, self.running, self.decayed, (math.log1p(self.count), math.log1p(self.history_size))))
        return full[self.selected].astype(np.float32)


_REDUCED_RAW_INDICES = np.asarray((1, 17, 23, 24, 26, 31, 32, 33, 36, 38), dtype=np.int64)
_SIGNED_RAW_INDICES = np.asarray((1, 31, 36, 24, 23, 26), dtype=np.int64)
_ABSOLUTE_RAW_INDICES = np.asarray((17, 31, 33, 32, 38, 1), dtype=np.int64)
_RAW_POSITION = {int(value): index for index, value in enumerate(_REDUCED_RAW_INDICES)}
_SIGNED_POSITIONS = np.asarray([_RAW_POSITION[int(value)] for value in _SIGNED_RAW_INDICES], dtype=np.int64)
_ABSOLUTE_POSITIONS = np.asarray([_RAW_POSITION[int(value)] for value in _ABSOLUTE_RAW_INDICES], dtype=np.int64)


def _reduced_patch_features(windows: np.ndarray) -> np.ndarray:
    """Exact V363 values for only the ten raw columns used by V373."""

    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 64:
        raise ValueError("V377 expects 64-tick patch windows")
    amplitude = np.abs(np.fft.rfft(w, axis=1))[:, 1:]
    probability = amplitude / np.maximum(amplitude.sum(axis=1, keepdims=True), 1e-8)
    spectral_entropy = -np.sum(probability * np.log(probability + 1e-12), axis=1)

    envelope = np.abs(v363_patch.hilbert(w, axis=1))
    envelope_mean = np.maximum(envelope.mean(axis=1), 1e-8)
    envelope_tv = np.mean(np.abs(np.diff(envelope, axis=1)), axis=1) / envelope_mean

    power = amplitude * amplitude
    power /= np.maximum(power.sum(axis=1, keepdims=True), 1e-8)
    bands = np.array_split(np.arange(power.shape[1]), 6)
    band_values = [
        np.log(np.maximum(power[:, indices].sum(axis=1), 1e-8))
        for indices in bands
    ]

    difference = np.diff(w, axis=1)
    second_difference = np.diff(w, n=2, axis=1)
    curvature_positive = np.maximum(second_difference, 0.0)
    curvature_negative = np.minimum(second_difference, 0.0)
    curvature_abs_mean = np.mean(np.abs(second_difference), axis=1)
    curvature_abs_max = np.max(np.abs(second_difference), axis=1)
    curvature_energy_ratio = np.log(
        (np.sum(curvature_positive * curvature_positive, axis=1) + 1e-6)
        / (np.sum(curvature_negative * curvature_negative, axis=1) + 1e-6)
    )

    centered = w - w.mean(axis=1, keepdims=True)
    energy = np.maximum(np.sum(centered * centered, axis=1), 1e-8)
    acf = np.column_stack(
        [
            np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / energy
            for lag in range(1, 17)
        ]
    )
    local_minimum = (acf[:, 1:-1] < acf[:, :-2]) & (
        acf[:, 1:-1] <= acf[:, 2:]
    )
    has_minimum = local_minimum.any(axis=1)
    minimum_index = np.argmax(local_minimum, axis=1) + 1
    minimum_index = np.where(has_minimum, minimum_index, acf.shape[1] - 1)
    after_index = np.minimum(minimum_index + 1, acf.shape[1] - 1)
    acf_after_minimum = acf[np.arange(w.shape[0]), after_index]

    z = centered / np.maximum(w.std(axis=1, keepdims=True), 1e-6)
    centers = np.linspace(-2.5, 2.5, 10)
    soft = np.exp(
        -0.5 * ((centers[None, None, :] - z[:, :, None]) / (5.0 / 9.0)) ** 2
    ).sum(axis=1)
    soft = np.exp(4.0 * (soft - soft.max(axis=1, keepdims=True)))
    soft /= np.maximum(soft.sum(axis=1, keepdims=True), 1e-8)
    mode = soft @ centers
    mode_minus_median = mode - np.median(z, axis=1)

    output = np.column_stack(
        (
            spectral_entropy,
            envelope_tv,
            band_values[1],
            band_values[2],
            band_values[4],
            curvature_abs_mean,
            curvature_abs_max,
            curvature_energy_ratio,
            acf_after_minimum,
            mode_minus_median,
        )
    )
    np.nan_to_num(output, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
    np.clip(output, -20.0, 20.0, out=output)
    if output.shape != (w.shape[0], _REDUCED_RAW_INDICES.size):
        raise RuntimeError("V377 reduced patch feature shape changed")
    return output


class AbsorptionState:
    """Streaming equivalent of the frozen V373 96-column memory bank."""

    def __init__(self, historical: np.ndarray):
        history = _finite(historical)
        self.replacement = float(np.median(history))
        center = float(np.median(history))
        scale = max(
            float(1.4826 * np.median(np.abs(history - center))),
            float(np.std(history)),
            1e-8,
        )
        history_z = np.clip((history - center) / scale, -12.0, 12.0)
        reference = _reduced_patch_features(v363_patch._reference_windows(history_z))  # noqa: SLF001
        self.reference_center = np.median(reference, axis=0)
        reference_scale = 1.4826 * np.median(
            np.abs(reference - self.reference_center), axis=0
        )
        fallback = np.std(reference, axis=0)
        self.reference_scale = np.maximum(
            np.where(reference_scale > 1e-6, reference_scale, fallback), 1e-4
        )
        self.center = center
        self.scale = scale
        self.window = list(history_z[-63:])
        self.count = 0
        self.signed_sum = np.zeros(6, dtype=np.float64)
        self.signed_fast = np.zeros(6, dtype=np.float64)
        self.signed_slow = np.zeros(6, dtype=np.float64)
        self.signed_positive_sum = np.zeros(6, dtype=np.float64)
        self.signed_negative_sum = np.zeros(6, dtype=np.float64)
        self.signed_positive_count = np.zeros(6, dtype=np.float64)
        self.signed_minimum = np.full(6, np.inf, dtype=np.float64)
        self.signed_maximum = np.full(6, -np.inf, dtype=np.float64)
        self.absolute_sum = np.zeros(6, dtype=np.float64)
        self.absolute_fast = np.zeros(6, dtype=np.float64)
        self.absolute_slow = np.zeros(6, dtype=np.float64)
        self.absolute_above_count = np.zeros(6, dtype=np.float64)
        self.absolute_excess_sum = np.zeros(6, dtype=np.float64)
        self.absolute_maximum = np.full(6, -np.inf, dtype=np.float64)
        self.absolute_age = np.zeros(6, dtype=np.float64)

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.replacement
        z = float(np.clip((clean - self.center) / self.scale, -12.0, 12.0))
        self.window.append(z)
        if len(self.window) > 64:
            self.window.pop(0)
        raw = _reduced_patch_features(
            np.asarray(self.window, dtype=np.float64)[None, :]
        )[0]
        calibrated = np.clip(
            (raw - self.reference_center) / self.reference_scale, -20.0, 20.0
        ).astype(np.float32).astype(np.float64)
        signed = calibrated[_SIGNED_POSITIONS]
        absolute = np.abs(calibrated[_ABSOLUTE_POSITIONS])
        self.count += 1
        denominator = float(self.count)

        self.signed_sum += signed
        self.signed_fast = 0.875 * self.signed_fast + 0.125 * signed
        self.signed_slow = (63.0 / 64.0) * self.signed_slow + signed / 64.0
        self.signed_positive_sum += np.maximum(signed, 0.0)
        self.signed_negative_sum += np.maximum(-signed, 0.0)
        self.signed_positive_count += signed > 0.0
        self.signed_minimum = np.minimum(self.signed_minimum, signed)
        self.signed_maximum = np.maximum(self.signed_maximum, signed)
        signed_parts = (
            self.signed_sum / denominator,
            self.signed_fast,
            self.signed_slow,
            self.signed_fast - self.signed_slow,
            self.signed_positive_sum / denominator,
            self.signed_negative_sum / denominator,
            self.signed_positive_count / denominator,
            self.signed_maximum - self.signed_minimum,
        )

        self.absolute_sum += absolute
        self.absolute_fast = 0.875 * self.absolute_fast + 0.125 * absolute
        self.absolute_slow = (63.0 / 64.0) * self.absolute_slow + absolute / 64.0
        self.absolute_above_count += absolute > 1.0
        self.absolute_excess_sum += np.maximum(absolute - 1.0, 0.0)
        new_maximum = absolute >= self.absolute_maximum
        self.absolute_maximum = np.maximum(self.absolute_maximum, absolute)
        self.absolute_age = np.where(new_maximum, 0.0, self.absolute_age + 1.0)
        absolute_parts = (
            self.absolute_sum / denominator,
            self.absolute_fast,
            self.absolute_slow,
            self.absolute_fast - self.absolute_slow,
            self.absolute_above_count / denominator,
            self.absolute_excess_sum / denominator,
            self.absolute_maximum - absolute,
            np.log1p(self.absolute_age),
        )
        output = np.concatenate(
            (
                np.stack(signed_parts, axis=1).reshape(-1),
                np.stack(absolute_parts, axis=1).reshape(-1),
            )
        )
        np.nan_to_num(output, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(output, -20.0, 20.0, out=output)
        if output.shape != (ABSORPTION_FEATURE_COUNT,) or not np.isfinite(output).all():
            raise RuntimeError("V377 absorption row changed")
        return output.astype(np.float32)


def _uc_design(raw: np.ndarray, base_logit: float) -> np.ndarray:
    scaled = np.clip(np.asarray(raw, dtype=np.float64) / 10.0, -3.0, 3.0)
    positive = np.maximum(scaled, 0.0)
    gate = float(np.clip(base_logit / 3.0, -1.0, 1.0))
    return np.concatenate((scaled, positive, scaled * gate, positive * gate))


def _absorption_design(
    raw: np.ndarray, center: np.ndarray, scale: np.ndarray, base_logit: float
) -> np.ndarray:
    standardized = np.clip(
        (np.asarray(raw, dtype=np.float64) - center) / scale, -4.0, 4.0
    )
    positive = np.maximum(standardized, 0.0)
    gate = float(np.clip(base_logit / 3.0, -1.0, 1.0))
    output = np.concatenate(
        (standardized, positive, standardized * gate, positive * gate)
    )
    if output.shape != (ABSORPTION_DESIGN_COUNT,) or not np.isfinite(output).all():
        raise RuntimeError("V377 absorption design changed")
    return output


class AdditionsState:
    def __init__(self, historical: Sequence[float] | np.ndarray, resources: Resources):
        history = _finite(historical)
        self.resources = resources
        self.responsibility = _process_responsibility(history, resources)
        self.positional = PositionalState(history)
        self.universal = UniversalCodeState(history)
        self.absorption = AbsorptionState(history)

    @staticmethod
    def _predict(model: Any, row: np.ndarray) -> float:
        result = np.asarray(model.predict(row.reshape(1, -1), raw_score=False, num_threads=1), dtype=np.float64)
        if result.shape != (1,) or not math.isfinite(float(result[0])):
            raise RuntimeError("V377 booster prediction failed")
        return float(result[0])

    def update(self, value: float, v4_row: np.ndarray, base_probability: float) -> float:
        legacy = 0.5 * sum(self._predict(model, v4_row) for model in self.resources.legacy)
        positional_row = self.positional.update(value)
        positional = self._predict(self.resources.positional, positional_row)
        combined = float(np.clip(
            base_probability
            + LEGACY_SHARE * (legacy - base_probability)
            + POSITIONAL_SHARE * float(self.responsibility[NEAR_IID_CODE]) * (positional - base_probability),
            EPSILON,
            1.0 - EPSILON,
        ))
        universal_row = self.universal.update(value)
        uc_design = _uc_design(universal_row, _logit(base_probability))
        universal_correction = float(np.mean(self.resources.uc_beta @ uc_design))
        score = _logit(combined) + universal_correction
        absorption_row = self.absorption.update(value)
        if self.absorption.count - 1 >= ABSORPTION_START_TICK:
            corrections = []
            for index in range(2):
                design = _absorption_design(
                    absorption_row,
                    self.resources.absorption_center[index],
                    self.resources.absorption_scale[index],
                    score,
                )
                corrections.append(float(self.resources.absorption_beta[index] @ design))
            score += float(np.mean(corrections))
        result = _sigmoid(score)
        if not math.isfinite(result):
            raise RuntimeError("V377 final prediction failed")
        return float(np.clip(result, EPSILON, 1.0 - EPSILON))


__all__ = ["AdditionsState", "Resources", "load_resources", "validate_resources"]
