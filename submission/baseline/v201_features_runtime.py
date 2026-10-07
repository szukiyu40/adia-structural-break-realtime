"""Causal history-regime and multiscale structural-operator features for V201.

This module belongs only to the ADIA Structural Break Challenge: Real-Time
Edition.  Every online feature at tick ``t`` uses the immutable history and the
same series' online prefix through ``t``.  It never uses the online horizon,
future ticks, or another series.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
SCHEMA_VERSION = "adia-v201-history-regime-operator-v1"
EPSILON = 1e-8
WINDOWS = (16, 32, 64, 128, 256)
END_STEP = 32
OPERATORS = ("location", "scale", "dependence", "voltail", "trend", "shape")

HISTORY_FEATURE_NAMES = (
    "regime_acf1",
    "regime_acf5",
    "regime_abs_acf1",
    "regime_sq_acf1",
    "regime_ar_instability",
    "regime_vol_dispersion",
    "regime_tail_ratio_log",
    "regime_moors_log",
    "regime_quiet_share",
    "regime_outer_share",
    "regime_quiet_run_share",
    "regime_trend_strength",
    "regime_variance_ratio8_log",
    "regime_low_frequency_share",
    "regime_histogram_entropy",
)

OPERATOR_FEATURE_NAMES = tuple(
    name
    for operator in OPERATORS
    for name in (
        f"operator_{operator}_suffix_log",
        f"operator_{operator}_episode_log",
        f"operator_{operator}_scale_support",
        f"operator_{operator}_episode_gap",
        f"operator_{operator}_best_end_fraction",
        f"operator_{operator}_best_window_fraction",
    )
)

INTERACTION_SPECS = (
    ("location", "episode", "regime_acf1"),
    ("location", "episode", "regime_trend_strength"),
    ("scale", "episode", "regime_vol_dispersion"),
    ("scale", "episode", "regime_tail_ratio_log"),
    ("dependence", "episode", "regime_acf1"),
    ("dependence", "episode", "regime_ar_instability"),
    ("voltail", "episode", "regime_vol_dispersion"),
    ("voltail", "episode", "regime_tail_ratio_log"),
    ("trend", "episode", "regime_trend_strength"),
    ("trend", "episode", "regime_low_frequency_share"),
    ("shape", "episode", "regime_moors_log"),
    ("shape", "episode", "regime_histogram_entropy"),
)

INTERACTION_FEATURE_NAMES = tuple(
    f"interaction_{operator}_{scope}_x_{regime.removeprefix('regime_')}"
    for operator, scope, regime in INTERACTION_SPECS
)

POSTERIOR_FEATURE_NAMES = (
    "posterior_suffix_max",
    "posterior_suffix_margin",
    "posterior_suffix_entropy",
    "posterior_suffix_effective_count",
    "posterior_episode_max",
    "posterior_episode_margin",
    "posterior_episode_entropy",
    "posterior_episode_effective_count",
    "posterior_suffix_episode_agreement",
)

FEATURE_NAMES = (
    HISTORY_FEATURE_NAMES
    + OPERATOR_FEATURE_NAMES
    + INTERACTION_FEATURE_NAMES
    + POSTERIOR_FEATURE_NAMES
)

FEATURE_GROUPS = (
    tuple("history_regime" for _ in HISTORY_FEATURE_NAMES)
    + tuple(
        f"operator_{name.split('_')[1]}"
        for name in OPERATOR_FEATURE_NAMES
    )
    + tuple("regime_operator_interaction" for _ in INTERACTION_FEATURE_NAMES)
    + tuple("operator_posterior" for _ in POSTERIOR_FEATURE_NAMES)
)

if (
    len(FEATURE_NAMES) != 72
    or len(FEATURE_GROUPS) != len(FEATURE_NAMES)
    or len(set(FEATURE_NAMES)) != len(FEATURE_NAMES)
):
    raise RuntimeError("V201 feature schema changed")


def _clean(values: Sequence[float] | np.ndarray, replacement: float | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if array.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    finite = np.isfinite(array)
    if replacement is None:
        replacement = float(np.median(array[finite])) if finite.any() else 0.0
    array[~finite] = replacement
    return array


def _robust_standardize(history: np.ndarray, online: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = float(np.median(history))
    q25, q75 = np.quantile(history, (0.25, 0.75), method="linear")
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-8:
        mad = float(np.median(np.abs(history - center)))
        scale = mad / 0.6744897501960817 if mad > 1e-8 else float(np.std(history))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    historical = np.clip((history - center) / scale, -12.0, 12.0)
    observed = np.clip((online - center) / scale, -12.0, 12.0)
    return historical, observed


def _acf(values: np.ndarray, lag: int = 1) -> float:
    if values.size <= lag + 2:
        return 0.0
    left = values[:-lag] - float(np.mean(values[:-lag]))
    right = values[lag:] - float(np.mean(values[lag:]))
    denominator = math.sqrt(max(float(np.dot(left, left) * np.dot(right, right)), 0.0))
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)) if denominator > EPSILON else 0.0


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not np.any(mask):
        return 0
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(boundaries[1::2] - boundaries[::2]))


def _slope_strength(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    time = np.arange(values.size, dtype=np.float64)
    time -= float(np.mean(time))
    centered = values - float(np.mean(values))
    denominator = max(float(np.dot(time, time)), EPSILON)
    slope = float(np.dot(time, centered) / denominator)
    residual = centered - slope * time
    noise = math.sqrt(max(float(np.mean(residual * residual)), EPSILON))
    return abs(slope) * math.sqrt(denominator) / noise / math.sqrt(values.size)


def _variance_ratio(values: np.ndarray, lag: int) -> float:
    if values.size <= lag + 4:
        return 1.0
    one = np.diff(values)
    distant = values[lag:] - values[:-lag]
    denominator = lag * max(float(np.var(one)), EPSILON)
    return max(float(np.var(distant)) / denominator, EPSILON)


def _low_frequency_share(values: np.ndarray) -> float:
    centered = values - float(np.mean(values))
    spectrum = np.square(np.abs(np.fft.rfft(centered)))
    if spectrum.size <= 2 or float(np.sum(spectrum[1:])) <= EPSILON:
        return 0.0
    upper = max(2, int(math.ceil(0.10 * spectrum.size)))
    return float(np.sum(spectrum[1:upper]) / np.sum(spectrum[1:]))


def _histogram_entropy(values: np.ndarray) -> float:
    counts, _ = np.histogram(np.clip(values, -4.0, 4.0), bins=16, range=(-4.0, 4.0))
    probabilities = counts[counts > 0].astype(np.float64)
    if probabilities.size <= 1:
        return 0.0
    probabilities /= float(np.sum(probabilities))
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(16.0))


def _block_log_scale_dispersion(values: np.ndarray, width: int = 64) -> float:
    scales = []
    for first in range(0, values.size - width + 1, width):
        block = values[first : first + width]
        q25, q75 = np.quantile(block, (0.25, 0.75), method="linear")
        scales.append(math.log(max(float((q75 - q25) / 1.349), 1e-4)))
    return float(np.std(scales)) if len(scales) >= 2 else 0.0


def _history_reference(history: np.ndarray) -> dict[str, np.ndarray | float]:
    mean = float(np.mean(history))
    variance = max(float(np.var(history)), EPSILON)
    standard_deviation = math.sqrt(variance)
    quantile_levels = (0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95)
    quantiles = np.quantile(history, quantile_levels, method="linear")
    q05, q125, q25, q375, median, q625, q75, q875, q95 = quantiles
    robust_scale = max(float((q75 - q25) / 1.349), EPSILON)
    normalized = (history - mean) / standard_deviation
    abs_values = np.abs(normalized)
    moors = max(float(((q875 - q625) + (q375 - q125)) / max(q75 - q25, EPSILON)), EPSILON)
    tail_ratio = max(float((q95 - q05) / max(q75 - q25, EPSILON)), EPSILON)
    expected_shortfall = float(np.mean(np.sort(abs_values)[-max(1, history.size // 10) :]))
    return {
        "mean": mean,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "median": float(median),
        "robust_scale": robust_scale,
        "normalized_quantiles": (quantiles - mean) / standard_deviation,
        "acf1": _acf(normalized, 1),
        "acf5": _acf(normalized, 5),
        "abs_acf1": _acf(abs_values, 1),
        "sq_acf1": _acf(normalized * normalized, 1),
        "tail_ratio": tail_ratio,
        "moors": moors,
        "quiet_share": float(np.mean(abs_values < 0.5)),
        "outer_share": float(np.mean(abs_values > 2.0)),
        "expected_shortfall": expected_shortfall,
    }


def history_regime_features(history: Sequence[float] | np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    historical = _clean(history)
    historical, _ = _robust_standardize(historical, np.asarray([0.0], dtype=np.float64))
    reference = _history_reference(historical)
    midpoint = historical.size // 2
    first = historical[:midpoint]
    second = historical[midpoint:]
    normalized = (historical - float(reference["mean"])) / float(reference["standard_deviation"])
    abs_values = np.abs(normalized)
    values = np.asarray(
        (
            float(reference["acf1"]),
            float(reference["acf5"]),
            float(reference["abs_acf1"]),
            float(reference["sq_acf1"]),
            abs(_acf(first, 1) - _acf(second, 1)),
            _block_log_scale_dispersion(historical),
            math.log(float(reference["tail_ratio"])),
            math.log(float(reference["moors"])),
            float(reference["quiet_share"]),
            float(reference["outer_share"]),
            _longest_run(abs_values < 0.5) / historical.size,
            _slope_strength(historical),
            math.log(_variance_ratio(historical, 8)),
            _low_frequency_share(historical),
            _histogram_entropy(normalized),
        ),
        dtype=np.float64,
    )
    if values.shape != (len(HISTORY_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise RuntimeError("invalid V201 history regime row")
    return np.clip(values, -12.0, 12.0), reference


def _step_strength(values: np.ndarray) -> float:
    if values.size < 16:
        return 0.0
    overall_variance = max(float(np.var(values)), EPSILON)
    best = 0.0
    minimum = max(4, values.size // 8)
    step = max(4, values.size // 16)
    for cut in range(minimum, values.size - minimum + 1, step):
        left = values[:cut]
        right = values[cut:]
        contrast = abs(float(np.mean(right) - np.mean(left)))
        weight = math.sqrt(cut * (values.size - cut) / values.size)
        best = max(best, contrast * weight / math.sqrt(overall_variance))
    return best


def _path_area(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    centered = values - float(np.mean(values))
    cumulative = np.cumsum(centered)
    scale = math.sqrt(values.size * max(float(np.var(values)), EPSILON))
    return float(np.mean(np.abs(cumulative)) / max(scale, EPSILON))


def _segment_operator_scores(segment: np.ndarray, reference: dict[str, np.ndarray | float]) -> np.ndarray:
    length = segment.size
    mean = float(np.mean(segment))
    variance = max(float(np.var(segment)), EPSILON)
    standard_deviation = math.sqrt(variance)
    q = np.quantile(segment, (0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95), method="linear")
    q05, q125, q25, q375, median, q625, q75, q875, q95 = q
    robust_scale = max(float((q75 - q25) / 1.349), EPSILON)
    normalized = (segment - mean) / standard_deviation
    abs_values = np.abs(normalized)

    location = math.sqrt(length) * math.sqrt(
        ((mean - float(reference["mean"])) / float(reference["standard_deviation"])) ** 2
        + 0.5 * ((float(median) - float(reference["median"])) / float(reference["robust_scale"])) ** 2
    )
    scale = math.sqrt(length / 2.0) * math.sqrt(
        math.log(variance / float(reference["variance"])) ** 2
        + 0.5 * math.log(robust_scale / float(reference["robust_scale"])) ** 2
    )
    dependence = math.sqrt(length) * math.sqrt(
        (_acf(normalized, 1) - float(reference["acf1"])) ** 2
        + 0.5 * (_acf(normalized, 5) - float(reference["acf5"])) ** 2
    )
    tail_ratio = max(float((q95 - q05) / max(q75 - q25, EPSILON)), EPSILON)
    expected_shortfall = float(np.mean(np.sort(abs_values)[-max(1, length // 10) :]))
    voltail = math.sqrt(length) * math.sqrt(
        (_acf(abs_values, 1) - float(reference["abs_acf1"])) ** 2
        + 0.5 * (_acf(normalized * normalized, 1) - float(reference["sq_acf1"])) ** 2
        + 0.25 * math.log(tail_ratio / float(reference["tail_ratio"])) ** 2
        + 0.25 * (float(np.mean(abs_values < 0.5)) - float(reference["quiet_share"])) ** 2
        + 0.25 * (float(np.mean(abs_values > 2.0)) - float(reference["outer_share"])) ** 2
        + 0.10 * (expected_shortfall - float(reference["expected_shortfall"])) ** 2
    )
    trend = max(_step_strength(segment), _slope_strength(segment), 2.0 * _path_area(segment))
    normalized_quantiles = (q - mean) / standard_deviation
    moors = max(float(((q875 - q625) + (q375 - q125)) / max(q75 - q25, EPSILON)), EPSILON)
    shape = math.sqrt(length) * math.sqrt(
        float(np.mean(np.square(normalized_quantiles - np.asarray(reference["normalized_quantiles"]))))
        + 0.25 * math.log(moors / float(reference["moors"])) ** 2
        + 0.25 * (float(np.mean(abs_values < 0.5)) - float(reference["quiet_share"])) ** 2
        + 0.25 * (float(np.mean(abs_values > 2.0)) - float(reference["outer_share"])) ** 2
    )
    scores = np.asarray((location, scale, dependence, voltail, trend, shape), dtype=np.float64)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)


def _operator_summary(prefix: np.ndarray, reference: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = prefix.size
    suffix_by_window: list[np.ndarray] = []
    episode = np.zeros(len(OPERATORS), dtype=np.float64)
    best_end = np.zeros(len(OPERATORS), dtype=np.int32)
    best_window = np.full(len(OPERATORS), WINDOWS[0], dtype=np.int32)
    endpoints = list(range(WINDOWS[0], count + 1, END_STEP))
    if not endpoints or endpoints[-1] != count:
        endpoints.append(count)
    for end in endpoints:
        for width in WINDOWS:
            if width > end:
                continue
            scores = _segment_operator_scores(prefix[end - width : end], reference)
            improved = scores > episode
            episode[improved] = scores[improved]
            best_end[improved] = end
            best_window[improved] = width
            if end == count:
                suffix_by_window.append(scores)
    if not suffix_by_window:
        suffix_by_window.append(_segment_operator_scores(prefix, reference))
    suffix_matrix = np.vstack(suffix_by_window)
    suffix = np.max(suffix_matrix, axis=0)
    support = np.mean(
        suffix_matrix >= (0.75 * np.maximum(suffix, EPSILON))[None, :], axis=0
    )
    logged_suffix = np.log1p(suffix)
    logged_episode = np.log1p(episode)
    rows = []
    for index in range(len(OPERATORS)):
        rows.extend(
            (
                logged_suffix[index],
                logged_episode[index],
                support[index],
                logged_episode[index] - logged_suffix[index],
                best_end[index] / count,
                best_window[index] / count,
            )
        )
    return np.asarray(rows, dtype=np.float64), logged_suffix, logged_episode


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exponent = np.exp(np.clip(shifted, -40.0, 0.0))
    return exponent / max(float(np.sum(exponent)), EPSILON)


def _posterior_features(suffix: np.ndarray, episode: np.ndarray) -> np.ndarray:
    suffix_probability = _softmax(suffix)
    episode_probability = _softmax(episode)

    def summarize(probability: np.ndarray) -> tuple[float, float, float, float]:
        ordered = np.sort(probability)
        entropy = -float(np.sum(probability * np.log(np.maximum(probability, EPSILON)))) / math.log(probability.size)
        effective = 1.0 / float(np.sum(probability * probability))
        return float(ordered[-1]), float(ordered[-1] - ordered[-2]), entropy, effective

    output = np.asarray(
        (*summarize(suffix_probability), *summarize(episode_probability), float(np.dot(suffix_probability, episode_probability))),
        dtype=np.float64,
    )
    return output


def checkpoint_features(
    history: Sequence[float] | np.ndarray,
    online: Sequence[float] | np.ndarray,
    checkpoints: Iterable[int],
) -> dict[int, np.ndarray]:
    raw_history = _clean(history)
    raw_online = _clean(online, float(np.median(raw_history)))
    historical, observed = _robust_standardize(raw_history, raw_online)
    history_values, reference = history_regime_features(raw_history)
    history_lookup = dict(zip(HISTORY_FEATURE_NAMES, history_values, strict=True))
    result: dict[int, np.ndarray] = {}
    for checkpoint in sorted(set(int(value) for value in checkpoints)):
        if checkpoint < WINDOWS[0] - 1 or checkpoint >= observed.size:
            continue
        prefix = observed[: checkpoint + 1]
        operator_values, suffix, episode = _operator_summary(prefix, reference)
        operator_lookup = {
            (operator, "suffix"): suffix[index]
            for index, operator in enumerate(OPERATORS)
        }
        operator_lookup.update(
            {
                (operator, "episode"): episode[index]
                for index, operator in enumerate(OPERATORS)
            }
        )
        interactions = np.asarray(
            [
                math.tanh(operator_lookup[(operator, scope)] / 4.0)
                * math.tanh(history_lookup[regime])
                for operator, scope, regime in INTERACTION_SPECS
            ],
            dtype=np.float64,
        )
        values = np.concatenate(
            (history_values, operator_values, interactions, _posterior_features(suffix, episode))
        )
        values = np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)
        values = np.clip(values, -12.0, 12.0).astype(np.float32)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid V201 checkpoint row at tick {checkpoint}")
        result[checkpoint] = values
    return result
