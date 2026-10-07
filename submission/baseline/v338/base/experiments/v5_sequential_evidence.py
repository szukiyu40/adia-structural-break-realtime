"""Lightweight historical-null sequential evidence for ADIA real time.

This standalone supplement deliberately leaves the v4 and v5 robust
candidates untouched.  It converts five bounded, historical-only calibrated
score streams into two-sided Page--CUSUM evidence at six fixed drift sizes.
The online state is O(5 * 6 + AR_ORDER) per observation and does not grow with
the length of the online stream.

The five score families are:

* raw historical-rank location;
* AR(10)-residual historical-rank location;
* AR(10)-residual log amplitude (scale);
* AR(10)-residual PIT tail surprisal; and
* the lag-one product of consecutive AR(10)-residual rank locations.

Every centering constant, long-run scale, ECDF edge and Page null scale is fit
from the historical segment only.  Online Page states start at zero at the
historical/online boundary.  The only historical state carried across that
boundary is the fixed AR lag buffer and the preceding residual rank needed by
the lag-product score.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Iterator, Sequence

import numpy as np


KAPPAS = (0.03, 0.06, 0.12, 0.24, 0.48, 0.96)
FAMILY_NAMES = (
    "raw_rank_location",
    "ar10_rank_location",
    "ar10_log_scale",
    "ar10_pit_likelihood",
    "ar10_lag_product",
)
SUMMARY_NAMES = (
    "current_positive_logsumexp",
    "current_negative_logsumexp",
    "current_max",
    "current_logsumexp",
    "decayed_max",
    "decayed_logsumexp",
)
FEATURE_NAMES = tuple(
    f"sequential_{family}_{summary}"
    for family in FAMILY_NAMES
    for summary in SUMMARY_NAMES
)

N_BINS = 64
AR_ORDER = 10
MIN_HISTORY = 128
MAX_HAC_LAG = 10
PAGE_NULL_QUANTILE = 0.95
PAGE_NULL_SCALE_FLOOR = 0.50
PAGE_DECAY = 0.995
LOG_AMPLITUDE_OFFSET = 0.125
SCORE_CLIP = 6.0
FEATURE_CLIP = 30.0
EPS = 1e-8

KAPPA_ARRAY = np.asarray(KAPPAS, dtype=np.float64)
FAMILY_COUNT = len(FAMILY_NAMES)
SUMMARY_COUNT = len(SUMMARY_NAMES)
FEATURE_COUNT = len(FEATURE_NAMES)
EXPECTED_FEATURE_COUNT = 30
if (
    FEATURE_COUNT != EXPECTED_FEATURE_COUNT
    or len(set(FEATURE_NAMES)) != FEATURE_COUNT
    or KAPPA_ARRAY.shape != (6,)
):
    raise RuntimeError("sequential evidence feature schema changed")


def _finite_history(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Copy, repair and minimally pad one historical sequence."""

    array = np.array(values, dtype=np.float64, copy=True).reshape(-1)
    finite = np.isfinite(array)
    replacement = float(np.median(array[finite])) if finite.any() else 0.0
    array[~finite] = replacement
    if array.size == 0:
        array = np.asarray([replacement], dtype=np.float64)
    if array.size < MIN_HISTORY:
        array = np.pad(array, (MIN_HISTORY - array.size, 0), mode="edge")
    return array


def _safe_acf(values: np.ndarray, lag: int) -> float:
    if values.size <= lag + 1:
        return 0.0
    left = values[:-lag].astype(np.float64, copy=False)
    right = values[lag:].astype(np.float64, copy=False)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= EPS:
        return 0.0
    return float(left @ right / denominator)


def _fit_ar_yule_walker(values: np.ndarray, order: int) -> np.ndarray:
    if values.size <= order + 2:
        return np.zeros(order, dtype=np.float64)
    acf = np.asarray(
        [1.0] + [_safe_acf(values, lag) for lag in range(1, order + 1)],
        dtype=np.float64,
    )
    toeplitz = np.empty((order, order), dtype=np.float64)
    for row in range(order):
        for column in range(order):
            toeplitz[row, column] = acf[abs(row - column)]
    toeplitz.flat[:: order + 1] += 2e-3
    try:
        coefficients = np.linalg.solve(toeplitz, acf[1:])
    except np.linalg.LinAlgError:
        coefficients = np.zeros(order, dtype=np.float64)
    return np.clip(coefficients, -1.5, 1.5)


def _historical_residuals(
    values: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, float, float]:
    order = int(coefficients.size)
    if values.size <= order:
        residuals = values.copy()
    else:
        prediction = np.zeros(values.size - order, dtype=np.float64)
        for lag, coefficient in enumerate(coefficients, start=1):
            prediction += coefficient * values[order - lag : values.size - lag]
        residuals = values[order:] - prediction
    center = float(residuals.mean()) if residuals.size else 0.0
    ordinary_scale = float(residuals.std(ddof=1)) if residuals.size > 1 else 1.0
    median = float(np.median(residuals)) if residuals.size else 0.0
    mad = float(np.median(np.abs(residuals - median))) if residuals.size else 0.0
    scale = max(ordinary_scale, 1.4826 * mad, EPS)
    return (residuals - center) / scale, center, scale


def _quantile_edges(values: np.ndarray) -> np.ndarray:
    probabilities = np.arange(1, N_BINS, dtype=np.float64) / N_BINS
    return np.quantile(values, probabilities).astype(np.float64, copy=False)


def _rank_locations(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, values, side="right")
    ranks = (bins.astype(np.float64) + 0.5) / N_BINS
    return 2.0 * (ranks - 0.5)


def _rank_location(value: float, edges: np.ndarray) -> float:
    rank = (float(np.searchsorted(edges, value, side="right")) + 0.5) / N_BINS
    return 2.0 * (rank - 0.5)


def _pit_surprisal_from_location(rank_location: np.ndarray | float):
    rank = 0.5 * (np.asarray(rank_location) + 1.0)
    two_sided_tail_probability = np.maximum(
        2.0 * np.minimum(rank, 1.0 - rank), 1.0 / N_BINS
    )
    return -np.log(two_sided_tail_probability)


def _long_run_scale(values: np.ndarray, maximum_lag: int = MAX_HAC_LAG) -> float:
    centered = values - values.mean()
    count = int(centered.size)
    if count < 2:
        return 1.0
    variance = float(centered @ centered / count)
    long_run_variance = variance
    for lag in range(1, min(maximum_lag, count - 1) + 1):
        weight = 1.0 - lag / (min(maximum_lag, count - 1) + 1.0)
        covariance = float(centered[:-lag] @ centered[lag:] / count)
        long_run_variance += 2.0 * weight * covariance
    # Negative sample autocorrelation must not collapse a Page score scale.
    return math.sqrt(max(long_run_variance, 0.25 * variance, EPS))


def _family_base_sequences(
    robust_history: np.ndarray,
    residual_standardized: np.ndarray,
    raw_edges: np.ndarray,
    residual_edges: np.ndarray,
) -> tuple[np.ndarray, ...]:
    raw_location = _rank_locations(robust_history, raw_edges)
    residual_location = _rank_locations(residual_standardized, residual_edges)
    log_scale = np.log(
        np.abs(residual_standardized) + LOG_AMPLITUDE_OFFSET
    )
    pit_likelihood = np.asarray(
        _pit_surprisal_from_location(residual_location), dtype=np.float64
    )
    lag_product = residual_location[1:] * residual_location[:-1]
    if lag_product.size == 0:
        lag_product = np.zeros(1, dtype=np.float64)
    return (
        raw_location,
        residual_location,
        log_scale,
        pit_likelihood,
        lag_product,
    )


def _standardize_historical_families(
    family_values: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    centers = np.empty(FAMILY_COUNT, dtype=np.float64)
    scales = np.empty(FAMILY_COUNT, dtype=np.float64)
    standardized: list[np.ndarray] = []
    for index, values in enumerate(family_values):
        centers[index] = float(values.mean())
        scales[index] = _long_run_scale(values)
        standardized.append(
            np.clip(
                (values - centers[index]) / scales[index],
                -SCORE_CLIP,
                SCORE_CLIP,
            )
        )
    return centers, scales, tuple(standardized)


def _page_null_scales(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-direction Page state scales by historical replay only."""

    positive = np.zeros(KAPPA_ARRAY.size, dtype=np.float64)
    negative = np.zeros(KAPPA_ARRAY.size, dtype=np.float64)
    positive_rows = np.empty((scores.size, KAPPA_ARRAY.size), dtype=np.float64)
    negative_rows = np.empty_like(positive_rows)
    for index, score in enumerate(scores):
        np.maximum(0.0, positive + score - KAPPA_ARRAY, out=positive)
        np.maximum(0.0, negative - score - KAPPA_ARRAY, out=negative)
        positive_rows[index] = positive
        negative_rows[index] = negative
    positive_scale = np.quantile(
        positive_rows, PAGE_NULL_QUANTILE, axis=0
    )
    negative_scale = np.quantile(
        negative_rows, PAGE_NULL_QUANTILE, axis=0
    )
    return (
        np.maximum(positive_scale, PAGE_NULL_SCALE_FLOOR),
        np.maximum(negative_scale, PAGE_NULL_SCALE_FLOOR),
    )


def _logmeanexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return 0.0
    return maximum + math.log(float(np.mean(np.exp(values - maximum))))


class StreamingSequentialEvidence:
    """Causal 30-feature multi-kappa Page state for one ADIA series."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
        history = _finite_history(historical)
        self.historical_center = float(np.median(history))
        mad = float(np.median(np.abs(history - self.historical_center)))
        ordinary_scale = float(history.std(ddof=1)) if history.size > 1 else 1.0
        self.historical_scale = max(1.4826 * mad, 0.10 * ordinary_scale, EPS)
        robust = np.clip(
            (history - self.historical_center) / self.historical_scale,
            -12.0,
            12.0,
        )

        self.ar_coefficients = _fit_ar_yule_walker(robust, AR_ORDER)
        (
            residual_standardized,
            self.residual_center,
            self.residual_scale,
        ) = _historical_residuals(robust, self.ar_coefficients)
        self.raw_edges = _quantile_edges(robust)
        self.residual_edges = _quantile_edges(residual_standardized)

        family_values = _family_base_sequences(
            robust,
            residual_standardized,
            self.raw_edges,
            self.residual_edges,
        )
        (
            self.family_centers,
            self.family_scales,
            standardized_historical,
        ) = _standardize_historical_families(family_values)
        null_scales = [_page_null_scales(values) for values in standardized_historical]
        self.positive_null_scales = np.vstack([value[0] for value in null_scales])
        self.negative_null_scales = np.vstack([value[1] for value in null_scales])

        self.recent_robust: deque[float] = deque(
            (float(value) for value in robust[-AR_ORDER:]), maxlen=AR_ORDER
        )
        historical_residual_location = _rank_locations(
            residual_standardized, self.residual_edges
        )
        self.previous_residual_location = float(historical_residual_location[-1])

        shape = (FAMILY_COUNT, KAPPA_ARRAY.size)
        self.page_positive = np.zeros(shape, dtype=np.float64)
        self.page_negative = np.zeros(shape, dtype=np.float64)
        self.decayed_positive = np.zeros(shape, dtype=np.float64)
        self.decayed_negative = np.zeros(shape, dtype=np.float64)
        self.n = 0

    def _base_scores(self, raw_value: float) -> np.ndarray:
        value = float(raw_value)
        if not math.isfinite(value):
            value = self.historical_center
        robust_value = float(
            np.clip(
                (value - self.historical_center) / self.historical_scale,
                -12.0,
                12.0,
            )
        )
        recent = np.fromiter(
            reversed(self.recent_robust), dtype=np.float64, count=AR_ORDER
        )
        residual = robust_value - float(self.ar_coefficients @ recent)
        residual_standardized = (
            residual - self.residual_center
        ) / self.residual_scale

        raw_location = _rank_location(robust_value, self.raw_edges)
        residual_location = _rank_location(
            residual_standardized, self.residual_edges
        )
        log_scale = math.log(
            abs(residual_standardized) + LOG_AMPLITUDE_OFFSET
        )
        pit_likelihood = float(
            _pit_surprisal_from_location(residual_location)
        )
        lag_product = residual_location * self.previous_residual_location

        self.recent_robust.append(robust_value)
        self.previous_residual_location = residual_location
        return np.asarray(
            [
                raw_location,
                residual_location,
                log_scale,
                pit_likelihood,
                lag_product,
            ],
            dtype=np.float64,
        )

    def _summaries(self) -> np.ndarray:
        normalized_positive = np.clip(
            self.page_positive / self.positive_null_scales,
            0.0,
            FEATURE_CLIP,
        )
        normalized_negative = np.clip(
            self.page_negative / self.negative_null_scales,
            0.0,
            FEATURE_CLIP,
        )
        normalized_decayed_positive = np.clip(
            self.decayed_positive / self.positive_null_scales,
            0.0,
            FEATURE_CLIP,
        )
        normalized_decayed_negative = np.clip(
            self.decayed_negative / self.negative_null_scales,
            0.0,
            FEATURE_CLIP,
        )

        output = np.empty((FAMILY_COUNT, SUMMARY_COUNT), dtype=np.float64)
        for family in range(FAMILY_COUNT):
            current_both = np.concatenate(
                [normalized_positive[family], normalized_negative[family]]
            )
            decayed_both = np.concatenate(
                [
                    normalized_decayed_positive[family],
                    normalized_decayed_negative[family],
                ]
            )
            output[family] = (
                _logmeanexp(normalized_positive[family]),
                _logmeanexp(normalized_negative[family]),
                float(current_both.max()),
                _logmeanexp(current_both),
                float(decayed_both.max()),
                _logmeanexp(decayed_both),
            )
        return output.ravel()

    def update(self, raw_value: float) -> np.ndarray:
        base = self._base_scores(raw_value)
        scores = np.clip(
            (base - self.family_centers) / self.family_scales,
            -SCORE_CLIP,
            SCORE_CLIP,
        )
        expanded = scores[:, None]
        np.maximum(
            0.0,
            self.page_positive + expanded - KAPPA_ARRAY[None, :],
            out=self.page_positive,
        )
        np.maximum(
            0.0,
            self.page_negative - expanded - KAPPA_ARRAY[None, :],
            out=self.page_negative,
        )
        np.maximum(
            self.page_positive,
            PAGE_DECAY * self.decayed_positive,
            out=self.decayed_positive,
        )
        np.maximum(
            self.page_negative,
            PAGE_DECAY * self.decayed_negative,
            out=self.decayed_negative,
        )
        self.n += 1

        result = np.asarray(self._summaries(), dtype=np.float32)
        if result.shape != (FEATURE_COUNT,):
            raise RuntimeError(
                f"sequential evidence width changed: {result.size} != {FEATURE_COUNT}"
            )
        return np.nan_to_num(
            result,
            copy=False,
            nan=0.0,
            posinf=FEATURE_CLIP,
            neginf=0.0,
        )


def iter_features(
    historical: Sequence[float] | np.ndarray,
    online: Iterable[float],
) -> Iterator[np.ndarray]:
    """Yield the sole causal feature path used by streaming and batch calls."""

    state = StreamingSequentialEvidence(historical)
    for point in online:
        yield state.update(point)


def batch_features(
    historical: Sequence[float] | np.ndarray,
    online: Iterable[float],
) -> np.ndarray:
    """Materialise ``iter_features`` without introducing a second algorithm."""

    rows = list(iter_features(historical, online))
    if not rows:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)
