"""Finite-sample calibrated streaming detector for the ADIA challenge.

This experimental candidate calibrates every rolling statistic against the
same series' historical segment.  The main feature bank uses non-overlapping
historical blocks and robust median/MAD standardisation.  A compact analytic
feature additionally reproduces the strongest forensic detector: the mean of
the eight largest historical-block percentiles over raw and AR(10) channels.

All online features are causal.  A window at the beginning of the online
segment is left-padded with the final historical observations, never with
future online values.  Training and inference both call ``iter_features``.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np


WINDOWS = (8, 16, 32, 64, 128, 256)
LAGS = (1, 2, 5, 10)
AR_ORDERS = (5, 10)
N_BINS = 16
SAMPLE_MODULUS = 5
SAMPLE_PROBABILITY = 1.0 / SAMPLE_MODULUS
MAX_NULL_BLOCKS = 64
MAX_ECDF_BLOCKS = 80
MODEL_FILENAME = "model.txt"
EPS = 1e-8
CLIP_VALUE = 12.0

# Statistics returned for a complete window.  The raw and AR(10) channels use
# all values.  AR(5) keeps the location/scale and dependence subset: its PIT
# and shape statistics were weak in the forensic screen and would add both
# memory and online work without useful independent signal.
ALL_STAT_NAMES = (
    "mean",
    "log_variance",
    "skew",
    "excess_kurtosis",
    "pit_chi",
    "pit_ks",
    "pit_edge_rate",
    "log_diff_variance",
    "acf_1",
    "acf_2",
    "acf_5",
    "acf_10",
    "abs_acf_1",
    "abs_acf_2",
    "abs_acf_5",
    "abs_acf_10",
)
RESIDUAL5_STAT_INDICES = (0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15)
PER_WINDOW_STAT_COUNT = len(ALL_STAT_NAMES) * 2 + len(RESIDUAL5_STAT_INDICES)
META_FEATURE_COUNT = 4
PER_WINDOW_FEATURE_COUNT = 1 + PER_WINDOW_STAT_COUNT  # block count + statistics
ANALYTIC_FEATURE_COUNT = 1
FEATURE_COUNT = (
    META_FEATURE_COUNT
    + len(WINDOWS) * PER_WINDOW_FEATURE_COUNT
    + ANALYTIC_FEATURE_COUNT
)


def _clip_scalar(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


def _safe_acf(values: np.ndarray, lag: int, transform_abs: bool = False) -> float:
    if values.size <= lag + 1:
        return 0.0
    x = np.abs(values) if transform_abs else values
    left = x[:-lag].astype(np.float64, copy=False)
    right = x[lag:].astype(np.float64, copy=False)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= EPS:
        return 0.0
    return float(left @ right / denominator)


def _fit_ar_yule_walker(values: np.ndarray, order: int) -> np.ndarray:
    """Fit a fixed, regularised AR model using historical values only."""

    if values.size <= order + 2:
        return np.zeros(order, dtype=np.float64)
    acf = np.asarray([1.0] + [_safe_acf(values, lag) for lag in range(1, order + 1)])
    matrix = np.empty((order, order), dtype=np.float64)
    for row in range(order):
        for column in range(order):
            matrix[row, column] = acf[abs(row - column)]
    matrix.flat[:: order + 1] += 2e-3
    try:
        coefficients = np.linalg.solve(matrix, acf[1:])
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
    scale = float(residuals.std(ddof=1)) if residuals.size > 1 else 1.0
    scale = max(scale, EPS)
    standardized = np.clip((residuals - center) / scale, -CLIP_VALUE, CLIP_VALUE)
    return standardized, center, scale


def _quantile_edges(values: np.ndarray) -> np.ndarray:
    probabilities = np.arange(1, N_BINS, dtype=np.float64) / N_BINS
    return np.quantile(values, probabilities).astype(np.float64, copy=False)


def _block_starts(
    length: int, window: int, stride: int, maximum: int
) -> np.ndarray:
    """Return evenly retained starts from a regular historical block grid."""

    if length < window:
        return np.asarray([], dtype=np.int64)
    starts = np.arange(0, length - window + 1, stride, dtype=np.int64)
    if starts.size <= maximum:
        return starts
    retained = np.rint(np.linspace(0, starts.size - 1, maximum)).astype(np.int64)
    return starts[retained]


def _non_overlapping_starts(length: int, window: int) -> np.ndarray:
    return _block_starts(length, window, window, MAX_NULL_BLOCKS)


def _tail_with_left_padding(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    if values.size >= length:
        return values[-length:].astype(np.float64, copy=False)
    if values.size == 0:
        return np.zeros(length, dtype=np.float64)
    return np.pad(values, (length - values.size, 0), mode="edge").astype(
        np.float64, copy=False
    )


def _blocks_from_starts(
    values: np.ndarray, starts: np.ndarray, window: int
) -> np.ndarray:
    if starts.size == 0:
        return _tail_with_left_padding(values, window).reshape(1, window)
    offsets = np.arange(window, dtype=np.int64)
    return values[starts[:, None] + offsets[None, :]]


def _statistics_for_blocks(
    values: np.ndarray,
    edges: np.ndarray,
    window: int,
    starts: np.ndarray,
) -> np.ndarray:
    """Vectorised counterpart of ``_RollingChannels._channel_statistics``."""

    blocks = _blocks_from_starts(values, starts, window).astype(np.float64, copy=False)
    count = blocks.shape[0]
    mean = blocks.mean(axis=1)
    raw2 = np.mean(blocks * blocks, axis=1)
    raw3 = np.mean(blocks * blocks * blocks, axis=1)
    raw4 = np.mean((blocks * blocks) ** 2, axis=1)
    variance = np.maximum(raw2 - mean * mean, EPS)
    third = raw3 - 3.0 * mean * raw2 + 2.0 * mean**3
    fourth = raw4 - 4.0 * mean * raw3 + 6.0 * mean * mean * raw2 - 3.0 * mean**4
    skew = third / np.maximum(variance**1.5, EPS)
    kurtosis = fourth / np.maximum(variance * variance, EPS) - 3.0

    bins = np.searchsorted(edges, blocks, side="right")
    offsets = (np.arange(count, dtype=np.int64) * N_BINS)[:, None]
    pit_counts = np.bincount(
        (bins + offsets).ravel(), minlength=count * N_BINS
    ).reshape(count, N_BINS)
    expected = window / N_BINS
    differences = pit_counts.astype(np.float64) - expected
    chi_square = np.sum(differences * differences / max(expected, EPS), axis=1)
    pit_chi = (chi_square - (N_BINS - 1)) / math.sqrt(2.0 * (N_BINS - 1))
    cumulative = np.cumsum(pit_counts, axis=1, dtype=np.float64) / window
    reference = np.arange(1, N_BINS + 1, dtype=np.float64) / N_BINS
    pit_ks = np.max(np.abs(cumulative - reference[None, :]), axis=1)
    pit_edge = (pit_counts[:, 0] + pit_counts[:, -1]) / window

    differences_values = np.diff(blocks, axis=1)
    if differences_values.shape[1] > 0:
        diff_variance = np.var(differences_values, axis=1)
    else:
        diff_variance = np.ones(count, dtype=np.float64)
    log_diff_variance = np.log(np.maximum(diff_variance, EPS))

    absolute = np.abs(blocks)
    abs_mean = absolute.mean(axis=1)
    abs_variance = np.maximum(np.mean(absolute * absolute, axis=1) - abs_mean**2, EPS)
    acfs: list[np.ndarray] = []
    abs_acfs: list[np.ndarray] = []
    for lag in LAGS:
        if window <= lag + 1:
            acfs.append(np.zeros(count, dtype=np.float64))
            abs_acfs.append(np.zeros(count, dtype=np.float64))
            continue
        cross = np.mean(blocks[:, :-lag] * blocks[:, lag:], axis=1)
        acfs.append((cross - mean * mean) / variance)
        abs_cross = np.mean(absolute[:, :-lag] * absolute[:, lag:], axis=1)
        abs_acfs.append((abs_cross - abs_mean * abs_mean) / abs_variance)

    return np.column_stack(
        [
            mean,
            np.log(variance),
            skew,
            kurtosis,
            pit_chi,
            pit_ks,
            pit_edge,
            log_diff_variance,
            *acfs,
            *abs_acfs,
        ]
    )


def _select_feature_statistics(
    raw: np.ndarray, residual5: np.ndarray, residual10: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [raw, residual5[..., RESIDUAL5_STAT_INDICES], residual10], axis=-1
    )


def _robust_location_scale(block_statistics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(block_statistics, axis=0)
    mad = np.median(np.abs(block_statistics - median[None, :]), axis=0)
    robust_scale = 1.4826 * mad
    ordinary_scale = (
        np.std(block_statistics, axis=0, ddof=1)
        if block_statistics.shape[0] > 1
        else np.zeros(block_statistics.shape[1], dtype=np.float64)
    )
    # The ordinary-scale floor stabilises the few blocks available for W=256
    # while median/MAD remains the primary finite-sample calibration.
    scale = np.maximum(robust_scale, 0.25 * ordinary_scale)
    scale = np.where(scale > 1e-5, scale, 1.0)
    return median.astype(np.float64), scale.astype(np.float64)


def _reference_statistics(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    starts = np.asarray([0], dtype=np.int64)
    return _statistics_for_blocks(values, edges, int(values.size), starts)[0]


def _analytic_effects(
    statistics: np.ndarray, reference: np.ndarray, window: int
) -> np.ndarray:
    """Eight two-sided forensic statistics for one channel/window."""

    stats = np.atleast_2d(statistics)
    root = math.sqrt(window)
    effects = np.column_stack(
        [
            root * np.abs(stats[:, 0] - reference[0]),
            math.sqrt(window / 2.0) * np.abs(stats[:, 1] - reference[1]),
            math.sqrt(window / 6.0) * np.abs(stats[:, 2] - reference[2]),
            math.sqrt(window / 24.0) * np.abs(stats[:, 3] - reference[3]),
            root * np.maximum(stats[:, 5], 0.0),
            root * np.max(np.abs(stats[:, 8:11] - reference[None, 8:11]), axis=1),
            root * np.abs(stats[:, 12] - reference[12]),
            math.sqrt(max(window - 1, 1) / 2.0)
            * np.abs(stats[:, 7] - reference[7]),
        ]
    )
    return effects if np.ndim(statistics) > 1 else effects[0]


def _analytic_effects_scalar(
    statistics: np.ndarray, reference: np.ndarray, window: int
) -> tuple[float, float, float, float, float, float, float, float]:
    """Allocation-light scalar path used at every online step."""

    root = math.sqrt(window)
    maximum_acf = max(
        abs(float(statistics[index] - reference[index])) for index in (8, 9, 10)
    )
    return (
        root * abs(float(statistics[0] - reference[0])),
        math.sqrt(window / 2.0) * abs(float(statistics[1] - reference[1])),
        math.sqrt(window / 6.0) * abs(float(statistics[2] - reference[2])),
        math.sqrt(window / 24.0) * abs(float(statistics[3] - reference[3])),
        root * max(float(statistics[5]), 0.0),
        root * maximum_acf,
        root * abs(float(statistics[12] - reference[12])),
        math.sqrt(max(window - 1, 1) / 2.0)
        * abs(float(statistics[7] - reference[7])),
    )


class _RollingChannels:
    """O(1) moments, PIT counts, differences and multi-lag dependence."""

    __slots__ = (
        "size",
        "values",
        "sum1",
        "sum2",
        "sum3",
        "sum4",
        "sum_abs",
        "sum_abs2",
        "sum_diff",
        "sum_diff2",
        "cross",
        "abs_cross",
        "pit_counts",
    )

    def __init__(self, size: int):
        self.size = int(size)
        self.values: deque[tuple[float, float, float, int, int]] = deque()
        # Tiny fixed-size accumulators are faster as Python lists than as many
        # scalar NumPy indexing operations in this 10M-step hot path.
        self.sum1 = [0.0] * 3
        self.sum2 = [0.0] * 3
        self.sum3 = [0.0] * 3
        self.sum4 = [0.0] * 3
        self.sum_abs = [0.0] * 3
        self.sum_abs2 = [0.0] * 3
        self.sum_diff = [0.0] * 3
        self.sum_diff2 = [0.0] * 3
        self.cross = [[0.0] * len(LAGS) for _ in range(3)]
        self.abs_cross = [[0.0] * len(LAGS) for _ in range(3)]
        # Only raw and AR(10) need PIT statistics online.
        self.pit_counts = [[0] * N_BINS for _ in range(2)]

    def update(self, point: tuple[float, float, float, int, int]) -> None:
        queue = self.values
        if len(queue) == self.size:
            old = queue[0]
            old_values = old[:3]
            if len(queue) > 1:
                second = queue[1]
                for channel in range(3):
                    difference = second[channel] - old_values[channel]
                    self.sum_diff[channel] -= difference
                    self.sum_diff2[channel] -= difference * difference
            for lag_index, lag in enumerate(LAGS):
                if len(queue) > lag:
                    other = queue[lag]
                    for channel in range(3):
                        self.cross[channel][lag_index] -= (
                            old_values[channel] * other[channel]
                        )
                        self.abs_cross[channel][lag_index] -= (
                            abs(old_values[channel]) * abs(other[channel])
                        )
            queue.popleft()
            for channel in range(3):
                value = old_values[channel]
                square = value * value
                self.sum1[channel] -= value
                self.sum2[channel] -= square
                self.sum3[channel] -= square * value
                self.sum4[channel] -= square * square
                self.sum_abs[channel] -= abs(value)
                self.sum_abs2[channel] -= abs(value) ** 2
            self.pit_counts[0][old[3]] -= 1
            self.pit_counts[1][old[4]] -= 1

        if queue:
            previous = queue[-1]
            for channel in range(3):
                difference = point[channel] - previous[channel]
                self.sum_diff[channel] += difference
                self.sum_diff2[channel] += difference * difference
        for lag_index, lag in enumerate(LAGS):
            if len(queue) >= lag:
                previous = queue[-lag]
                for channel in range(3):
                    self.cross[channel][lag_index] += previous[channel] * point[channel]
                    self.abs_cross[channel][lag_index] += (
                        abs(previous[channel]) * abs(point[channel])
                    )

        queue.append(point)
        for channel in range(3):
            value = point[channel]
            square = value * value
            self.sum1[channel] += value
            self.sum2[channel] += square
            self.sum3[channel] += square * value
            self.sum4[channel] += square * square
            self.sum_abs[channel] += abs(value)
            self.sum_abs2[channel] += abs(value) ** 2
        self.pit_counts[0][point[3]] += 1
        self.pit_counts[1][point[4]] += 1

    def _channel_statistics(self, channel: int, pit_row: Optional[int]) -> list[float]:
        n = len(self.values)
        if n < 2:
            return [0.0] * len(ALL_STAT_NAMES)
        mean = self.sum1[channel] / n
        raw2 = self.sum2[channel] / n
        variance = max(raw2 - mean * mean, EPS)
        raw3 = self.sum3[channel] / n
        raw4 = self.sum4[channel] / n
        third = raw3 - 3.0 * mean * raw2 + 2.0 * mean**3
        fourth = raw4 - 4.0 * mean * raw3 + 6.0 * mean * mean * raw2 - 3.0 * mean**4
        skew = third / max(variance**1.5, EPS)
        kurtosis = fourth / max(variance * variance, EPS) - 3.0

        pit_chi = 0.0
        pit_ks = 0.0
        pit_edge = 0.0
        if pit_row is not None:
            counts = self.pit_counts[pit_row]
            expected = n / N_BINS
            chi_square = 0.0
            cumulative = 0
            for bin_index, count in enumerate(counts):
                difference = float(count) - expected
                chi_square += difference * difference / max(expected, EPS)
                cumulative += int(count)
                discrepancy = abs(cumulative / n - (bin_index + 1) / N_BINS)
                pit_ks = max(pit_ks, discrepancy)
            pit_chi = (chi_square - (N_BINS - 1)) / math.sqrt(
                2.0 * (N_BINS - 1)
            )
            pit_edge = float(counts[0] + counts[-1]) / n

        pair_count = n - 1
        diff_mean = self.sum_diff[channel] / pair_count
        diff_variance = max(
            self.sum_diff2[channel] / pair_count - diff_mean * diff_mean, EPS
        )

        abs_mean = self.sum_abs[channel] / n
        abs_variance = max(
            self.sum_abs2[channel] / n - abs_mean * abs_mean, EPS
        )
        acfs: list[float] = []
        abs_acfs: list[float] = []
        for lag_index, lag in enumerate(LAGS):
            if n <= lag + 1:
                acfs.append(0.0)
                abs_acfs.append(0.0)
                continue
            cross = self.cross[channel][lag_index] / (n - lag)
            acfs.append((cross - mean * mean) / variance)
            abs_cross = self.abs_cross[channel][lag_index] / (n - lag)
            abs_acfs.append((abs_cross - abs_mean * abs_mean) / abs_variance)

        return [
            mean,
            math.log(variance),
            skew,
            kurtosis,
            pit_chi,
            pit_ks,
            pit_edge,
            math.log(diff_variance),
            *acfs,
            *abs_acfs,
        ]

    def statistics(self) -> tuple[list[float], list[float], list[float]]:
        raw = self._channel_statistics(0, 0)
        residual5 = self._channel_statistics(1, None)
        residual10 = self._channel_statistics(2, 1)
        return raw, residual5, residual10


class StreamingFeaturesV3:
    """Causal, per-series calibrated state used by both fit and inference."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
        historical_array = np.asarray(historical, dtype=np.float64)
        finite = historical_array[np.isfinite(historical_array)]
        if finite.size < max(WINDOWS) + max(AR_ORDERS) + 2:
            base = finite if finite.size else np.asarray([0.0], dtype=np.float64)
            finite = np.pad(
                base,
                (0, max(WINDOWS) + max(AR_ORDERS) + 2 - base.size),
                mode="edge",
            )
        self.hist_len = int(finite.size)
        self.mean = float(finite.mean())
        self.sd = max(float(finite.std(ddof=1)), EPS)
        raw = np.clip((finite - self.mean) / self.sd, -CLIP_VALUE, CLIP_VALUE)

        self.ar_coefficients = {
            order: _fit_ar_yule_walker(raw, order) for order in AR_ORDERS
        }
        residual5, residual5_mean, residual5_sd = _historical_residuals(
            raw, self.ar_coefficients[5]
        )
        residual10, residual10_mean, residual10_sd = _historical_residuals(
            raw, self.ar_coefficients[10]
        )
        self.residual_center = {5: residual5_mean, 10: residual10_mean}
        self.residual_scale = {5: residual5_sd, 10: residual10_sd}
        self.channel_history = (raw, residual5, residual10)
        self.raw_edges = _quantile_edges(raw)
        self.residual10_edges = _quantile_edges(residual10)
        self.channel_edges = (
            self.raw_edges,
            _quantile_edges(residual5),
            self.residual10_edges,
        )
        self.references = (
            _reference_statistics(raw, self.raw_edges),
            _reference_statistics(residual5, self.channel_edges[1]),
            _reference_statistics(residual10, self.residual10_edges),
        )

        medians: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        counts: list[int] = []
        # Rows are the 96 (window, channel, statistic) null distributions.
        # Padding with +inf makes all row-wise empirical ranks one vectorised
        # comparison rather than 96 Python-level search calls per online step.
        analytic_null_matrix = np.full(
            (len(WINDOWS) * 2 * 8, MAX_ECDF_BLOCKS), np.inf, dtype=np.float64
        )
        analytic_null_counts = np.empty(len(WINDOWS) * 2 * 8, dtype=np.int16)
        self.windows: list[_RollingChannels] = []
        for window in WINDOWS:
            raw_starts = _non_overlapping_starts(raw.size, window)
            residual5_starts = _non_overlapping_starts(residual5.size, window)
            residual10_starts = _non_overlapping_starts(residual10.size, window)
            raw_blocks = _statistics_for_blocks(raw, self.raw_edges, window, raw_starts)
            residual5_blocks = _statistics_for_blocks(
                residual5, self.channel_edges[1], window, residual5_starts
            )
            residual10_blocks = _statistics_for_blocks(
                residual10, self.residual10_edges, window, residual10_starts
            )
            # AR residual histories are shorter by their model order, so the
            # three channels can legitimately have different block counts.
            # Calibrate each statistic marginally, then concatenate locations
            # and scales in the exact online feature order.
            raw_median, raw_scale = _robust_location_scale(raw_blocks)
            residual5_median, residual5_scale = _robust_location_scale(
                residual5_blocks
            )
            residual10_median, residual10_scale = _robust_location_scale(
                residual10_blocks
            )
            median = _select_feature_statistics(
                raw_median, residual5_median, residual10_median
            )
            scale = _select_feature_statistics(
                raw_scale, residual5_scale, residual10_scale
            )
            medians.append(median)
            scales.append(scale)
            counts.append(
                int(min(raw_blocks.shape[0], residual5_blocks.shape[0], residual10_blocks.shape[0]))
            )

            # Exact current top-8 calibration uses the forensic half-window
            # grid (capped at 80) in addition to the non-overlap MAD bank.
            stride = max(1, window // 2)
            raw_ecdf_starts = _block_starts(
                raw.size, window, stride, MAX_ECDF_BLOCKS
            )
            residual10_ecdf_starts = _block_starts(
                residual10.size, window, stride, MAX_ECDF_BLOCKS
            )
            raw_ecdf_stats = _statistics_for_blocks(
                raw, self.raw_edges, window, raw_ecdf_starts
            )
            residual10_ecdf_stats = _statistics_for_blocks(
                residual10, self.residual10_edges, window, residual10_ecdf_starts
            )
            raw_effects = _analytic_effects(raw_ecdf_stats, self.references[0], window)
            residual10_effects = _analytic_effects(
                residual10_ecdf_stats, self.references[2], window
            )
            raw_sorted = np.sort(raw_effects, axis=0).T
            residual10_sorted = np.sort(residual10_effects, axis=0).T
            analytic_offset = WINDOWS.index(window) * 16
            for statistic_index in range(8):
                raw_null = raw_sorted[statistic_index]
                residual10_null = residual10_sorted[statistic_index]
                raw_row = analytic_offset + statistic_index
                residual10_row = analytic_offset + 8 + statistic_index
                analytic_null_matrix[raw_row, : raw_null.size] = raw_null
                analytic_null_matrix[
                    residual10_row, : residual10_null.size
                ] = residual10_null
                analytic_null_counts[raw_row] = raw_null.size
                analytic_null_counts[residual10_row] = residual10_null.size

            rolling = _RollingChannels(window)
            raw_seed = _tail_with_left_padding(raw, window - 1)
            residual5_seed = _tail_with_left_padding(residual5, window - 1)
            residual10_seed = _tail_with_left_padding(residual10, window - 1)
            for index in range(window - 1):
                raw_value = float(raw_seed[index])
                residual5_value = float(residual5_seed[index])
                residual10_value = float(residual10_seed[index])
                rolling.update(
                    (
                        raw_value,
                        residual5_value,
                        residual10_value,
                        int(np.searchsorted(self.raw_edges, raw_value, side="right")),
                        int(
                            np.searchsorted(
                                self.residual10_edges, residual10_value, side="right"
                            )
                        ),
                    )
                )
            self.windows.append(rolling)

        self.null_medians = np.vstack(medians)
        self.null_scales = np.vstack(scales)
        self.null_block_counts = np.asarray(counts, dtype=np.int16)
        self.analytic_null_matrix = analytic_null_matrix
        self.analytic_null_counts = analytic_null_counts
        self.previous_raw: deque[float] = deque(raw[-10:].tolist(), maxlen=10)
        self.n = 0
        self.last_analytic_top8 = 0.0

    def _residual(self, raw_value: float, order: int) -> float:
        coefficients = self.ar_coefficients[order]
        prediction = 0.0
        for lag in range(1, order + 1):
            prediction += coefficients[lag - 1] * self.previous_raw[-lag]
        residual = (
            raw_value - prediction - self.residual_center[order]
        ) / self.residual_scale[order]
        return _clip_scalar(float(residual), -CLIP_VALUE, CLIP_VALUE)

    def update(self, raw_value: float) -> np.ndarray:
        value = float(raw_value)
        if not math.isfinite(value):
            value = self.mean
        raw = _clip_scalar(
            float((value - self.mean) / self.sd), -CLIP_VALUE, CLIP_VALUE
        )
        residual5 = self._residual(raw, 5)
        residual10 = self._residual(raw, 10)
        self.previous_raw.append(raw)
        self.n += 1
        point = (
            raw,
            residual5,
            residual10,
            int(np.searchsorted(self.raw_edges, raw, side="right")),
            int(np.searchsorted(self.residual10_edges, residual10, side="right")),
        )

        features: list[float] = [
            math.log1p(self.n),
            math.sqrt(self.n),
            math.log1p(self.hist_len),
            min(self.n / 256.0, 8.0),
        ]
        analytic_probabilities: list[float] = []
        for window_index, (window, rolling) in enumerate(zip(WINDOWS, self.windows)):
            rolling.update(point)
            raw_stats, residual5_stats, residual10_stats = rolling.statistics()
            selected = np.asarray(
                raw_stats
                + [residual5_stats[index] for index in RESIDUAL5_STAT_INDICES]
                + residual10_stats,
                dtype=np.float64,
            )
            calibrated = (selected - self.null_medians[window_index]) / self.null_scales[
                window_index
            ]
            np.clip(calibrated, -30.0, 30.0, out=calibrated)
            features.append(math.log1p(int(self.null_block_counts[window_index])))
            features.extend(calibrated.tolist())

            raw_effects = _analytic_effects_scalar(
                raw_stats, self.references[0], window
            )
            residual10_effects = _analytic_effects_scalar(
                residual10_stats, self.references[2], window
            )
            analytic_probabilities.extend(raw_effects)
            analytic_probabilities.extend(residual10_effects)

        analytic_effect_array = np.asarray(analytic_probabilities, dtype=np.float64)
        ranks = np.sum(
            self.analytic_null_matrix <= analytic_effect_array[:, None], axis=1
        )
        probabilities = (ranks + 0.5) / (self.analytic_null_counts + 1.0)
        largest = np.partition(probabilities, -8)[-8:]
        self.last_analytic_top8 = float(largest.mean())
        features.append(self.last_analytic_top8)
        result = np.asarray(features, dtype=np.float32)
        if result.size != FEATURE_COUNT:
            raise RuntimeError(
                f"v3 feature width changed: {result.size} != {FEATURE_COUNT}"
            )
        return result


def iter_features(
    historical: Sequence[float] | np.ndarray,
    online: Iterable[float],
) -> Iterator[np.ndarray]:
    """The single feature path shared by training and inference."""

    state = StreamingFeaturesV3(historical)
    for point in online:
        yield state.update(point)


def _time_counts(datasets):
    max_time = max(len(x_online) for _, _, x_online, _ in datasets)
    alive = np.zeros(max_time, dtype=np.int64)
    positive = np.zeros(max_time, dtype=np.int64)
    for _, _, x_online, tau in datasets:
        length = len(x_online)
        alive[:length] += 1
        if tau is not None:
            positive[int(tau) : length] += 1
    return alive, positive


def _sample_residue(dataset_id: int) -> int:
    # 31 mod 5 == 1, so exactly one time residue is retained per series.
    return int((-17 * int(dataset_id)) % SAMPLE_MODULUS)


def _keep_training_row(dataset_id: int, time_online: int) -> bool:
    return int(time_online) % SAMPLE_MODULUS == _sample_residue(dataset_id)


def _sampled_count(dataset_id: int, length: int) -> int:
    first = _sample_residue(dataset_id)
    if first >= length:
        return 0
    return 1 + (length - 1 - first) // SAMPLE_MODULUS


def build_training_matrix(datasets, positive_ramp: int = 16):
    """Build a metric-aligned, deterministic 20% point sample."""

    alive, positive = _time_counts(datasets)
    maximum_rows = sum(
        _sampled_count(int(dataset_id), len(x_online))
        for dataset_id, _, x_online, _ in datasets
    )
    matrix = np.empty((maximum_rows, FEATURE_COUNT), dtype=np.float32)
    target = np.empty(maximum_rows, dtype=np.uint8)
    sample_weight = np.empty(maximum_rows, dtype=np.float32)
    row_index = 0
    for dataset_id, x_historical, x_online, tau in datasets:
        for time_online, feature_row in enumerate(
            iter_features(x_historical, x_online)
        ):
            if not _keep_training_row(int(dataset_id), time_online):
                continue
            label = int(tau is not None and time_online >= int(tau))
            opposite = (
                alive[time_online] - positive[time_online]
                if label
                else positive[time_online]
            )
            if opposite <= 0:
                continue
            weight = float(opposite) / SAMPLE_PROBABILITY
            if label and positive_ramp > 0:
                age = time_online - int(tau) + 1
                weight *= min(1.0, age / positive_ramp)
            matrix[row_index] = feature_row
            target[row_index] = label
            sample_weight[row_index] = weight
            row_index += 1

    matrix = matrix[:row_index]
    target = target[:row_index]
    sample_weight = sample_weight[:row_index]
    if row_index == 0:
        raise ValueError("No trainable rows remain after deterministic sampling")
    sample_weight /= max(float(sample_weight.mean()), EPS)
    return matrix, target, sample_weight


def fit_booster(
    matrix: np.ndarray, target: np.ndarray, sample_weight: np.ndarray
) -> lgb.LGBMClassifier:
    """A compact tree model sized for the 10M-step cloud workload."""

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=475,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=8,
        min_child_samples=500,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.25,
        reg_lambda=6.0,
        random_state=20260803,
        deterministic=True,
        force_col_wise=True,
        n_jobs=max(1, min(8, (os.cpu_count() or 2) - 1)),
        verbosity=-1,
    )
    model.fit(matrix, target, sample_weight=sample_weight)
    return model


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
    positive_ramp: int = 16,
):
    os.makedirs(model_directory_path, exist_ok=True)
    matrix, target, sample_weight = build_training_matrix(
        datasets, positive_ramp=positive_ramp
    )
    model = fit_booster(matrix, target, sample_weight)
    model.booster_.save_model(os.path.join(model_directory_path, MODEL_FILENAME))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    booster = lgb.Booster(model_file=os.path.join(model_directory_path, MODEL_FILENAME))
    yield
    for x_historical, x_online in datasets:
        for features in iter_features(x_historical, x_online):
            prediction = float(
                booster.predict(features.reshape(1, -1), num_threads=1)[0]
            )
            yield min(max(prediction, 1e-7), 1.0 - 1e-7)


INFER_PARALLELISM = 1
