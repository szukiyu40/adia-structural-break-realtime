"""Deterministic 67-dimensional process descriptor from one historical series.

The implementation is deliberately inference-safe: it receives one immutable
historical prefix and never needs labels, online values, another series, or the
final online horizon.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


FEATURE_NAMES = (
    "hist_len", "log_hist_len", "hist_mean", "hist_log_sd", "z_skew",
    "z_kurt", "z_q01", "z_q05", "z_q10", "z_q25", "z_q50", "z_q75",
    "z_q90", "z_q95", "z_q99", "z_iqr", "z_mad", "z_mean_abs",
    "z_median_abs", "z_tail_gt1", "z_tail_gt2", "z_tail_gt3",
    "z_positive_rate", "z_flip_rate", "z_hist_entropy", "acf_raw_lag1",
    "acf_raw_lag2", "acf_raw_lag3", "acf_raw_lag5", "acf_raw_lag10",
    "acf_raw_lag20", "acf_raw_lag50", "acf_abs_lag1", "acf_sq_lag1",
    "acf_abs_lag2", "acf_sq_lag2", "acf_abs_lag5", "acf_sq_lag5",
    "acf_abs_lag10", "acf_sq_lag10", "acf_abs_lag20", "acf_sq_lag20",
    "diff_var", "diff_mean_abs", "diff_skew", "diff_kurt",
    "acf_diff_lag1", "acf_diff_lag2", "acf_diff_lag5", "ar5_coef1",
    "ar5_coef2", "ar5_coef3", "ar5_coef4", "ar5_coef5", "innov_sd",
    "innov_skew", "innov_kurt", "innov_acf_lag1", "innov_acf_lag2",
    "innov_acf_lag5", "innov_abs_acf_lag1", "innov_sq_acf_lag1",
    "hist_trend_scaled", "half_mean_diff", "half_logvar_ratio",
    "quarter_mean_range", "quarter_logvar_range",
)

AR_ORDER = 5
AR_RIDGE = 1e-4
VARIANCE_EPSILON = 1e-6
ENTROPY_EDGES = np.linspace(-4.0, 4.0, 17, dtype=np.float64)


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(output)
    replacement = float(np.median(output[finite])) if finite.any() else 0.0
    output[~finite] = replacement
    return output


def _correlation(values: np.ndarray, lag: int) -> float:
    if values.size <= lag + 1:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 1e-18:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    centered = values - np.mean(values)
    second = float(np.mean(centered * centered))
    if second <= 1e-18:
        return 0.0, 0.0
    scaled = centered / math.sqrt(second)
    return float(np.mean(scaled**3)), float(np.mean(scaled**4) - 3.0)


def _ar5(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    correlations = np.asarray(
        (1.0, *(_correlation(z, lag) for lag in range(1, AR_ORDER + 1))),
        dtype=np.float64,
    )
    toeplitz = np.empty((AR_ORDER, AR_ORDER), dtype=np.float64)
    for row in range(AR_ORDER):
        for column in range(AR_ORDER):
            toeplitz[row, column] = correlations[abs(row - column)]
    coefficients = np.linalg.solve(
        toeplitz + AR_RIDGE * np.eye(AR_ORDER),
        correlations[1:],
    )
    design = np.column_stack([
        z[AR_ORDER - lag : -lag] for lag in range(1, AR_ORDER + 1)
    ])
    innovations = z[AR_ORDER:] - design @ coefficients
    return coefficients, innovations


def extract_history_process_features(
    historical: Sequence[float] | np.ndarray,
) -> np.ndarray:
    history = _finite(historical)
    if history.size < 64:
        raise ValueError("process descriptor requires at least 64 history ticks")
    count = int(history.size)
    mean = float(np.mean(history))
    sample_sd = float(np.std(history, ddof=1))
    sample_sd = max(sample_sd, 1e-12)
    z = (history - mean) / sample_sd
    z_skew = float(np.mean(z**3))
    z_kurt = float(np.mean(z**4) - 3.0)
    quantiles = np.quantile(
        z, (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
        method="linear",
    )
    median = float(quantiles[4])
    absolute = np.abs(z)
    entropy_counts = np.histogram(
        np.clip(z, ENTROPY_EDGES[0], ENTROPY_EDGES[-1]),
        bins=ENTROPY_EDGES,
    )[0].astype(np.float64)
    entropy_probability = entropy_counts[entropy_counts > 0.0] / count
    entropy = float(-np.sum(entropy_probability * np.log(entropy_probability)))

    raw_lags = (1, 2, 3, 5, 10, 20, 50)
    volatility_lags = (1, 2, 5, 10, 20)
    raw_acf = [_correlation(z, lag) for lag in raw_lags]
    absolute_acf = [_correlation(absolute, lag) for lag in volatility_lags]
    squared_acf = [_correlation(z * z, lag) for lag in volatility_lags]

    difference = np.diff(z)
    diff_skew, diff_kurt = _skew_kurtosis(difference)
    diff_acf = [_correlation(difference, lag) for lag in (1, 2, 5)]
    ar_coefficients, innovations = _ar5(z)
    innovation_skew, innovation_kurt = _skew_kurtosis(innovations)
    innovation_acf = [_correlation(innovations, lag) for lag in (1, 2, 5)]

    tick = np.arange(count, dtype=np.float64)
    trend = float(np.polyfit(tick, z, deg=1)[0] * count)
    halves = np.array_split(z, 2)
    quarters = np.array_split(z, 4)
    half_mean_diff = float(np.mean(halves[1]) - np.mean(halves[0]))
    half_variances = np.asarray([np.var(part) for part in halves])
    half_logvar_ratio = float(np.log(
        (half_variances[1] + VARIANCE_EPSILON)
        / (half_variances[0] + VARIANCE_EPSILON)
    ))
    quarter_means = np.asarray([np.mean(part) for part in quarters])
    quarter_variances = np.asarray([np.var(part) for part in quarters])

    features = np.asarray((
        float(count), math.log1p(count), mean, math.log(sample_sd),
        z_skew, z_kurt, *quantiles.tolist(),
        float(quantiles[5] - quantiles[3]),
        float(np.median(np.abs(z - median))), float(np.mean(absolute)),
        float(np.median(absolute)), float(np.mean(absolute > 1.0)),
        float(np.mean(absolute > 2.0)), float(np.mean(absolute > 3.0)),
        float(np.mean(z > 0.0)), float(np.mean(z[:-1] * z[1:] < 0.0)),
        entropy, *raw_acf,
        absolute_acf[0], squared_acf[0], absolute_acf[1], squared_acf[1],
        absolute_acf[2], squared_acf[2], absolute_acf[3], squared_acf[3],
        absolute_acf[4], squared_acf[4],
        float(np.var(difference)), float(np.mean(np.abs(difference))),
        diff_skew, diff_kurt, *diff_acf, *ar_coefficients.tolist(),
        float(np.std(innovations)), innovation_skew, innovation_kurt,
        *innovation_acf, _correlation(np.abs(innovations), 1),
        _correlation(innovations * innovations, 1), trend, half_mean_diff,
        half_logvar_ratio, float(np.max(quarter_means) - np.min(quarter_means)),
        float(
            np.max(np.log(quarter_variances + VARIANCE_EPSILON))
            - np.min(np.log(quarter_variances + VARIANCE_EPSILON))
        ),
    ), dtype=np.float64)
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise RuntimeError("invalid process descriptor")
    return features

