"""Distribution-free streaming detector candidate for ADIA Structural Break.

The feature extractor compares every online suffix with the individual
series' own historical distribution.  Empirical-CDF (PIT) features make the
shape tests distribution-free, while fixed historical AR(5) prewhitening
turns correlation changes into residual distribution changes.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Iterable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np


WINDOWS = (8, 16, 32, 64, 128, 256)
LAGS = (1, 2, 5)
N_BINS = 16
AR_ORDER = 5
SAMPLE_MODULUS = 5
MODEL_FILENAME = "model.txt"
EPS = 1e-8


def _safe_acf(values: np.ndarray, lag: int, transform_abs: bool = False) -> float:
    if values.size <= lag + 1:
        return 0.0
    x = np.abs(values) if transform_abs else values
    left = x[:-lag].astype(np.float64, copy=False)
    right = x[lag:].astype(np.float64, copy=False)
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / denom) if denom > EPS else 0.0


def _moments(values: np.ndarray) -> tuple[float, float]:
    if values.size < 4:
        return 0.0, 0.0
    centered = values - values.mean()
    variance = max(float(np.mean(centered * centered)), EPS)
    skew = float(np.mean(centered**3) / variance**1.5)
    kurt = float(np.mean(centered**4) / (variance * variance) - 3.0)
    return skew, kurt


def _quantile_edges(values: np.ndarray) -> np.ndarray:
    probabilities = np.arange(1, N_BINS, dtype=np.float64) / N_BINS
    return np.quantile(values, probabilities).astype(np.float64, copy=False)


def _fit_ar_yule_walker(values: np.ndarray) -> np.ndarray:
    if values.size <= AR_ORDER + 2:
        return np.zeros(AR_ORDER, dtype=np.float64)
    acf = np.asarray([1.0] + [_safe_acf(values, lag) for lag in range(1, AR_ORDER + 1)])
    matrix = np.empty((AR_ORDER, AR_ORDER), dtype=np.float64)
    for row in range(AR_ORDER):
        for column in range(AR_ORDER):
            matrix[row, column] = acf[abs(row - column)]
    matrix.flat[:: AR_ORDER + 1] += 1e-3
    try:
        coefficients = np.linalg.solve(matrix, acf[1:])
    except np.linalg.LinAlgError:
        coefficients = np.zeros(AR_ORDER, dtype=np.float64)
    return np.clip(coefficients, -1.5, 1.5)


def _historical_residuals(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    if values.size <= AR_ORDER:
        return values.copy()
    prediction = np.zeros(values.size - AR_ORDER, dtype=np.float64)
    for lag, coefficient in enumerate(coefficients, start=1):
        prediction += coefficient * values[AR_ORDER - lag : values.size - lag]
    return values[AR_ORDER:] - prediction


class RollingComparison:
    """O(1) raw moments, PIT counts and dependence for one rolling suffix."""

    __slots__ = (
        "size",
        "values",
        "sum_z",
        "sum_z2",
        "sum_z3",
        "sum_z4",
        "sum_abs",
        "sum_abs2",
        "sum_r",
        "sum_r2",
        "z_counts",
        "r_counts",
        "z_cross",
        "abs_cross",
        "r_cross",
    )

    def __init__(self, size: int):
        self.size = size
        self.values: deque[tuple[float, int, float, int]] = deque()
        self.sum_z = 0.0
        self.sum_z2 = 0.0
        self.sum_z3 = 0.0
        self.sum_z4 = 0.0
        self.sum_abs = 0.0
        self.sum_abs2 = 0.0
        self.sum_r = 0.0
        self.sum_r2 = 0.0
        self.z_counts = np.zeros(N_BINS, dtype=np.int16)
        self.r_counts = np.zeros(N_BINS, dtype=np.int16)
        self.z_cross = np.zeros(len(LAGS), dtype=np.float64)
        self.abs_cross = np.zeros(len(LAGS), dtype=np.float64)
        self.r_cross = np.zeros(len(LAGS), dtype=np.float64)

    def update(self, z: float, z_bin: int, residual: float, residual_bin: int) -> None:
        values = self.values
        if len(values) == self.size:
            old_z, old_z_bin, old_r, old_r_bin = values[0]
            for index, lag in enumerate(LAGS):
                if len(values) > lag:
                    other_z, _, other_r, _ = values[lag]
                    self.z_cross[index] -= old_z * other_z
                    self.abs_cross[index] -= abs(old_z) * abs(other_z)
                    self.r_cross[index] -= old_r * other_r
            values.popleft()
            old_z2 = old_z * old_z
            self.sum_z -= old_z
            self.sum_z2 -= old_z2
            self.sum_z3 -= old_z2 * old_z
            self.sum_z4 -= old_z2 * old_z2
            self.sum_abs -= abs(old_z)
            self.sum_abs2 -= abs(old_z) ** 2
            self.sum_r -= old_r
            self.sum_r2 -= old_r * old_r
            self.z_counts[old_z_bin] -= 1
            self.r_counts[old_r_bin] -= 1

        for index, lag in enumerate(LAGS):
            if len(values) >= lag:
                other_z, _, other_r, _ = values[-lag]
                self.z_cross[index] += other_z * z
                self.abs_cross[index] += abs(other_z) * abs(z)
                self.r_cross[index] += other_r * residual

        values.append((z, z_bin, residual, residual_bin))
        z2 = z * z
        self.sum_z += z
        self.sum_z2 += z2
        self.sum_z3 += z2 * z
        self.sum_z4 += z2 * z2
        self.sum_abs += abs(z)
        self.sum_abs2 += abs(z) ** 2
        self.sum_r += residual
        self.sum_r2 += residual * residual
        self.z_counts[z_bin] += 1
        self.r_counts[residual_bin] += 1

    @staticmethod
    def _pit_statistics(counts: np.ndarray, n: int) -> tuple[float, float, float]:
        expected = n / N_BINS
        differences = counts.astype(np.float64) - expected
        chi_square = float(np.sum(differences * differences / max(expected, EPS)))
        chi_z = (chi_square - (N_BINS - 1)) / math.sqrt(2.0 * (N_BINS - 1))
        cumulative = np.cumsum(counts, dtype=np.float64) / n
        reference = np.arange(1, N_BINS + 1, dtype=np.float64) / N_BINS
        ks = math.sqrt(n) * float(np.max(np.abs(cumulative - reference)))
        edge_rate = float((counts[0] + counts[-1]) / n)
        edge_z = (edge_rate - 2.0 / N_BINS) * math.sqrt(n)
        return chi_z, ks, edge_z

    def features(
        self,
        hist_skew: float,
        hist_kurt: float,
        hist_z2_sd: float,
        hist_acf: np.ndarray,
        hist_abs_acf1: float,
        hist_residual_acf1: float,
    ) -> tuple[list[float], list[float]]:
        n = len(self.values)
        if n < 2:
            # Keep the schema identical to the fully initialized path below.
            # There are 21 raw-distribution values and 11 residual/dependence
            # values, followed by the eight family summaries.
            return [0.0] * 32, [0.0] * 8
        root_n = math.sqrt(n)
        mean = self.sum_z / n
        variance = max(self.sum_z2 / n - mean * mean, EPS)
        sd = math.sqrt(variance)
        m3 = self.sum_z3 / n - 3.0 * mean * (self.sum_z2 / n) + 2.0 * mean**3
        m4 = (
            self.sum_z4 / n
            - 4.0 * mean * (self.sum_z3 / n)
            + 6.0 * mean * mean * (self.sum_z2 / n)
            - 3.0 * mean**4
        )
        skew = m3 / max(sd**3, EPS)
        kurt = m4 / max(variance * variance, EPS) - 3.0
        mean_stat = mean * root_n
        variance_stat = (self.sum_z2 / n - 1.0) * root_n / hist_z2_sd
        skew_stat = (skew - hist_skew) * math.sqrt(n / 6.0)
        kurt_stat = (kurt - hist_kurt) * math.sqrt(n / 24.0)
        chi_z, ks, edge_z = self._pit_statistics(self.z_counts, n)

        acf_statistics: list[float] = []
        for index, lag in enumerate(LAGS):
            if n <= lag + 1:
                acf_statistics.append(0.0)
                continue
            cross = self.z_cross[index] / (n - lag)
            acf = (cross - mean * mean) / variance
            acf_statistics.append((acf - hist_acf[index]) * root_n)

        mean_abs = self.sum_abs / n
        abs_variance = max(self.sum_abs2 / n - mean_abs * mean_abs, EPS)
        if n > 2:
            abs_cross = self.abs_cross[0] / (n - 1)
            abs_acf1 = (abs_cross - mean_abs * mean_abs) / abs_variance
            abs_acf_stat = (abs_acf1 - hist_abs_acf1) * root_n
        else:
            abs_acf_stat = 0.0

        residual_mean = self.sum_r / n
        residual_variance = max(self.sum_r2 / n - residual_mean * residual_mean, EPS)
        residual_mean_stat = residual_mean * root_n
        residual_variance_stat = (self.sum_r2 / n - 1.0) * root_n
        r_chi_z, r_ks, r_edge_z = self._pit_statistics(self.r_counts, n)
        if n > 2:
            r_cross = self.r_cross[0] / (n - 1)
            r_acf1 = (r_cross - residual_mean * residual_mean) / residual_variance
            r_acf_stat = (r_acf1 - hist_residual_acf1) * root_n
        else:
            r_acf_stat = 0.0

        values = [
            mean_stat,
            abs(mean_stat),
            variance_stat,
            abs(variance_stat),
            skew_stat,
            abs(skew_stat),
            kurt_stat,
            abs(kurt_stat),
            chi_z,
            abs(chi_z),
            ks,
            edge_z,
            abs(edge_z),
            acf_statistics[0],
            abs(acf_statistics[0]),
            acf_statistics[1],
            abs(acf_statistics[1]),
            acf_statistics[2],
            abs(acf_statistics[2]),
            abs_acf_stat,
            abs(abs_acf_stat),
            residual_mean_stat,
            abs(residual_mean_stat),
            residual_variance_stat,
            abs(residual_variance_stat),
            r_chi_z,
            abs(r_chi_z),
            r_ks,
            r_edge_z,
            abs(r_edge_z),
            r_acf_stat,
            abs(r_acf_stat),
        ]
        family = [
            abs(mean_stat),
            abs(variance_stat),
            max(ks, abs(chi_z), abs(edge_z)),
            max(abs(skew_stat), abs(kurt_stat)),
            max(abs(value) for value in acf_statistics),
            abs(abs_acf_stat),
            max(abs(residual_mean_stat), abs(residual_variance_stat), r_ks, abs(r_chi_z)),
            abs(r_acf_stat),
        ]
        return values, family


class StreamingFeaturesV2:
    """Leak-free distribution and dependence comparisons for one series."""

    def __init__(self, historical: List[float] | np.ndarray):
        historical_array = np.asarray(historical, dtype=np.float64)
        finite = historical_array[np.isfinite(historical_array)]
        if finite.size < 10:
            finite = np.pad(finite if finite.size else np.asarray([0.0]), (0, 10 - finite.size))
        self.hist_len = int(finite.size)
        self.mean = float(finite.mean())
        self.sd = max(float(finite.std(ddof=1)), EPS)
        standardized = np.clip((finite - self.mean) / self.sd, -20.0, 20.0)
        self.z_edges = _quantile_edges(standardized)
        self.hist_skew, self.hist_kurt = _moments(standardized)
        self.hist_z2_sd = max(float(np.std(standardized * standardized)), 0.25)
        self.hist_acf = np.asarray([_safe_acf(standardized, lag) for lag in LAGS])
        self.hist_abs_acf1 = _safe_acf(standardized, 1, transform_abs=True)

        self.ar_coefficients = _fit_ar_yule_walker(standardized)
        residuals = _historical_residuals(standardized, self.ar_coefficients)
        self.residual_mean = float(residuals.mean())
        self.residual_sd = max(float(residuals.std(ddof=1)), EPS)
        residuals = np.clip((residuals - self.residual_mean) / self.residual_sd, -20.0, 20.0)
        self.residual_edges = _quantile_edges(residuals)
        self.hist_residual_acf1 = _safe_acf(residuals, 1)
        self.previous_z: deque[float] = deque(standardized[-AR_ORDER:].tolist(), maxlen=AR_ORDER)

        self.n = 0
        self.windows = [RollingComparison(size) for size in WINDOWS]
        self.decayed_max = np.zeros(8, dtype=np.float64)
        self.running_max = np.zeros(8, dtype=np.float64)

    def update(self, raw_value: float) -> np.ndarray:
        value = float(raw_value)
        if not math.isfinite(value):
            value = self.mean
        z = float(np.clip((value - self.mean) / self.sd, -20.0, 20.0))
        lagged = list(reversed(self.previous_z))
        prediction = sum(
            self.ar_coefficients[index] * lagged[index]
            for index in range(min(len(lagged), AR_ORDER))
        )
        residual = float(np.clip((z - prediction - self.residual_mean) / self.residual_sd, -20.0, 20.0))
        self.previous_z.append(z)
        self.n += 1
        z_bin = int(np.searchsorted(self.z_edges, z, side="right"))
        residual_bin = int(np.searchsorted(self.residual_edges, residual, side="right"))

        features: list[float] = [
            math.log1p(self.n),
            math.sqrt(self.n),
            math.log1p(self.hist_len),
            self.hist_skew,
            self.hist_kurt,
            self.hist_acf[0],
            self.hist_acf[1],
            self.hist_acf[2],
            self.hist_abs_acf1,
            abs(z),
            z * z,
            abs(residual),
            residual * residual,
        ]
        current_family = np.zeros(8, dtype=np.float64)
        for window in self.windows:
            window.update(z, z_bin, residual, residual_bin)
            window_values, family = window.features(
                self.hist_skew,
                self.hist_kurt,
                self.hist_z2_sd,
                self.hist_acf,
                self.hist_abs_acf1,
                self.hist_residual_acf1,
            )
            features.extend(window_values)
            current_family = np.maximum(current_family, np.asarray(family))

        self.decayed_max = np.maximum(current_family, 0.995 * self.decayed_max)
        self.running_max = np.maximum(self.running_max, current_family)
        features.extend(current_family.tolist())
        features.extend(self.decayed_max.tolist())
        features.extend(self.running_max.tolist())
        result = np.asarray(features, dtype=np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=50.0, neginf=-50.0)


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


def _keep_training_row(dataset_id: int, time_online: int) -> bool:
    if time_online < 32:
        return True
    if time_online < 64:
        return ((dataset_id * 13 + time_online) % 2) == 0
    return ((dataset_id * 17 + time_online) % SAMPLE_MODULUS) == 0


def build_training_matrix(datasets, positive_ramp: int = 16):
    alive, positive = _time_counts(datasets)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for dataset_id, x_historical, x_online, tau in datasets:
        state = StreamingFeaturesV2(x_historical)
        for time_online, point in enumerate(x_online):
            feature_row = state.update(point)
            if not _keep_training_row(int(dataset_id), time_online):
                continue
            label = int(tau is not None and time_online >= int(tau))
            opposite = alive[time_online] - positive[time_online] if label else positive[time_online]
            if opposite <= 0:
                continue
            weight = float(opposite)
            if label and positive_ramp > 0:
                age = time_online - int(tau) + 1
                weight *= min(1.0, age / positive_ramp)
            rows.append(feature_row)
            labels.append(label)
            weights.append(weight)
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    target = np.asarray(labels, dtype=np.uint8)
    sample_weight = np.asarray(weights, dtype=np.float32)
    sample_weight /= max(float(sample_weight.mean()), EPS)
    return matrix, target, sample_weight


def fit_booster(matrix: np.ndarray, target: np.ndarray, sample_weight: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=47,
        max_depth=9,
        min_child_samples=400,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.78,
        reg_alpha=0.20,
        reg_lambda=5.0,
        random_state=20260802,
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
):
    os.makedirs(model_directory_path, exist_ok=True)
    matrix, target, sample_weight = build_training_matrix(datasets)
    model = fit_booster(matrix, target, sample_weight)
    model.booster_.save_model(os.path.join(model_directory_path, MODEL_FILENAME))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    booster = lgb.Booster(model_file=os.path.join(model_directory_path, MODEL_FILENAME))
    yield
    for x_historical, x_online in datasets:
        state = StreamingFeaturesV2(x_historical)
        for point in x_online:
            features = state.update(point)
            prediction = float(booster.predict(features.reshape(1, -1), num_threads=1)[0])
            yield min(max(prediction, 1e-7), 1.0 - 1e-7)


INFER_PARALLELISM = 1
