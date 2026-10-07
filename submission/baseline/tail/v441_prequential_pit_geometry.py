"""History-prequential conditional-PIT geometry for ADIA.

Every fitted object and null reference is derived from one series' historical
segment.  Online updates use only the current/past prefix and start from the
terminal historical lag and volatility state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
AR_ORDERS = (0, 1, 3, 8)
MAX_AR_ORDER = max(AR_ORDERS)
BIC_TEMPERATURE = 8.0
RIDGE = 1.0
FAST_ALPHA = 0.05
SLOW_ALPHA = 0.005
VARIANCE_FLOOR = 0.25
REFERENCE_WARMUP = 64
BLOCK_WIDTHS = (16, 32, 64, 128, 256)
REFERENCE_STRIDE = 8
MEMORY_FAST_ALPHA = 0.10
MEMORY_SLOW_ALPHA = 0.02
MEMORY_DECAY = 0.995
OCCUPATION_ALPHA = 0.02
CUSUM_DRIFT = 0.78
FEATURE_CLIP = 20.0
EPSILON = 1e-8

FEATURE_NAMES = (
    "pit_energy_w16",
    "pit_energy_w32",
    "pit_energy_w64",
    "pit_energy_w128",
    "pit_energy_w256",
    "pit_location_max",
    "pit_scale_max",
    "pit_skew_max",
    "pit_kurtosis_max",
    "pit_dependence_max",
    "pit_cross_max",
    "pit_cross_top2_mean",
    "pit_cross_geometric_mean",
    "pit_fast_ewma",
    "pit_slow_ewma",
    "pit_runmax",
    "pit_decayed_max",
    "pit_occupation",
    "pit_cusum",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(array)
    replacement = float(np.median(array[finite])) if np.any(finite) else 0.0
    array[~finite] = replacement
    return array


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    scale = max(1.4826 * mad, 0.10 * standard_deviation, EPSILON)
    return center, scale


@dataclass(frozen=True)
class ARModel:
    order: int
    coefficients: np.ndarray

    def predict(self, lags: Sequence[float]) -> float:
        if self.order == 0:
            return float(self.coefficients[0])
        lag_array = np.asarray(lags, dtype=np.float64)
        recent = lag_array[-self.order :][::-1]
        return float(self.coefficients[0] + self.coefficients[1:] @ recent)


@dataclass(frozen=True)
class PredictorEnsemble:
    models: tuple[ARModel, ...]
    weights: np.ndarray

    def predict(self, lags: Sequence[float]) -> float:
        predictions = np.asarray(
            [model.predict(lags) for model in self.models], dtype=np.float64
        )
        return float(self.weights @ predictions)


def _fit_predictor(history_z: np.ndarray) -> tuple[PredictorEnsemble, np.ndarray]:
    if history_z.size < 512:
        raise ValueError("V441 requires at least 512 historical ticks")
    target = history_z[MAX_AR_ORDER:]
    sample_count = target.size
    models: list[ARModel] = []
    predictions: list[np.ndarray] = []
    bics: list[float] = []
    for order in AR_ORDERS:
        if order == 0:
            design = np.ones((sample_count, 1), dtype=np.float64)
        else:
            columns = [np.ones(sample_count, dtype=np.float64)]
            columns.extend(
                history_z[MAX_AR_ORDER - lag : history_z.size - lag]
                for lag in range(1, order + 1)
            )
            design = np.column_stack(columns)
        penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE
        penalty[0, 0] = EPSILON
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        fitted = design @ coefficients
        mse = max(float(np.mean((target - fitted) ** 2)), EPSILON)
        bic = sample_count * math.log(mse) + design.shape[1] * math.log(sample_count)
        models.append(ARModel(order=order, coefficients=coefficients))
        predictions.append(fitted)
        bics.append(bic)
    bic_array = np.asarray(bics, dtype=np.float64)
    log_weights = -0.5 * (bic_array - float(np.min(bic_array))) / BIC_TEMPERATURE
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    weights /= float(np.sum(weights))
    ensemble_prediction = np.sum(
        np.stack(predictions, axis=1) * weights[None, :], axis=1
    )
    residuals = target - ensemble_prediction
    return PredictorEnsemble(tuple(models), weights), residuals


def _empirical_percentile(sorted_reference: np.ndarray, value: float) -> float:
    index = int(np.searchsorted(sorted_reference, float(value), side="right"))
    return float((index + 1.0) / (sorted_reference.size + 1.0))


def _legendre_core(pit: np.ndarray) -> np.ndarray:
    q = np.clip(2.0 * np.asarray(pit, dtype=np.float64) - 1.0, -1.0, 1.0)
    q2 = q * q
    l1 = math.sqrt(3.0) * q
    l2 = math.sqrt(5.0) * (3.0 * q2 - 1.0) / 2.0
    l3 = math.sqrt(7.0) * (5.0 * q2 * q - 3.0 * q) / 2.0
    l4 = 3.0 * (35.0 * q2 * q2 - 30.0 * q2 + 3.0) / 8.0
    return np.column_stack((l1, l2, l3, l4))


def _pit_values(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    ranks = np.searchsorted(sorted_reference, values, side="right").astype(np.float64)
    return (ranks + 0.5) / (sorted_reference.size + 1.0)


def _basis_with_dependence(pit: np.ndarray) -> np.ndarray:
    core = _legendre_core(pit)
    l1 = core[:, 0]
    dep1 = np.zeros_like(l1)
    dep5 = np.zeros_like(l1)
    dep1[1:] = l1[1:] * l1[:-1]
    dep5[5:] = l1[5:] * l1[:-5]
    return np.column_stack((core, dep1, dep5))[5:]


def _regularized_inverse(matrix: np.ndarray) -> np.ndarray:
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=np.float64))
    scale = max(float(np.trace(covariance)) / covariance.shape[0], EPSILON)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * (0.05 * scale + 1e-6)
    return np.linalg.inv(covariance)


@dataclass(frozen=True)
class BlockReference:
    width: int
    center: np.ndarray
    inverse: np.ndarray
    order_center: np.ndarray
    order_scale: np.ndarray
    dependence_center: np.ndarray
    dependence_inverse: np.ndarray
    energy_reference: np.ndarray
    order_references: tuple[np.ndarray, ...]
    dependence_reference: np.ndarray

    @classmethod
    def fit(cls, basis: np.ndarray, width: int) -> "BlockReference":
        cumulative = np.vstack((np.zeros((1, basis.shape[1])), np.cumsum(basis, axis=0)))
        endpoints = np.arange(width, basis.shape[0] + 1, REFERENCE_STRIDE, dtype=np.int64)
        if endpoints.size == 0 or endpoints[-1] != basis.shape[0]:
            endpoints = np.r_[endpoints, basis.shape[0]]
        means = (cumulative[endpoints] - cumulative[endpoints - width]) / float(width)
        if means.shape[0] < 16:
            raise ValueError(f"insufficient V441 history blocks for width {width}")
        center = np.mean(means, axis=0)
        inverse = _regularized_inverse(means)
        centered = means - center
        energy = np.einsum("ij,jk,ik->i", centered, inverse, centered)

        order_center = np.median(means[:, :4], axis=0)
        order_mad = np.median(np.abs(means[:, :4] - order_center), axis=0)
        order_std = np.std(means[:, :4], axis=0, ddof=1)
        order_scale = np.maximum.reduce(
            (1.4826 * order_mad, 0.10 * order_std, np.full(4, EPSILON))
        )
        order_deviation = np.abs((means[:, :4] - order_center) / order_scale)

        dependence_center = np.mean(means[:, 4:], axis=0)
        dependence_inverse = _regularized_inverse(means[:, 4:])
        dep_centered = means[:, 4:] - dependence_center
        dependence = np.einsum(
            "ij,jk,ik->i", dep_centered, dependence_inverse, dep_centered
        )
        return cls(
            width=width,
            center=center,
            inverse=inverse,
            order_center=order_center,
            order_scale=order_scale,
            dependence_center=dependence_center,
            dependence_inverse=dependence_inverse,
            energy_reference=np.sort(energy),
            order_references=tuple(
                np.sort(order_deviation[:, index]) for index in range(4)
            ),
            dependence_reference=np.sort(dependence),
        )

    def score(self, block_mean: np.ndarray) -> tuple[float, np.ndarray, float]:
        centered = block_mean - self.center
        energy = float(centered @ self.inverse @ centered)
        energy_rank = _empirical_percentile(self.energy_reference, energy)
        order_deviation = np.abs(
            (block_mean[:4] - self.order_center) / self.order_scale
        )
        order_ranks = np.asarray(
            [
                _empirical_percentile(reference, value)
                for reference, value in zip(
                    self.order_references, order_deviation, strict=True
                )
            ],
            dtype=np.float64,
        )
        dep_centered = block_mean[4:] - self.dependence_center
        dependence = float(
            dep_centered @ self.dependence_inverse @ dep_centered
        )
        dependence_rank = _empirical_percentile(
            self.dependence_reference, dependence
        )
        return energy_rank, order_ranks, dependence_rank


def _history_innovations(
    residuals: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    residual_center, residual_scale = _robust_location_scale(residuals)
    standardized = np.clip(
        (residuals - residual_center) / residual_scale, -20.0, 20.0
    )
    initial = max(float(np.mean(standardized[:REFERENCE_WARMUP] ** 2)), EPSILON)
    fast_variance = initial
    slow_variance = initial
    innovations = np.empty_like(standardized)
    for index, residual in enumerate(standardized):
        forecast_variance = VARIANCE_FLOOR + 0.5 * (fast_variance + slow_variance)
        innovations[index] = residual / math.sqrt(forecast_variance)
        squared = min(float(residual * residual), 400.0)
        fast_variance = (1.0 - FAST_ALPHA) * fast_variance + FAST_ALPHA * squared
        slow_variance = (1.0 - SLOW_ALPHA) * slow_variance + SLOW_ALPHA * squared
    return innovations, residual_center, residual_scale, fast_variance, slow_variance


class PrequentialPITGeometry:
    """Fixed history predictor plus causal conditional-PIT monitoring."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
        history = _finite(historical)
        if history.size < 512:
            raise ValueError("V441 requires at least 512 historical ticks")
        self.replacement = float(np.median(history))
        self.center, self.scale = _robust_location_scale(history)
        history_z = np.clip((history - self.center) / self.scale, -12.0, 12.0)
        self.predictor, residuals = _fit_predictor(history_z)
        (
            innovations,
            self.residual_center,
            self.residual_scale,
            self.fast_variance,
            self.slow_variance,
        ) = _history_innovations(residuals)
        calibration = innovations[REFERENCE_WARMUP:]
        self.sorted_innovations = np.sort(calibration)
        historical_pit = _pit_values(calibration, self.sorted_innovations)
        historical_basis = _basis_with_dependence(historical_pit)
        self.references = tuple(
            BlockReference.fit(historical_basis, width) for width in BLOCK_WIDTHS
        )
        history_core = _legendre_core(historical_pit)
        self.previous_l1: deque[float] = deque(
            history_core[-5:, 0].tolist(), maxlen=5
        )
        self.lags: deque[float] = deque(
            history_z[-MAX_AR_ORDER:].tolist(), maxlen=MAX_AR_ORDER
        )
        self.cumulative: list[np.ndarray] = [np.zeros(6, dtype=np.float64)]
        self.fast_memory = 0.0
        self.slow_memory = 0.0
        self.runmax = 0.0
        self.decayed = 0.0
        self.occupation = 0.0
        self.cusum = 0.0

    def update(self, raw_value: float) -> np.ndarray:
        value = float(raw_value) if np.isfinite(raw_value) else self.replacement
        z_value = float(np.clip((value - self.center) / self.scale, -12.0, 12.0))
        prediction = self.predictor.predict(self.lags)
        residual = float(
            np.clip(
                (z_value - prediction - self.residual_center) / self.residual_scale,
                -20.0,
                20.0,
            )
        )
        forecast_variance = VARIANCE_FLOOR + 0.5 * (
            self.fast_variance + self.slow_variance
        )
        innovation = residual / math.sqrt(forecast_variance)
        pit = float(
            (np.searchsorted(self.sorted_innovations, innovation, side="right") + 0.5)
            / (self.sorted_innovations.size + 1.0)
        )
        core = _legendre_core(np.asarray([pit], dtype=np.float64))[0]
        previous = list(self.previous_l1)
        dep1 = core[0] * previous[-1]
        dep5 = core[0] * previous[0]
        basis = np.asarray((*core, dep1, dep5), dtype=np.float64)
        self.previous_l1.append(float(core[0]))
        self.cumulative.append(self.cumulative[-1] + basis)

        squared = min(residual * residual, 400.0)
        self.fast_variance = (
            (1.0 - FAST_ALPHA) * self.fast_variance + FAST_ALPHA * squared
        )
        self.slow_variance = (
            (1.0 - SLOW_ALPHA) * self.slow_variance + SLOW_ALPHA * squared
        )
        self.lags.append(z_value)

        observed = len(self.cumulative) - 1
        energy_ranks = np.zeros(len(BLOCK_WIDTHS), dtype=np.float64)
        order_ranks: list[np.ndarray] = []
        dependence_ranks: list[float] = []
        active_energy: list[float] = []
        for index, reference in enumerate(self.references):
            width = reference.width
            if observed < width:
                continue
            block_mean = (
                self.cumulative[-1] - self.cumulative[-width - 1]
            ) / float(width)
            energy, orders, dependence = reference.score(block_mean)
            energy_ranks[index] = energy
            active_energy.append(energy)
            order_ranks.append(orders)
            dependence_ranks.append(dependence)

        if active_energy:
            ordered_energy = np.sort(np.asarray(active_energy, dtype=np.float64))
            cross_max = float(ordered_energy[-1])
            top2 = float(np.mean(ordered_energy[-2:]))
            geometric = float(
                math.exp(np.mean(np.log(np.clip(ordered_energy, 1e-6, 1.0))))
            )
            order_maximum = np.max(np.stack(order_ranks), axis=0)
            dependence_maximum = float(np.max(dependence_ranks))
        else:
            cross_max = top2 = geometric = dependence_maximum = 0.0
            order_maximum = np.zeros(4, dtype=np.float64)

        self.fast_memory = (
            (1.0 - MEMORY_FAST_ALPHA) * self.fast_memory
            + MEMORY_FAST_ALPHA * top2
        )
        self.slow_memory = (
            (1.0 - MEMORY_SLOW_ALPHA) * self.slow_memory
            + MEMORY_SLOW_ALPHA * top2
        )
        self.runmax = max(self.runmax, top2)
        self.decayed = max(top2, MEMORY_DECAY * self.decayed)
        self.occupation = (
            (1.0 - OCCUPATION_ALPHA) * self.occupation
            + OCCUPATION_ALPHA * float(top2 >= 0.90)
        )
        self.cusum = min(
            FEATURE_CLIP, max(0.0, self.cusum + top2 - CUSUM_DRIFT)
        )
        output = np.asarray(
            (
                *energy_ranks,
                *order_maximum,
                dependence_maximum,
                cross_max,
                top2,
                geometric,
                self.fast_memory,
                self.slow_memory,
                self.runmax,
                self.decayed,
                self.occupation,
                self.cusum,
            ),
            dtype=np.float32,
        )
        np.nan_to_num(output, copy=False, nan=0.0, posinf=FEATURE_CLIP, neginf=0.0)
        np.clip(output, 0.0, FEATURE_CLIP, out=output)
        if output.shape != (FEATURE_COUNT,) or not np.isfinite(output).all():
            raise RuntimeError("V441 feature row is invalid")
        return output


def batch_features(
    historical: Sequence[float] | np.ndarray,
    online: Sequence[float] | np.ndarray,
) -> np.ndarray:
    state = PrequentialPITGeometry(historical)
    values = _finite(online)
    output = np.empty((values.size, FEATURE_COUNT), dtype=np.float32)
    for tick, value in enumerate(values):
        output[tick] = state.update(float(value))
    return output


if FEATURE_COUNT != 19 or len(set(FEATURE_NAMES)) != FEATURE_COUNT:
    raise RuntimeError("V441 feature schema changed")


__all__ = [
    "BLOCK_WIDTHS",
    "COMPETITION",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "PrequentialPITGeometry",
    "batch_features",
]
