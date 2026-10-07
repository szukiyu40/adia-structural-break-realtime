"""Lean causal component states for the frozen ADIA V780 formula.

The module contains target-free streaming equivalents of the only research
operators required by V780.  It deliberately contains no submission resource
selection or final score coefficients; those stay in the production wrapper.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Sequence

import numpy as np


EPSILON = 1e-7
CURVATURE_WINDOW = 64
CURVATURE_REFERENCE_WINDOWS = 64
CURVATURE_CLIP = 20.0
POSITIONAL_WIDTH = 94
SCOREPATH_CALIBRATION_START = 64
SCOREPATH_BOUNDARY = 128
SCOREPATH_THRESHOLDS = np.asarray(
    (0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95), dtype=np.float64
)
SCOREPATH_Q75_INDEX = 3
NLAR_HISTORY_LIMIT = 2048
NLAR_LAG_ORDER = 4
NLAR_HIDDEN_UNITS = 12
NLAR_RIDGE = 0.01
NLAR_SEED = 20260817 + 616
NLAR_WINDOWS = (64, 128, 256)
NLAR_Z_CLIP = 8.0
NLAR_SCORE_CLIP = 12.0
NLAR_P_FLOOR = 1e-6
NLAR_EPROCESS_POWER = 0.5
NLAR_EPROCESS_DECAY = 0.995


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(result)
    replacement = float(np.median(result[finite])) if np.any(finite) else 0.0
    result[~finite] = replacement
    return result


def _probability_logit(value: float) -> float:
    probability = float(np.clip(value, EPSILON, 1.0 - EPSILON))
    return float(np.log(probability) - np.log1p(-probability))


class LeanPositionalState:
    """Allocation-reduced exact transport of V377 ``PositionalState``."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
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
        self.cumulative = [[0.0] for _ in range(4)]
        self.cumulative_square = [[0.0] for _ in range(4)]

    def _moments(self, stream: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros(bins, dtype=np.float64)
        variances = np.zeros(bins, dtype=np.float64)
        cumulative = self.cumulative[stream]
        square = self.cumulative_square[stream]
        for index in range(bins):
            left = (self.count * index) // bins
            right = (self.count * (index + 1)) // bins
            width = right - left
            if width:
                total = cumulative[right] - cumulative[left]
                mean = total / width
                means[index] = mean
                variances[index] = max(
                    (square[right] - square[left]) / width - mean * mean, 0.0
                )
        return means, variances

    @staticmethod
    def _aggregate(values: np.ndarray) -> np.ndarray:
        differences = np.diff(values)
        path = float(np.sum(np.abs(differences)))
        endpoint = abs(float(values[-1] - values[0]))
        curvature = (
            float(np.max(np.abs(np.diff(values, n=2))))
            if values.size >= 3
            else 0.0
        )
        return np.asarray(
            (
                float(np.max(values) - np.min(values)),
                path,
                endpoint,
                max(path - endpoint, 0.0),
                curvature,
            )
        )

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.center
        z = float(np.clip((clean - self.center) / self.scale, -12.0, 12.0))
        raw_difference = z - self.previous_z
        difference = float(
            np.clip(
                (raw_difference - self.difference_center) / self.difference_scale,
                -12.0,
                12.0,
            )
        )
        self.previous_z = z
        current = (
            z,
            abs(z) - self.history_abs_mean,
            z * z - self.history_square_mean,
            difference,
        )
        self.count += 1
        for stream, item in enumerate(current):
            self.cumulative[stream].append(self.cumulative[stream][-1] + item)
            self.cumulative_square[stream].append(
                self.cumulative_square[stream][-1] + item * item
            )

        means8: list[np.ndarray] = []
        variances: dict[tuple[int, int], np.ndarray] = {}
        parts: list[np.ndarray] = []
        for stream in range(4):
            for bins in (4, 8):
                means, variance = self._moments(stream, bins)
                parts.append(means)
                variances[(stream, bins)] = variance
                if bins == 8:
                    means8.append(means)
        for stream in (0, 3):
            for bins in (4, 8):
                parts.append(np.log(variances[(stream, bins)] + 1e-8))
        parts.extend(self._aggregate(values) for values in means8)
        parts.append(
            np.asarray((math.log1p(self.count), math.log1p(self.history_size)))
        )
        output = np.concatenate(parts)
        np.nan_to_num(output, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(output, -20.0, 20.0, out=output)
        if output.shape != (POSITIONAL_WIDTH,) or not np.isfinite(output).all():
            raise RuntimeError("V784 positional row changed")
        return output.astype(np.float32)


def _curvature_raw(windows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(windows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != CURVATURE_WINDOW:
        raise ValueError("V784 curvature expects 64-tick windows")
    second = np.diff(matrix, n=2, axis=1)
    absolute = np.abs(second)
    positive = np.maximum(second, 0.0)
    negative = np.minimum(second, 0.0)
    output = np.column_stack(
        (
            np.mean(absolute, axis=1),
            np.max(absolute, axis=1),
            np.log(
                (np.sum(positive * positive, axis=1) + 1e-6)
                / (np.sum(negative * negative, axis=1) + 1e-6)
            ),
        )
    )
    np.nan_to_num(
        output,
        copy=False,
        nan=0.0,
        posinf=CURVATURE_CLIP,
        neginf=-CURVATURE_CLIP,
    )
    np.clip(output, -CURVATURE_CLIP, CURVATURE_CLIP, out=output)
    return output


class LeanCurvatureState:
    """Exact V748-selected 32 columns without FFT/Hilbert/ACF work."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
        history = _finite(historical)
        self.replacement = float(np.median(history))
        center = float(np.median(history))
        scale = max(
            float(1.4826 * np.median(np.abs(history - center))),
            float(np.std(history)),
            1e-8,
        )
        history_z = np.clip((history - center) / scale, -12.0, 12.0)
        windows = np.lib.stride_tricks.sliding_window_view(
            history_z, CURVATURE_WINDOW
        )
        if windows.shape[0] > CURVATURE_REFERENCE_WINDOWS:
            indices = np.unique(
                np.rint(
                    np.linspace(
                        0,
                        windows.shape[0] - 1,
                        CURVATURE_REFERENCE_WINDOWS,
                    )
                ).astype(np.int64)
            )
            windows = windows[indices]
        reference = _curvature_raw(windows)
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
        self.window = list(history_z[-(CURVATURE_WINDOW - 1) :])
        self.count = 0

        self.signed_sum = 0.0
        self.signed_fast = 0.0
        self.signed_slow = 0.0
        self.signed_positive_sum = 0.0
        self.signed_negative_sum = 0.0
        self.signed_positive_count = 0.0
        self.signed_minimum = math.inf
        self.signed_maximum = -math.inf

        self.absolute_sum = np.zeros(3, dtype=np.float64)
        self.absolute_fast = np.zeros(3, dtype=np.float64)
        self.absolute_slow = np.zeros(3, dtype=np.float64)
        self.absolute_above_count = np.zeros(3, dtype=np.float64)
        self.absolute_excess_sum = np.zeros(3, dtype=np.float64)
        self.absolute_maximum = np.full(3, -np.inf, dtype=np.float64)
        self.absolute_age = np.zeros(3, dtype=np.float64)

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.replacement
        z = float(np.clip((clean - self.center) / self.scale, -12.0, 12.0))
        self.window.append(z)
        if len(self.window) > CURVATURE_WINDOW:
            self.window.pop(0)
        raw = _curvature_raw(np.asarray(self.window, dtype=np.float64)[None, :])[0]
        calibrated = np.clip(
            (raw - self.reference_center) / self.reference_scale,
            -CURVATURE_CLIP,
            CURVATURE_CLIP,
        ).astype(np.float32).astype(np.float64)
        signed = float(calibrated[0])
        # V373 absolute order is raw31, raw33, raw32.
        absolute = np.abs(calibrated[[0, 2, 1]])
        self.count += 1
        denominator = float(self.count)

        self.signed_sum += signed
        self.signed_fast = 0.875 * self.signed_fast + 0.125 * signed
        self.signed_slow = (63.0 / 64.0) * self.signed_slow + signed / 64.0
        self.signed_positive_sum += max(signed, 0.0)
        self.signed_negative_sum += max(-signed, 0.0)
        self.signed_positive_count += signed > 0.0
        self.signed_minimum = min(self.signed_minimum, signed)
        self.signed_maximum = max(self.signed_maximum, signed)
        signed_output = np.asarray(
            (
                self.signed_sum / denominator,
                self.signed_fast,
                self.signed_slow,
                self.signed_fast - self.signed_slow,
                self.signed_positive_sum / denominator,
                self.signed_negative_sum / denominator,
                self.signed_positive_count / denominator,
                self.signed_maximum - self.signed_minimum,
            ),
            dtype=np.float64,
        )

        self.absolute_sum += absolute
        self.absolute_fast = 0.875 * self.absolute_fast + 0.125 * absolute
        self.absolute_slow = (63.0 / 64.0) * self.absolute_slow + absolute / 64.0
        self.absolute_above_count += absolute > 1.0
        self.absolute_excess_sum += np.maximum(absolute - 1.0, 0.0)
        new_maximum = absolute >= self.absolute_maximum
        self.absolute_maximum = np.maximum(self.absolute_maximum, absolute)
        self.absolute_age = np.where(new_maximum, 0.0, self.absolute_age + 1.0)
        absolute_output = np.column_stack(
            (
                self.absolute_sum / denominator,
                self.absolute_fast,
                self.absolute_slow,
                self.absolute_fast - self.absolute_slow,
                self.absolute_above_count / denominator,
                self.absolute_excess_sum / denominator,
                self.absolute_maximum - absolute,
                np.log1p(self.absolute_age),
            )
        ).reshape(-1)
        output = np.concatenate((signed_output, absolute_output))
        np.nan_to_num(output, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(output, -20.0, 20.0, out=output)
        if output.shape != (32,) or not np.isfinite(output).all():
            raise RuntimeError("V784 curvature row changed")
        return output.astype(np.float32)


class ScorePathState:
    """Exact V575 open energy plus V598 recursive soft occupation."""

    def __init__(self) -> None:
        self.calibration_probability: list[float] = []
        self.calibration: np.ndarray | None = None
        self.current_run = np.zeros(SCOREPATH_THRESHOLDS.size, dtype=np.float64)
        self.longest_run = np.zeros_like(self.current_run)
        self.occupation = np.zeros_like(self.current_run)
        self.total_excess = np.zeros_like(self.current_run)
        self.open_excess = np.zeros_like(self.current_run)
        self.pit_total = 0.0
        self.elapsed = 0

    def update(self, probability: float, tick: int) -> tuple[float, float]:
        tick = int(tick)
        value = float(np.clip(probability, EPSILON, 1.0 - EPSILON))
        if SCOREPATH_CALIBRATION_START <= tick < SCOREPATH_BOUNDARY:
            self.calibration_probability.append(value)
            if tick == SCOREPATH_BOUNDARY - 1:
                probabilities = np.asarray(
                    self.calibration_probability, dtype=np.float64
                )
                logits = np.log(probabilities) - np.log1p(-probabilities)
                self.calibration = np.sort(logits, kind="stable")
                if self.calibration.shape != (64,):
                    raise RuntimeError("V784 score-path calibration changed")
        if tick < SCOREPATH_BOUNDARY:
            return 0.0, 0.5
        if self.calibration is None:
            raise RuntimeError("V784 score-path started before calibration froze")

        logit = float(np.log(value) - np.log1p(-value))
        left = int(np.searchsorted(self.calibration, logit, side="left"))
        right = int(np.searchsorted(self.calibration, logit, side="right"))
        pit = (left + right) / (2.0 * self.calibration.size)
        active = pit >= SCOREPATH_THRESHOLDS
        excess = np.maximum(pit - SCOREPATH_THRESHOLDS, 0.0)
        self.current_run = np.where(active, self.current_run + 1.0, 0.0)
        self.longest_run = np.maximum(self.longest_run, self.current_run)
        self.occupation += active
        self.total_excess += excess
        self.open_excess = np.where(active, self.open_excess + excess, 0.0)
        self.elapsed += 1
        elapsed = float(self.elapsed)
        run_fraction = self.current_run / elapsed
        normalized_excess = excess / np.maximum(
            1.0 - SCOREPATH_THRESHOLDS, 1e-12
        )
        open_energy = float(
            np.float32(
                np.clip(
                    np.mean(np.sqrt(run_fraction) * normalized_excess), 0.0, 1.0
                )
            )
        )
        self.pit_total += pit
        soft_occupation = float(np.float32(self.pit_total / elapsed))
        return open_energy, soft_occupation


def _nlar_frozen_map() -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(NLAR_SEED)
    weight = random.standard_normal((NLAR_HIDDEN_UNITS, NLAR_LAG_ORDER))
    weight /= np.maximum(
        np.sqrt(np.sum(weight * weight, axis=1, keepdims=True)), 1e-12
    )
    bias = random.uniform(-1.0, 1.0, NLAR_HIDDEN_UNITS)
    return np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)


def _nlar_basis(sequence: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequence, dtype=np.float64).reshape(-1)
    count = values.size - NLAR_LAG_ORDER
    lags = np.column_stack(
        [
            values[NLAR_LAG_ORDER - lag : values.size - lag]
            for lag in range(1, NLAR_LAG_ORDER + 1)
        ]
    )
    hidden = np.tanh(lags @ weight.T + bias)
    return (
        np.column_stack((np.ones(count, dtype=np.float64), lags, hidden)),
        values[NLAR_LAG_ORDER:],
    )


def _nlar_location_scale(history: np.ndarray) -> tuple[float, float]:
    center = float(np.median(history))
    q25, q75 = np.quantile(history, (0.25, 0.75), method="linear")
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1e-10:
        scale = float(1.4826 * np.median(np.abs(history - center)))
    if not math.isfinite(scale) or scale < 1e-10:
        scale = float(np.std(history, ddof=1))
    if not math.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return center, scale


def _nlar_residual_scale(residual: np.ndarray) -> tuple[float, float]:
    center = float(np.median(residual))
    q25, q75 = np.quantile(residual, (0.25, 0.75), method="linear")
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1e-8:
        scale = float(1.4826 * np.median(np.abs(residual - center)))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(residual, ddof=1))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return center, scale


def _nlar_studentizer(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(history, axis=0, dtype=np.float64)
    centered = history - center
    mad = 1.4826 * np.median(np.abs(centered), axis=0)
    standard = np.std(history, axis=0, ddof=1)
    scale = np.maximum(mad, 0.25 * standard)
    active = np.isfinite(scale) & (scale > 1e-8)
    return center, np.where(active, scale, 1.0), active


def _nlar_window_norms(history: np.ndarray, width: int) -> np.ndarray:
    prefix = np.empty((history.shape[0] + 1, history.shape[1]), dtype=np.float64)
    prefix[0] = 0.0
    prefix[1:] = np.cumsum(history, axis=0, dtype=np.float64)
    ends = np.arange(width, history.shape[0] + 1, dtype=np.int64)
    total = prefix[ends] - prefix[ends - width]
    return np.sqrt(np.sum(total * total, axis=1, dtype=np.float64)) / math.sqrt(
        float(width * history.shape[1])
    )


class _CumulativeVectorPath:
    """Preserve V616's exact cumsum-then-subtract arithmetic order."""

    def __init__(self, history: np.ndarray):
        prefix = np.empty(
            (history.shape[0] + 1, history.shape[1]), dtype=np.float64
        )
        prefix[0] = 0.0
        prefix[1:] = np.cumsum(history, axis=0, dtype=np.float64)
        self.prefix: deque[np.ndarray] = deque(
            (row.copy() for row in prefix[-(max(NLAR_WINDOWS) + 1) :]),
            maxlen=max(NLAR_WINDOWS) + 1,
        )

    def update(self, value: np.ndarray) -> tuple[float, ...]:
        current = self.prefix[-1] + value
        self.prefix.append(current)
        result = []
        for width in NLAR_WINDOWS:
            total = current - self.prefix[-width - 1]
            result.append(
                float(
                    np.sqrt(np.sum(total * total, dtype=np.float64))
                    / math.sqrt(float(width * total.size))
                )
            )
        return tuple(result)


class NLAREProcessState:
    """Incremental V616 three-source monitor followed by V645 e-process."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
        raw = np.asarray(historical, dtype=np.float64).reshape(-1)
        minimum = max(NLAR_WINDOWS) + NLAR_LAG_ORDER + 128
        if raw.size < minimum or not np.isfinite(raw).all():
            raise ValueError("V784 NLAR requires a long finite history")
        raw = raw[-NLAR_HISTORY_LIMIT:]
        self.center, self.scale = _nlar_location_scale(raw)
        history_z = np.clip(
            (raw - self.center) / self.scale, -NLAR_Z_CLIP, NLAR_Z_CLIP
        )
        self.weight, self.bias = _nlar_frozen_map()
        phi, target = _nlar_basis(history_z, self.weight, self.bias)
        penalty = np.eye(phi.shape[1], dtype=np.float64) * NLAR_RIDGE
        penalty[0, 0] = 0.0
        gram = phi.T @ phi + penalty
        moment = phi.T @ target
        try:
            self.beta = np.linalg.solve(gram, moment)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.lstsq(gram, moment, rcond=None)[0]
        raw_residual = target - phi @ self.beta
        self.residual_center, self.residual_scale = _nlar_residual_scale(raw_residual)
        residual = np.clip(
            (raw_residual - self.residual_center) / self.residual_scale,
            -NLAR_Z_CLIP,
            NLAR_Z_CLIP,
        )

        mean_raw = phi * residual[:, None]
        variance_raw = phi * (residual[:, None] * residual[:, None] - 1.0)
        self.mean_center, self.mean_scale, self.mean_active = _nlar_studentizer(mean_raw)
        self.variance_center, self.variance_scale, self.variance_active = _nlar_studentizer(variance_raw)
        history_mean = (mean_raw - self.mean_center) / self.mean_scale
        history_variance = (variance_raw - self.variance_center) / self.variance_scale
        history_mean[:, ~self.mean_active] = 0.0
        history_variance[:, ~self.variance_active] = 0.0
        history_mean = np.clip(history_mean, -NLAR_SCORE_CLIP, NLAR_SCORE_CLIP)
        history_variance = np.clip(
            history_variance, -NLAR_SCORE_CLIP, NLAR_SCORE_CLIP
        )

        self.mean_null = tuple(
            np.sort(_nlar_window_norms(history_mean, width))
            for width in NLAR_WINDOWS
        )
        self.variance_null = tuple(
            np.sort(_nlar_window_norms(history_variance, width))
            for width in NLAR_WINDOWS
        )
        self.mean_path = _CumulativeVectorPath(history_mean)
        self.variance_path = _CumulativeVectorPath(history_variance)
        self.absolute_residual_reference = np.sort(np.abs(residual))
        self.lags: deque[float] = deque(
            history_z[-NLAR_LAG_ORDER:].tolist(), maxlen=NLAR_LAG_ORDER
        )
        self.last_sources = np.zeros(3, dtype=np.float64)
        self.wealth = 0.0
        self.peak = 0.0

    @staticmethod
    def _pit(reference: np.ndarray, query: float) -> float:
        rank = int(np.searchsorted(reference, query, side="right"))
        return float((rank + 0.5) / (reference.size + 1.0))

    def update(self, raw_value: float) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("V784 NLAR online value is non-finite")
        z = float(
            np.clip((value - self.center) / self.scale, -NLAR_Z_CLIP, NLAR_Z_CLIP)
        )
        lag = np.asarray(tuple(reversed(self.lags)), dtype=np.float64)
        hidden = np.tanh(self.weight @ lag + self.bias)
        phi = np.concatenate((np.asarray((1.0,)), lag, hidden))
        residual = float(
            np.clip(
                (z - float(phi @ self.beta) - self.residual_center)
                / self.residual_scale,
                -NLAR_Z_CLIP,
                NLAR_Z_CLIP,
            )
        )
        self.lags.append(z)
        mean = (phi * residual - self.mean_center) / self.mean_scale
        variance = (
            phi * (residual * residual - 1.0) - self.variance_center
        ) / self.variance_scale
        mean[~self.mean_active] = 0.0
        variance[~self.variance_active] = 0.0
        mean = np.clip(mean, -NLAR_SCORE_CLIP, NLAR_SCORE_CLIP)
        variance = np.clip(variance, -NLAR_SCORE_CLIP, NLAR_SCORE_CLIP)

        mean_norms = self.mean_path.update(mean)
        variance_norms = self.variance_path.update(variance)
        mean_pits = [
            self._pit(reference, query)
            for reference, query in zip(self.mean_null, mean_norms, strict=True)
        ]
        variance_pits = [
            self._pit(reference, query)
            for reference, query in zip(
                self.variance_null, variance_norms, strict=True
            )
        ]
        # V616 persists its 22-column bank as float32 before V645 reloads the
        # three selected columns as float64.  Preserve that transport boundary.
        sources = np.asarray(
            (
                max(mean_pits),
                max(variance_pits),
                self._pit(self.absolute_residual_reference, abs(residual)),
            ),
            dtype=np.float32,
        ).astype(np.float64)
        self.last_sources = sources.copy()
        p_value = np.clip(1.0 - sources, NLAR_P_FLOOR, 1.0)
        evalue = NLAR_EPROCESS_POWER * np.power(
            p_value, NLAR_EPROCESS_POWER - 1.0
        )
        increment = float(np.log(np.mean(evalue)))
        self.wealth = max(0.0, self.wealth + increment)
        self.peak = max(self.wealth, NLAR_EPROCESS_DECAY * self.peak)
        result = float(np.float32(math.log1p(self.peak)))
        if not math.isfinite(result):
            raise RuntimeError("V784 NLAR e-process is non-finite")
        return result


__all__ = [
    "LeanCurvatureState",
    "LeanPositionalState",
    "NLAREProcessState",
    "ScorePathState",
]
