"""Causal multi-LAG and intrinsic-time features for ADIA V322 EDA."""

from __future__ import annotations

import numpy as np


FIXED_LAGS = (1, 2, 4, 8, 16, 32, 64)
ENERGY_LAGS = (1, 4, 16, 64)
ACF_LAGS = (1, 2, 4, 8, 16, 32)
MIN_PERIOD = 4
MAX_PERIOD = 128
CLIP = 12.0

FEATURE_NAMES = (
    *(f"lag_point_surprise_{lag}" for lag in FIXED_LAGS),
    *(f"lag_energy_shift_{lag}" for lag in ENERGY_LAGS),
    *(f"lag_acf_shift_{lag}" for lag in ENERGY_LAGS),
    "multilag_point_mean",
    "multilag_point_max",
    "multilag_point_top2",
    "multilag_short_long_gap",
    "multilag_scale_dispersion",
    "acf_vector_shift_64",
    "acf_vector_shift_128",
    "period_strength",
    "period_stability",
    "adaptive_seasonal_point",
    "adaptive_seasonal_energy",
    "adaptive_phase_point",
    "adaptive_phase_window",
    "adaptive_period_acf_loss",
    "period_gated_seasonal_point",
    "period_gated_seasonal_energy",
    "period_gated_phase_point",
    "period_gated_phase_window",
    "period_gated_acf_loss",
    "topperiod_ensemble_point",
    "period_gated_topperiod_ensemble",
)

FEATURE_GROUPS = (
    *("fixed_lag_point" for _ in FIXED_LAGS),
    *("fixed_lag_energy" for _ in ENERGY_LAGS),
    *("fixed_lag_acf" for _ in ENERGY_LAGS),
    *("multilag_geometry" for _ in range(7)),
    *("period_static" for _ in range(2)),
    *("adaptive_period" for _ in range(5)),
    *("period_gated" for _ in range(5)),
    *("topperiod_ensemble" for _ in range(2)),
)


def _clean(values: np.ndarray, replacement: float | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(array)
    if replacement is None:
        replacement = float(np.median(array[finite])) if finite.any() else 0.0
    array[~finite] = replacement
    return array


def _acf(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or values.size <= lag + 2:
        return 0.0
    left = values[lag:]
    right = values[:-lag]
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _acf_fft(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - np.mean(values)
    size = centered.size
    transform_size = 1 << int(np.ceil(np.log2(max(2 * size, 2))))
    spectrum = np.fft.rfft(centered, n=transform_size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=transform_size)[: max_lag + 1]
    counts = np.arange(size, size - max_lag - 1, -1, dtype=np.float64)
    covariance = covariance / np.maximum(counts, 1.0)
    denominator = max(float(covariance[0]), 1e-12)
    return np.clip(covariance / denominator, -1.0, 1.0)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    ordinary = float(np.std(values))
    return median, max(mad, 0.25 * ordinary, 1e-5)


def _period_definition(history_z: np.ndarray) -> tuple[int, float, float, tuple[int, ...]]:
    acf = _acf_fft(history_z, MAX_PERIOD)
    candidates = acf[MIN_PERIOD : MAX_PERIOD + 1]
    best_strength = max(float(np.max(candidates)), 0.0)
    eligible = np.flatnonzero(candidates >= best_strength - 0.03) + MIN_PERIOD
    period = int(eligible[0]) if eligible.size else MIN_PERIOD
    first = _acf(history_z[: history_z.size // 2], period)
    second = _acf(history_z[history_z.size // 2 :], period)
    stability = max(0.0, min(first, second, float(acf[period])))

    order = np.argsort(candidates)[::-1] + MIN_PERIOD
    selected: list[int] = []
    for lag in order:
        lag = int(lag)
        if all(abs(lag - retained) > 2 for retained in selected):
            selected.append(lag)
        if len(selected) == 3:
            break
    while len(selected) < 3:
        selected.append(period)
    return period, best_strength, stability, tuple(selected)


def _history_resources(history_z: np.ndarray) -> dict[str, object]:
    lag_null: dict[int, tuple[float, float, float]] = {}
    history_acf: dict[int, float] = {}
    for lag in FIXED_LAGS:
        differences = history_z[lag:] - history_z[:-lag]
        center, scale = _robust_location_scale(differences)
        lag_null[lag] = (center, scale, max(float(np.mean(differences * differences)), 1e-8))
        history_acf[lag] = _acf(history_z, lag)

    period, strength, stability, top_periods = _period_definition(history_z)
    phase_mean = np.zeros(period, dtype=np.float64)
    phase_scale = np.ones(period, dtype=np.float64)
    indices = np.arange(history_z.size, dtype=np.int64) % period
    for phase in range(period):
        sample = history_z[indices == phase]
        if sample.size:
            phase_mean[phase], phase_scale[phase] = _robust_location_scale(sample)
    confidence = float(np.clip((strength - 0.10) / 0.50, 0.0, 1.0) * np.clip(stability / 0.40, 0.0, 1.0))
    return {
        "lag_null": lag_null,
        "history_acf": history_acf,
        "period": period,
        "period_strength": strength,
        "period_stability": stability,
        "period_confidence": confidence,
        "phase_mean": phase_mean,
        "phase_scale": phase_scale,
        "top_periods": top_periods,
    }


def _lag_energy_shift(values: np.ndarray, lag: int, reference_energy: float) -> float:
    width = max(32, 2 * lag)
    end = values[-(width + lag) :]
    differences = end[lag:] - end[:-lag]
    energy = max(float(np.mean(differences * differences)), 1e-8)
    return float(min(abs(np.log(energy / reference_energy)), 12.0))


def _acf_vector_shift(values: np.ndarray, resources: dict[str, object], width: int) -> float:
    recent = values[-width:]
    reference = resources["history_acf"]
    shifts = [abs(_acf(recent, lag) - float(reference[lag])) for lag in ACF_LAGS]
    return float(np.sqrt(np.mean(np.square(shifts))))


def _period_features(values: np.ndarray, history_size: int, resources: dict[str, object]) -> tuple[float, ...]:
    period = int(resources["period"])
    lag_null = resources["lag_null"]
    if period in lag_null:
        diff_center, diff_scale, reference_energy = lag_null[period]
    else:
        history = values[:history_size]
        difference = history[period:] - history[:-period]
        diff_center, diff_scale = _robust_location_scale(difference)
        reference_energy = max(float(np.mean(difference * difference)), 1e-8)

    point = float(np.log1p(min(abs((values[-1] - values[-1 - period] - diff_center) / diff_scale), 30.0)))
    energy = _lag_energy_shift(values, period, reference_energy)
    phase_mean = np.asarray(resources["phase_mean"], dtype=np.float64)
    phase_scale = np.asarray(resources["phase_scale"], dtype=np.float64)
    current_index = values.size - 1
    phase = current_index % period
    phase_point = float(np.log1p(min(abs((values[-1] - phase_mean[phase]) / phase_scale[phase]), 30.0)))
    width = min(32, values.size - history_size if values.size > history_size else 1)
    positions = np.arange(values.size - width, values.size, dtype=np.int64)
    phase_z = np.abs((values[-width:] - phase_mean[positions % period]) / phase_scale[positions % period])
    phase_window = float(np.mean(np.log1p(np.clip(phase_z, 0.0, 30.0))))
    recent_width = min(values.size, max(64, min(4 * period, 256)))
    acf_loss = abs(_acf(values[-recent_width:], period) - float(resources["history_acf"].get(period, _acf(values[:history_size], period))))

    predictions = np.asarray([values[-1 - int(lag)] for lag in resources["top_periods"]], dtype=np.float64)
    ensemble = float(np.log1p(min(abs(values[-1] - float(np.mean(predictions))) / max(diff_scale, 1e-5), 30.0)))
    return point, energy, phase_point, phase_window, float(acf_loss), ensemble


def checkpoint_features(history: np.ndarray, online: np.ndarray, checkpoints: tuple[int, ...]) -> dict[int, np.ndarray]:
    history_clean = _clean(history)
    center, scale = _robust_location_scale(history_clean)
    history_z = np.clip((history_clean - center) / scale, -CLIP, CLIP)
    online_z = np.clip((_clean(online, center) - center) / scale, -CLIP, CLIP)
    resources = _history_resources(history_z)
    output: dict[int, np.ndarray] = {}

    for checkpoint in checkpoints:
        if checkpoint < 0 or checkpoint >= online_z.size:
            continue
        values = np.concatenate((history_z, online_z[: checkpoint + 1]))
        point_scores = []
        for lag in FIXED_LAGS:
            lag_center, lag_scale, _ = resources["lag_null"][lag]
            raw = abs((values[-1] - values[-1 - lag] - lag_center) / lag_scale)
            point_scores.append(float(np.log1p(min(raw, 30.0))))
        energies = [
            _lag_energy_shift(values, lag, float(resources["lag_null"][lag][2]))
            for lag in ENERGY_LAGS
        ]
        acf_shifts = [
            abs(_acf(values[-max(64, 2 * lag) :], lag) - float(resources["history_acf"][lag]))
            for lag in ENERGY_LAGS
        ]
        ordered = np.sort(np.asarray(point_scores, dtype=np.float64))
        geometry = (
            float(np.mean(ordered)),
            float(ordered[-1]),
            float(np.mean(ordered[-2:])),
            float(abs(np.mean(ordered[:3]) - np.mean(ordered[-3:]))),
            float(np.std(ordered)),
            _acf_vector_shift(values, resources, 64),
            _acf_vector_shift(values, resources, 128),
        )
        periodic = _period_features(values, history_z.size, resources)
        confidence = float(resources["period_confidence"])
        period_values = (
            float(resources["period_strength"]),
            float(resources["period_stability"]),
            *periodic[:5],
            *(confidence * value for value in periodic[:5]),
            periodic[5],
            confidence * periodic[5],
        )
        row = np.asarray((*point_scores, *energies, *acf_shifts, *geometry, *period_values), dtype=np.float64)
        if row.shape != (len(FEATURE_NAMES),) or not np.isfinite(row).all():
            raise RuntimeError("invalid V322 feature row")
        output[int(checkpoint)] = row
    return output

