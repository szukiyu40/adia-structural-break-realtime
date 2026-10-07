"""Causal EvoForest-WM-inspired patch features for ADIA.

Only representation families absent from the established V4 bank are kept.
Every online row uses a trailing 64-tick patch, normalized and calibrated by
same-series historical patches.  The implementation is deterministic NumPy /
SciPy and never inspects an unseen suffix or another online series.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import hilbert


COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
SCHEMA_VERSION = "adia-v363-evoforest-patch-bank-v1"
WINDOW = 64
MAX_REFERENCE_WINDOWS = 64
EPSILON = 1e-8
VALUE_CLIP = 12.0
FEATURE_CLIP = 20.0
DECAY = 0.99

FAMILY_WIDTHS = (
    ("spectral", 2),
    ("turning", 2),
    ("trf_gausswin", 12),
    ("hilbert_env", 2),
    ("morphology_updown", 4),
    ("fftbands", 6),
    ("perm_hjorth", 3),
    ("curvature", 3),
    ("acf_first", 3),
    ("histogram_mode", 3),
    ("ricker_wavelet", 4),
)

RAW_FEATURE_NAMES = (
    "spectral_centroid",
    "spectral_entropy",
    "turning_rate",
    "positive_difference_fraction",
    *(f"trf_gausswin_{index:02d}" for index in range(12)),
    "hilbert_envelope_cv",
    "hilbert_envelope_total_variation",
    "up_max",
    "down_max",
    "up_down_log_max_ratio",
    "up_down_log_energy_ratio",
    *(f"fft_log_band_{index}" for index in range(6)),
    "permutation_entropy_3",
    "hjorth_mobility",
    "hjorth_complexity",
    "curvature_abs_mean",
    "curvature_abs_max",
    "curvature_log_energy_ratio",
    "acf_first_minimum",
    "acf_first_zero",
    "acf_value_after_first_minimum",
    "histogram_soft_mode",
    "histogram_mode_minus_median",
    "histogram_central_mass",
    *(f"ricker_ppv_scale_{scale}" for scale in (15, 25, 40, 60)),
)

FEATURE_NAMES = tuple(
    f"wm_{transport}_{name}"
    for transport in ("signed", "absolute", "running_absmax", "decayed_absmax")
    for name in RAW_FEATURE_NAMES
) + ("wm_log1p_online_n", "wm_log1p_history_n")
RAW_FEATURE_COUNT = len(RAW_FEATURE_NAMES)
FEATURE_COUNT = len(FEATURE_NAMES)


def _gaussian_windows() -> np.ndarray:
    centers = np.tile(np.asarray((0.10, 0.30, 0.50, 0.70, 0.90)), 3)[:12]
    widths = np.repeat(np.asarray((0.06, 0.14, 0.30)), 5)[:12]
    position = np.arange(WINDOW, dtype=np.float64) / WINDOW
    weights = np.exp(
        -((position[None, :] - centers[:, None]) ** 2)
        / (2.0 * widths[:, None] ** 2 + EPSILON)
    )
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), EPSILON)


GAUSSIAN_WINDOWS = _gaussian_windows()


def _ricker_kernels() -> tuple[np.ndarray, ...]:
    x = np.linspace(-7.0, 7.0, 15)
    kernels = []
    for scale in (1.5, 2.5, 4.0, 6.0):
        values = (1.0 - (x / scale) ** 2) * np.exp(-(x * x) / (2.0 * scale * scale))
        values -= values.mean()
        values /= max(float(np.abs(values).sum()), EPSILON)
        kernels.append(values)
    return tuple(kernels)


RICKER_KERNELS = _ricker_kernels()


def _finite(values: Sequence[float] | np.ndarray, replacement: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    output[~np.isfinite(output)] = replacement
    return output


def _patch_features(windows: np.ndarray) -> np.ndarray:
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != WINDOW:
        raise ValueError(f"V363 expects (*, {WINDOW}) windows")
    parts: list[np.ndarray] = []

    amplitude = np.abs(np.fft.rfft(w, axis=1))[:, 1:]
    probability = amplitude / np.maximum(amplitude.sum(axis=1, keepdims=True), EPSILON)
    frequency = np.arange(1, probability.shape[1] + 1, dtype=np.float64) / probability.shape[1]
    parts.append(
        np.column_stack(
            (
                probability @ frequency,
                -np.sum(probability * np.log(probability + 1e-12), axis=1),
            )
        )
    )

    difference = np.diff(w, axis=1)
    parts.append(
        np.column_stack(
            (
                np.mean(difference[:, 1:] * difference[:, :-1] < 0.0, axis=1),
                np.mean(difference > 0.0, axis=1),
            )
        )
    )
    parts.append(w @ GAUSSIAN_WINDOWS.T)

    envelope = np.abs(hilbert(w, axis=1))
    envelope_mean = np.maximum(envelope.mean(axis=1), EPSILON)
    parts.append(
        np.column_stack(
            (
                envelope.std(axis=1) / envelope_mean,
                np.mean(np.abs(np.diff(envelope, axis=1)), axis=1) / envelope_mean,
            )
        )
    )

    positive = np.maximum(difference, 0.0)
    negative = np.maximum(-difference, 0.0)
    positive_max = positive.max(axis=1)
    negative_max = negative.max(axis=1)
    parts.append(
        np.column_stack(
            (
                positive_max,
                negative_max,
                np.log((positive_max + 1e-6) / (negative_max + 1e-6)),
                np.log(
                    (np.sum(positive * positive, axis=1) + 1e-6)
                    / (np.sum(negative * negative, axis=1) + 1e-6)
                ),
            )
        )
    )

    power = amplitude * amplitude
    power /= np.maximum(power.sum(axis=1, keepdims=True), EPSILON)
    bands = np.array_split(np.arange(power.shape[1]), 6)
    parts.append(
        np.column_stack(
            [np.log(np.maximum(power[:, indices].sum(axis=1), EPSILON)) for indices in bands]
        )
    )

    code = (
        (w[:, :-2] < w[:, 1:-1]).astype(np.uint8) * 4
        + (w[:, 1:-1] < w[:, 2:]).astype(np.uint8) * 2
        + (w[:, :-2] < w[:, 2:]).astype(np.uint8)
    )
    pattern_probability = np.column_stack([np.mean(code == value, axis=1) for value in range(8)])
    permutation_entropy = -np.sum(
        pattern_probability * np.log(pattern_probability + 1e-12), axis=1
    ) / math.log(6.0)
    second_difference = np.diff(w, n=2, axis=1)
    mobility = np.sqrt(
        (np.var(difference, axis=1) + EPSILON) / (np.var(w, axis=1) + EPSILON)
    )
    complexity = np.sqrt(
        (np.var(second_difference, axis=1) + EPSILON)
        / (np.var(difference, axis=1) + EPSILON)
    ) / np.maximum(mobility, EPSILON)
    parts.append(np.column_stack((permutation_entropy, mobility, complexity)))

    curvature_positive = np.maximum(second_difference, 0.0)
    curvature_negative = np.minimum(second_difference, 0.0)
    parts.append(
        np.column_stack(
            (
                np.mean(np.abs(second_difference), axis=1),
                np.max(np.abs(second_difference), axis=1),
                np.log(
                    (np.sum(curvature_positive * curvature_positive, axis=1) + 1e-6)
                    / (np.sum(curvature_negative * curvature_negative, axis=1) + 1e-6)
                ),
            )
        )
    )

    centered = w - w.mean(axis=1, keepdims=True)
    energy = np.maximum(np.sum(centered * centered, axis=1), EPSILON)
    acf = np.column_stack(
        [np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / energy for lag in range(1, 17)]
    )
    local_minimum = (acf[:, 1:-1] < acf[:, :-2]) & (acf[:, 1:-1] <= acf[:, 2:])
    has_minimum = local_minimum.any(axis=1)
    minimum_index = np.argmax(local_minimum, axis=1) + 1
    minimum_index = np.where(has_minimum, minimum_index, acf.shape[1] - 1)
    negative_acf = acf < 0.0
    has_zero = negative_acf.any(axis=1)
    zero_index = np.argmax(negative_acf, axis=1)
    zero_index = np.where(has_zero, zero_index, acf.shape[1] - 1)
    after_index = np.minimum(minimum_index + 1, acf.shape[1] - 1)
    parts.append(
        np.column_stack(
            (
                (minimum_index + 1) / WINDOW,
                (zero_index + 1) / WINDOW,
                acf[np.arange(w.shape[0]), after_index],
            )
        )
    )

    z = centered / np.maximum(w.std(axis=1, keepdims=True), 1e-6)
    centers = np.linspace(-2.5, 2.5, 10)
    soft = np.exp(-0.5 * ((centers[None, None, :] - z[:, :, None]) / (5.0 / 9.0)) ** 2).sum(axis=1)
    soft = np.exp(4.0 * (soft - soft.max(axis=1, keepdims=True)))
    soft /= np.maximum(soft.sum(axis=1, keepdims=True), EPSILON)
    mode = soft @ centers
    parts.append(
        np.column_stack(
            (
                mode,
                mode - np.median(z, axis=1),
                np.mean(np.abs(z) < 0.5, axis=1),
            )
        )
    )

    parts.append(
        np.column_stack(
            [np.mean(convolve1d(w, kernel, axis=1, mode="constant") > 0.0, axis=1) for kernel in RICKER_KERNELS]
        )
    )
    output = np.column_stack(parts)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP)
    np.clip(output, -FEATURE_CLIP, FEATURE_CLIP, out=output)
    if output.shape != (w.shape[0], RAW_FEATURE_COUNT):
        raise RuntimeError(f"V363 raw feature shape changed: {output.shape}")
    return output


def _reference_windows(history_z: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(history_z, WINDOW)
    if windows.shape[0] <= MAX_REFERENCE_WINDOWS:
        return windows
    indices = np.unique(
        np.rint(np.linspace(0, windows.shape[0] - 1, MAX_REFERENCE_WINDOWS)).astype(np.int64)
    )
    return windows[indices]


def batch_features(
    historical: Sequence[float] | np.ndarray,
    online: Sequence[float] | np.ndarray,
) -> np.ndarray:
    history_raw = np.asarray(historical, dtype=np.float64).reshape(-1)
    if history_raw.size < WINDOW:
        raise ValueError(f"V363 requires at least {WINDOW} history ticks")
    finite = np.isfinite(history_raw)
    replacement = float(np.median(history_raw[finite])) if finite.any() else 0.0
    history = _finite(history_raw, replacement)
    observed = _finite(online, replacement)
    if observed.size == 0:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    center = float(np.median(history))
    scale = max(float(1.4826 * np.median(np.abs(history - center))), float(np.std(history)), EPSILON)
    history_z = np.clip((history - center) / scale, -VALUE_CLIP, VALUE_CLIP)
    online_z = np.clip((observed - center) / scale, -VALUE_CLIP, VALUE_CLIP)

    reference = _patch_features(_reference_windows(history_z))
    reference_center = np.median(reference, axis=0)
    reference_scale = 1.4826 * np.median(np.abs(reference - reference_center), axis=0)
    fallback = np.std(reference, axis=0)
    reference_scale = np.where(reference_scale > 1e-6, reference_scale, fallback)
    reference_scale = np.maximum(reference_scale, 1e-4)

    joined = np.r_[history_z[-(WINDOW - 1) :], online_z]
    online_windows = np.lib.stride_tricks.sliding_window_view(joined, WINDOW)
    raw = _patch_features(online_windows)
    signed = np.clip((raw - reference_center) / reference_scale, -FEATURE_CLIP, FEATURE_CLIP)
    absolute = np.abs(signed)
    running = np.maximum.accumulate(absolute, axis=0)
    decayed = np.empty_like(absolute)
    state = np.zeros(RAW_FEATURE_COUNT, dtype=np.float64)
    for index in range(absolute.shape[0]):
        state = np.maximum(DECAY * state, absolute[index])
        decayed[index] = state
    context = np.column_stack(
        (
            np.log1p(np.arange(1, observed.size + 1, dtype=np.float64)),
            np.full(observed.size, np.log1p(history.size), dtype=np.float64),
        )
    )
    output = np.column_stack((signed, absolute, running, decayed, context))
    np.nan_to_num(output, copy=False, nan=0.0, posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP)
    np.clip(output, -FEATURE_CLIP, FEATURE_CLIP, out=output)
    if output.shape != (observed.size, FEATURE_COUNT) or not np.isfinite(output).all():
        raise RuntimeError("V363 invalid feature matrix")
    return output.astype(np.float32)


def feature_contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "window": WINDOW,
        "maximum_history_reference_windows": MAX_REFERENCE_WINDOWS,
        "raw_feature_count": RAW_FEATURE_COUNT,
        "feature_count": FEATURE_COUNT,
        "family_widths": [list(value) for value in FAMILY_WIDTHS],
        "feature_names": list(FEATURE_NAMES),
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_series_online_values_used": False,
    }


if RAW_FEATURE_COUNT != 44 or FEATURE_COUNT != 178 or len(set(FEATURE_NAMES)) != FEATURE_COUNT:
    raise RuntimeError(f"V363 schema changed: raw={RAW_FEATURE_COUNT}, total={FEATURE_COUNT}")
