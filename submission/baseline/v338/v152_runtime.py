"""Causal conjugate-Bayes single-change evidence for ADIA real time.

This small supplement is deliberately distinct from the existing recursive
Page--CUSUM and adjacent-window matched-null candidates.  At every online
tick it evaluates a fixed, log-spaced set of possible post-change durations
under two conjugate models:

* a Gaussian mean shift with known unit variance and a Normal prior; and
* a joint mean/precision shift with a Normal--Gamma prior.

The duration is integrated out with a fixed log-uniform prior.  The resulting
mixture Bayes factor and the posterior mass on durations <= 16 are calibrated
by replaying only a held-out suffix of the same series' historical segment.
The two bounded channels are historical-rank-standardized raw observations
and AR(10) innovations.

No online row beyond the current prefix is inspected.  In particular, the
final online length, breakpoint, target, and any labelled break taxonomy are
absent from this module.  Online memory is fixed at ``max(DURATIONS)`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


DURATIONS = (4, 8, 16, 32, 64, 128)
CHANNEL_NAMES = ("raw_rank", "ar10_residual_rank")
SUMMARY_NAMES = (
    "mean_bf_surprise",
    "normal_gamma_bf_surprise",
    "normal_gamma_recent_surprise",
)
FEATURE_NAMES = tuple(
    f"bayes_{channel}_{summary}"
    for channel in CHANNEL_NAMES
    for summary in SUMMARY_NAMES
)

AR_ORDER = 10
N_BINS = 64
MIN_HISTORY = 1_024
REFERENCE_FRACTION = 0.60
MAX_CALIBRATION_POINTS = 512
ROBUST_CLIP = 12.0
CHANNEL_CLIP = 4.0
FEATURE_CLIP = 12.0

# H1 mean prior: mu ~ Normal(0, MEAN_PRIOR_VARIANCE).
MEAN_PRIOR_VARIANCE = 0.75**2
# H1 joint prior: precision ~ Gamma(shape, rate),
# mu | precision ~ Normal(0, 1 / (kappa * precision)).
NORMAL_GAMMA_SHAPE = 4.0
NORMAL_GAMMA_RATE = 4.0
NORMAL_GAMMA_KAPPA = 1.0 / MEAN_PRIOR_VARIANCE
# A fixed prior odds is used only to gate the posterior recent-duration mass.
CHANGE_PRIOR_PROBABILITY = 0.10
RECENT_DURATION_MAX = 16

EPS = 1e-8
CHANNEL_COUNT = len(CHANNEL_NAMES)
SUMMARY_COUNT = len(SUMMARY_NAMES)
FEATURE_COUNT = len(FEATURE_NAMES)
EXPECTED_FEATURE_COUNT = 6

DURATION_ARRAY = np.asarray(DURATIONS, dtype=np.int64)
LOG_DURATION_PRIOR = -np.log(DURATION_ARRAY.astype(np.float64))
LOG_GAMMA_INCREMENT = np.asarray(
    [
        math.lgamma(NORMAL_GAMMA_SHAPE + 0.5 * float(value))
        - math.lgamma(NORMAL_GAMMA_SHAPE)
        for value in DURATION_ARRAY
    ],
    dtype=np.float64,
)
LOG_CHANGE_PRIOR_ODDS = math.log(
    CHANGE_PRIOR_PROBABILITY / (1.0 - CHANGE_PRIOR_PROBABILITY)
)
MAX_DURATION = max(DURATIONS)
MIN_DURATION = min(DURATIONS)

if (
    FEATURE_COUNT != EXPECTED_FEATURE_COUNT
    or len(set(FEATURE_NAMES)) != FEATURE_COUNT
    or tuple(sorted(set(DURATIONS))) != DURATIONS
    or CHANNEL_COUNT != 2
    or SUMMARY_COUNT != 3
):
    raise RuntimeError("v8 Bayesian change-point feature schema changed")


def _finite_history(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Own, repair, and remotely left-pad a historical series."""

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
    left = values[:-lag] - values[:-lag].mean()
    right = values[lag:] - values[lag:].mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= EPS:
        return 0.0
    return float(left @ right / denominator)


def _fit_ar_yule_walker(values: np.ndarray) -> np.ndarray:
    acf = np.asarray(
        [1.0] + [_safe_acf(values, lag) for lag in range(1, AR_ORDER + 1)],
        dtype=np.float64,
    )
    toeplitz = np.empty((AR_ORDER, AR_ORDER), dtype=np.float64)
    for row in range(AR_ORDER):
        for column in range(AR_ORDER):
            toeplitz[row, column] = acf[abs(row - column)]
    toeplitz.flat[:: AR_ORDER + 1] += 2e-3
    try:
        coefficients = np.linalg.solve(toeplitz, acf[1:])
    except np.linalg.LinAlgError:
        coefficients = np.zeros(AR_ORDER, dtype=np.float64)
    return np.clip(coefficients, -1.5, 1.5)


def _quantile_edges(values: np.ndarray) -> np.ndarray:
    probabilities = np.arange(1, N_BINS, dtype=np.float64) / N_BINS
    return np.quantile(values, probabilities).astype(np.float64, copy=False)


def _rank_location(values: np.ndarray | float, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, values, side="right")
    ranks = (np.asarray(bins, dtype=np.float64) + 0.5) / N_BINS
    return 2.0 * ranks - 1.0


def _rank_standardizer(values: np.ndarray) -> tuple[float, float]:
    center = float(values.mean()) if values.size else 0.0
    scale = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return center, max(scale, EPS)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _even_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=np.intp)
    return np.rint(np.linspace(0, size - 1, maximum)).astype(np.intp)


class _BayesianSuffixCore:
    """Fixed-memory sufficient statistics for six possible break ages."""

    __slots__ = ("ring", "cursor", "count", "sums", "sumsq")

    def __init__(self) -> None:
        self.ring = np.zeros((MAX_DURATION, CHANNEL_COUNT), dtype=np.float64)
        self.cursor = 0
        self.count = 0
        shape = (len(DURATIONS), CHANNEL_COUNT)
        self.sums = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)

    def update(self, channels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current = np.asarray(channels, dtype=np.float64)
        if current.shape != (CHANNEL_COUNT,):
            raise ValueError("Bayesian channel row has the wrong shape")
        square = current * current
        for index, duration in enumerate(DURATIONS):
            self.sums[index] += current
            self.sumsq[index] += square
            if self.count >= duration:
                outgoing = self.ring[(self.cursor - duration) % MAX_DURATION]
                self.sums[index] -= outgoing
                self.sumsq[index] -= outgoing * outgoing
        self.ring[self.cursor] = current
        self.cursor = (self.cursor + 1) % MAX_DURATION
        self.count += 1

        valid_count = int(np.searchsorted(DURATION_ARRAY, self.count, side="right"))
        if valid_count == 0:
            unavailable = np.full(CHANNEL_COUNT, -np.inf, dtype=np.float64)
            return unavailable, unavailable.copy(), np.zeros(CHANNEL_COUNT)

        n = DURATION_ARRAY[:valid_count].astype(np.float64)[:, None]
        total = self.sums[:valid_count]
        total_square = np.maximum(self.sumsq[:valid_count], 0.0)

        mean_log_bf = (
            -0.5 * np.log1p(n * MEAN_PRIOR_VARIANCE)
            + 0.5
            * MEAN_PRIOR_VARIANCE
            * total
            * total
            / (1.0 + n * MEAN_PRIOR_VARIANCE)
        )

        kappa_n = NORMAL_GAMMA_KAPPA + n
        shape_n = NORMAL_GAMMA_SHAPE + 0.5 * n
        rate_n = NORMAL_GAMMA_RATE + 0.5 * (
            total_square - total * total / kappa_n
        )
        rate_n = np.maximum(rate_n, EPS)
        gamma_increment = LOG_GAMMA_INCREMENT[:valid_count, None]
        joint_log_bf = (
            0.5 * np.log(NORMAL_GAMMA_KAPPA / kappa_n)
            + gamma_increment
            + NORMAL_GAMMA_SHAPE * math.log(NORMAL_GAMMA_RATE)
            - shape_n * np.log(rate_n)
            + 0.5 * total_square
        )

        log_prior = LOG_DURATION_PRIOR[:valid_count]
        normalizer = _logsumexp(log_prior)
        mean_mixture = np.empty(CHANNEL_COUNT, dtype=np.float64)
        joint_mixture = np.empty(CHANNEL_COUNT, dtype=np.float64)
        recent_score = np.empty(CHANNEL_COUNT, dtype=np.float64)
        recent_mask = DURATION_ARRAY[:valid_count] <= RECENT_DURATION_MAX
        for channel in range(CHANNEL_COUNT):
            mean_mixture[channel] = (
                _logsumexp(log_prior + mean_log_bf[:, channel]) - normalizer
            )
            joint_terms = log_prior + joint_log_bf[:, channel]
            joint_total = _logsumexp(joint_terms)
            joint_mixture[channel] = joint_total - normalizer
            if recent_mask.any():
                recent_mass = math.exp(
                    _logsumexp(joint_terms[recent_mask]) - joint_total
                )
            else:
                recent_mass = 0.0
            recent_score[channel] = recent_mass * _sigmoid(
                joint_mixture[channel] + LOG_CHANGE_PRIOR_ODDS
            )
        return mean_mixture, joint_mixture, recent_score


class StreamingBayesianChangePoint:
    """History-calibrated, prefix-causal six-feature ADIA state."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
        history = _finite_history(historical)
        cutoff = int(math.floor(REFERENCE_FRACTION * history.size))
        cutoff = max(cutoff, AR_ORDER + 64)
        cutoff = min(cutoff, history.size - 2 * MIN_DURATION)
        reference = history[:cutoff]

        self.raw_center = float(np.median(reference))
        mad = float(np.median(np.abs(reference - self.raw_center)))
        ordinary_scale = float(reference.std(ddof=1))
        self.raw_scale = max(1.4826 * mad, 0.10 * ordinary_scale, EPS)
        robust = np.clip(
            (history - self.raw_center) / self.raw_scale,
            -ROBUST_CLIP,
            ROBUST_CLIP,
        )
        robust_reference = robust[:cutoff]

        self.ar_coefficients = _fit_ar_yule_walker(robust_reference)
        residual = np.empty(history.size - AR_ORDER, dtype=np.float64)
        for index in range(AR_ORDER, history.size):
            lags = robust[index - AR_ORDER : index][::-1]
            residual[index - AR_ORDER] = robust[index] - float(
                self.ar_coefficients @ lags
            )
        reference_residual = residual[: cutoff - AR_ORDER]
        self.residual_center = float(reference_residual.mean())
        residual_mad = float(
            np.median(np.abs(reference_residual - np.median(reference_residual)))
        )
        residual_std = float(reference_residual.std(ddof=1))
        self.residual_scale = max(residual_std, 1.4826 * residual_mad, EPS)
        residual_standardized = (
            residual - self.residual_center
        ) / self.residual_scale

        self.raw_edges = _quantile_edges(robust_reference)
        self.residual_edges = _quantile_edges(
            residual_standardized[: cutoff - AR_ORDER]
        )
        raw_reference_rank = _rank_location(
            robust_reference[AR_ORDER:], self.raw_edges
        )
        residual_reference_rank = _rank_location(
            residual_standardized[: cutoff - AR_ORDER], self.residual_edges
        )
        self.raw_rank_center, self.raw_rank_scale = _rank_standardizer(
            raw_reference_rank
        )
        (
            self.residual_rank_center,
            self.residual_rank_scale,
        ) = _rank_standardizer(residual_reference_rank)

        historical_channels = np.column_stack(
            (
                np.clip(
                    (self._raw_rank(robust[AR_ORDER:]) - self.raw_rank_center)
                    / self.raw_rank_scale,
                    -CHANNEL_CLIP,
                    CHANNEL_CLIP,
                ),
                np.clip(
                    (
                        self._residual_rank(residual_standardized)
                        - self.residual_rank_center
                    )
                    / self.residual_rank_scale,
                    -CHANNEL_CLIP,
                    CHANNEL_CLIP,
                ),
            )
        )
        pseudo_start = cutoff - AR_ORDER
        calibration_rows_by_stage = self._fit_calibrations(
            historical_channels[pseudo_start:]
        )
        self.calibrations = tuple(
            tuple(
                np.sort(calibration_rows[:, index])
                for index in range(FEATURE_COUNT)
            )
            for calibration_rows in calibration_rows_by_stage
        )
        self.calibration_sizes = tuple(
            tuple(int(values.size) for values in stage)
            for stage in self.calibrations
        )

        self.recent_robust = np.array(
            robust[-AR_ORDER:], dtype=np.float64, copy=True
        )
        self.core = _BayesianSuffixCore()

    def _raw_rank(self, value: np.ndarray | float) -> np.ndarray:
        return _rank_location(value, self.raw_edges)

    def _residual_rank(self, value: np.ndarray | float) -> np.ndarray:
        return _rank_location(value, self.residual_edges)

    @staticmethod
    def _raw_summaries(
        mean_mixture: np.ndarray,
        joint_mixture: np.ndarray,
        recent_score: np.ndarray,
    ) -> np.ndarray:
        output = np.empty(FEATURE_COUNT, dtype=np.float64)
        for channel in range(CHANNEL_COUNT):
            offset = channel * SUMMARY_COUNT
            output[offset : offset + SUMMARY_COUNT] = (
                mean_mixture[channel],
                joint_mixture[channel],
                recent_score[channel],
            )
        return output

    def _fit_calibrations(
        self, pseudo_channels: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        core = _BayesianSuffixCore()
        rows_by_stage: list[list[np.ndarray]] = [
            [] for _ in DURATIONS
        ]
        for channels in pseudo_channels:
            mean_mixture, joint_mixture, recent_score = core.update(channels)
            if core.count >= MIN_DURATION:
                stage = int(
                    np.searchsorted(DURATION_ARRAY, core.count, side="right")
                )
                rows_by_stage[stage - 1].append(
                    self._raw_summaries(
                        mean_mixture, joint_mixture, recent_score
                    )
                )
        result: list[np.ndarray] = []
        previous = np.zeros((1, FEATURE_COUNT), dtype=np.float64)
        for rows in rows_by_stage:
            matrix = np.vstack(rows) if rows else previous.copy()
            retained = _even_indices(matrix.shape[0], MAX_CALIBRATION_POINTS)
            matrix = matrix[retained]
            result.append(matrix)
            previous = matrix
        return tuple(result)

    @staticmethod
    def _upper_tail_surprise(reference: np.ndarray, value: float) -> float:
        left = int(np.searchsorted(reference, value, side="left"))
        probability = (reference.size - left + 1.0) / (reference.size + 1.0)
        return min(-math.log(max(probability, EPS)), FEATURE_CLIP)

    def _channels(self, raw_value: float) -> np.ndarray:
        value = float(raw_value)
        if not math.isfinite(value):
            value = self.raw_center
        robust = float(
            np.clip(
                (value - self.raw_center) / self.raw_scale,
                -ROBUST_CLIP,
                ROBUST_CLIP,
            )
        )
        residual = robust - float(
            self.ar_coefficients @ self.recent_robust[::-1]
        )
        residual_standardized = (
            residual - self.residual_center
        ) / self.residual_scale
        channels = np.asarray(
            [
                (float(self._raw_rank(robust)) - self.raw_rank_center)
                / self.raw_rank_scale,
                (
                    float(self._residual_rank(residual_standardized))
                    - self.residual_rank_center
                )
                / self.residual_rank_scale,
            ],
            dtype=np.float64,
        )
        channels = np.clip(channels, -CHANNEL_CLIP, CHANNEL_CLIP)
        self.recent_robust[:-1] = self.recent_robust[1:]
        self.recent_robust[-1] = robust
        return channels

    def update(self, raw_value: float) -> np.ndarray:
        channels = self._channels(raw_value)
        mean_mixture, joint_mixture, recent_score = self.core.update(channels)
        if self.core.count < MIN_DURATION:
            return np.zeros(FEATURE_COUNT, dtype=np.float32)
        raw = self._raw_summaries(mean_mixture, joint_mixture, recent_score)
        stage = int(
            np.searchsorted(DURATION_ARRAY, self.core.count, side="right")
        )
        calibrations = self.calibrations[stage - 1]
        result = np.asarray(
            [
                self._upper_tail_surprise(reference, float(value))
                for reference, value in zip(calibrations, raw)
            ],
            dtype=np.float32,
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
    """Yield the only online implementation path, one current prefix at a time."""

    state = StreamingBayesianChangePoint(historical)
    for value in online:
        yield state.update(value)


def batch_features(
    historical: Sequence[float] | np.ndarray,
    online: Iterable[float],
) -> np.ndarray:
    rows = list(iter_features(historical, online))
    if not rows:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


# Frozen V152 production layer.  The feature implementation above remains the
# single streaming path audited against the full-row V8 cache.
COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
CANDIDATE_ID = "v152_bayesian_ar16_v147"
MANIFEST_FILENAME = "v152_manifest.json"
CALIBRATION_FILENAME = "v152_ecdf_calibration.npz"
FULLFIT_FILENAME = "v152_fullfit_report.json"
AUDIT_FILENAME = "v152_independent_audit.json"
SOURCE_PATHS = ("main.py", "v152_runtime.py")
SELECTED_FEATURE_INDEX = 4
SELECTED_FEATURE_NAME = FEATURE_NAMES[SELECTED_FEATURE_INDEX]
ROLLING_WINDOW = 16
ALPHA = 0.04
PROBABILITY_CLIP = 1e-7
DETERMINISM_TOLERANCE = 1e-8


def _sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _logit_probability(probability: float) -> float:
    clipped = min(
        max(float(probability), PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP
    )
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid_score(score: float) -> float:
    if score >= 0.0:
        value = 1.0 / (1.0 + math.exp(-score))
    else:
        exponential = math.exp(score)
        value = exponential / (1.0 + exponential)
    return min(max(value, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)


@dataclass(frozen=True)
class V152Resources:
    sorted_feature_values: np.ndarray
    time_offsets: np.ndarray
    alpha: float


def validate_resources(model_directory_path: str | Path) -> V152Resources:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "status",
        "competition",
        "candidate_id",
        "calibration_file",
        "calibration_sha256",
        "calibration_size_bytes",
        "fullfit_report_file",
        "fullfit_report_sha256",
        "audit_file",
        "audit_sha256",
        "contract",
        "source_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("V152 manifest inventory changed")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("calibration_file", CALIBRATION_FILENAME),
        ("fullfit_report_file", FULLFIT_FILENAME),
        ("audit_file", AUDIT_FILENAME),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"V152 manifest {key} changed")
    contract = {
        "selected_feature_index": SELECTED_FEATURE_INDEX,
        "selected_feature_name": SELECTED_FEATURE_NAME,
        "rolling_window": ROLLING_WINDOW,
        "alpha": ALPHA,
        "calibration": "same_tick_all_training_series_ecdf",
        "long_series_policy": "saturate_at_last_calibrated_tick",
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_test_series_values_used": False,
        "deterministic": True,
        "determinism_tolerance": DETERMINISM_TOLERANCE,
    }
    if manifest.get("contract") != contract:
        raise ValueError("V152 contract changed")
    source_hashes = manifest.get("source_sha256")
    source_root = Path(__file__).resolve().parent
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(
        SOURCE_PATHS
    ):
        raise ValueError("V152 source inventory changed")
    for name in SOURCE_PATHS:
        digest = source_hashes.get(name)
        if not _is_sha256(digest) or _sha256_file(source_root / name) != digest:
            raise RuntimeError(f"V152 source hash mismatch: {name}")
    calibration_path = root / CALIBRATION_FILENAME
    if (
        not calibration_path.is_file()
        or not _is_sha256(manifest.get("calibration_sha256"))
        or _sha256_file(calibration_path) != manifest["calibration_sha256"]
        or calibration_path.stat().st_size != manifest.get("calibration_size_bytes")
    ):
        raise RuntimeError("V152 calibration resource changed")
    fullfit_path = root / FULLFIT_FILENAME
    if (
        not fullfit_path.is_file()
        or not _is_sha256(manifest.get("fullfit_report_sha256"))
        or _sha256_file(fullfit_path) != manifest["fullfit_report_sha256"]
    ):
        raise RuntimeError("V152 fullfit report changed")
    fullfit = json.loads(fullfit_path.read_text(encoding="utf-8"))
    if (
        fullfit.get("status") != "complete"
        or fullfit.get("candidate_id") != CANDIDATE_ID
        or fullfit.get("fixed_config", {}).get("alpha") != ALPHA
        or fullfit.get("complete_nested", {}).get(
            "promote_to_independent_audit"
        )
        is not True
        or fullfit.get("independent_audit", {}).get("passed") is not True
    ):
        raise RuntimeError("V152 fullfit report is not production eligible")
    audit_path = root / AUDIT_FILENAME
    if (
        not audit_path.is_file()
        or not _is_sha256(manifest.get("audit_sha256"))
        or _sha256_file(audit_path) != manifest["audit_sha256"]
    ):
        raise RuntimeError("V152 audit resource changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("passed") is not True
        or not all(audit.get("checks", {}).values())
        or any(
            float(value) > DETERMINISM_TOLERANCE
            for value in audit.get("maximum_absolute_differences", {}).values()
        )
    ):
        raise RuntimeError("V152 independent audit is not production eligible")
    with np.load(calibration_path) as loaded:
        values = loaded["sorted_feature_values"].astype(np.float32)
        offsets = loaded["time_offsets"].astype(np.int64)
        alpha_values = loaded["alpha"].astype(np.float64)
    if (
        values.ndim != 1
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != values.size
        or np.any(offsets[1:] <= offsets[:-1])
        or not np.isfinite(values).all()
        or alpha_values.shape != (1,)
        or float(alpha_values[0]) != ALPHA
    ):
        raise RuntimeError("V152 calibration arrays are invalid")
    for index in range(offsets.size - 1):
        first = int(offsets[index])
        last = int(offsets[index + 1])
        if np.any(values[first + 1 : last] < values[first : last - 1]):
            raise RuntimeError(f"V152 ECDF reference is not sorted at tick {index}")
    values.setflags(write=False)
    offsets.setflags(write=False)
    return V152Resources(
        sorted_feature_values=values,
        time_offsets=offsets,
        alpha=ALPHA,
    )


class CausalV152State:
    """Same-series Bayesian state plus a frozen training-only ECDF map."""

    def __init__(
        self,
        historical: Sequence[float] | np.ndarray,
        resources: V152Resources,
    ):
        self.bayesian = StreamingBayesianChangePoint(historical)
        self.resources = resources
        self.ring = np.zeros(ROLLING_WINDOW, dtype=np.float64)
        self.running_sum = 0.0
        self.time_online = -1
        self.correction = 0.0

    def update(self, raw_value: float) -> None:
        self.time_online += 1
        feature_row = self.bayesian.update(raw_value)
        # The frozen evaluation cache stores the raw Bayesian feature as
        # float32, applies log1p in float32, and stores the causal mean as
        # float32 before the ECDF lookup.  Preserve those exact rounding
        # boundaries: near-ties are common and even a sub-1e-7 drift can move
        # a large tied block in the rank calibration.
        transformed = float(
            np.log1p(np.float32(feature_row[SELECTED_FEATURE_INDEX]))
        )
        slot = self.time_online % ROLLING_WINDOW
        if self.time_online >= ROLLING_WINDOW:
            self.running_sum -= float(self.ring[slot])
        self.ring[slot] = transformed
        self.running_sum += transformed
        causal_mean = float(
            np.float32(
                self.running_sum
                / min(self.time_online + 1, ROLLING_WINDOW)
            )
        )
        maximum_tick = self.resources.time_offsets.size - 2
        calibration_tick = min(self.time_online, maximum_tick)
        first = int(self.resources.time_offsets[calibration_tick])
        last = int(self.resources.time_offsets[calibration_tick + 1])
        reference = self.resources.sorted_feature_values[first:last]
        left = int(np.searchsorted(reference, causal_mean, side="left"))
        right = int(np.searchsorted(reference, causal_mean, side="right"))
        quantile = (left + right) / (2.0 * reference.size)
        self.correction = 0.5 - quantile

    def apply(self, v147_probability: float) -> float:
        if self.time_online < 0:
            raise RuntimeError("V152 apply called before update")
        score = _logit_probability(v147_probability)
        score += self.resources.alpha * self.correction
        return _sigmoid_score(score)
