"""Hybrid ADIA candidate: the complete v2 bank plus calibrated v3 signals.

The v2 detector is the locally strongest candidate.  This module preserves
all 229 of its features and adds only the complementary v3 measurements:

* the historical null-block count for each window;
* calibrated raw-channel statistics except the duplicate mean statistic;
* all calibrated AR(10) statistics; and
* the final analytic top-eight empirical probability.

Both underlying streaming states see the same historical sequence and every
online point in the same causal order.  Training and inference share the
``iter_features`` path.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np

from experiments import v2_candidate as v2
from experiments import v3_candidate as v3


WINDOWS = v2.WINDOWS
if WINDOWS != v3.WINDOWS:
    raise RuntimeError("v2 and v3 window schemas no longer agree")

MODEL_FILENAME = "model.txt"
EPS = 1e-8
SAMPLE_MODULUS = 5
SAMPLE_PROBABILITY = 1.0 / SAMPLE_MODULUS
ALLOWED_POSITIVE_RAMPS = (0, 8, 16)
DEFAULT_POSITIVE_RAMP = 8
DEFAULT_FEATURE_FRACTION = 0.78
DEFAULT_SEED = 20260804

V2_FEATURE_COUNT = 229
V3_RAW_SUPPLEMENT_STAT_NAMES = tuple(v3.ALL_STAT_NAMES[1:])
V3_AR10_SUPPLEMENT_STAT_NAMES = tuple(v3.ALL_STAT_NAMES)
V3_PER_WINDOW_SUPPLEMENT_COUNT = (
    1
    + len(V3_RAW_SUPPLEMENT_STAT_NAMES)
    + len(V3_AR10_SUPPLEMENT_STAT_NAMES)
)
V3_SUPPLEMENT_FEATURE_COUNT = (
    len(WINDOWS) * V3_PER_WINDOW_SUPPLEMENT_COUNT + 1
)

# This literal is intentional: an accidental change in either source schema
# must fail loudly instead of silently changing the model input width.
EXPECTED_FEATURE_COUNT = 422
FEATURE_COUNT = V2_FEATURE_COUNT + V3_SUPPLEMENT_FEATURE_COUNT
if FEATURE_COUNT != EXPECTED_FEATURE_COUNT:
    raise RuntimeError(
        f"v4 feature schema changed: {FEATURE_COUNT} != {EXPECTED_FEATURE_COUNT}"
    )


_V2_META_NAMES = (
    "online_log1p_n",
    "online_sqrt_n",
    "historical_log1p_length",
    "historical_skew",
    "historical_excess_kurtosis",
    "historical_acf_1",
    "historical_acf_2",
    "historical_acf_5",
    "historical_abs_value_acf_1",
    "current_abs_z",
    "current_z_squared",
    "current_abs_ar5_residual",
    "current_ar5_residual_squared",
)
_V2_WINDOW_NAMES = (
    "mean_stat",
    "abs_mean_stat",
    "variance_stat",
    "abs_variance_stat",
    "skew_stat",
    "abs_skew_stat",
    "kurtosis_stat",
    "abs_kurtosis_stat",
    "pit_chi_stat",
    "abs_pit_chi_stat",
    "pit_ks_stat",
    "pit_edge_stat",
    "abs_pit_edge_stat",
    "acf_1_stat",
    "abs_acf_1_stat",
    "acf_2_stat",
    "abs_acf_2_stat",
    "acf_5_stat",
    "abs_acf_5_stat",
    "abs_value_acf_1_stat",
    "abs_abs_value_acf_1_stat",
    "ar5_residual_mean_stat",
    "abs_ar5_residual_mean_stat",
    "ar5_residual_variance_stat",
    "abs_ar5_residual_variance_stat",
    "ar5_residual_pit_chi_stat",
    "abs_ar5_residual_pit_chi_stat",
    "ar5_residual_pit_ks_stat",
    "ar5_residual_pit_edge_stat",
    "abs_ar5_residual_pit_edge_stat",
    "ar5_residual_acf_1_stat",
    "abs_ar5_residual_acf_1_stat",
)
_V2_FAMILY_NAMES = (
    "mean",
    "variance",
    "pit",
    "shape",
    "acf",
    "abs_value_acf",
    "ar5_residual_distribution",
    "ar5_residual_acf",
)


def _build_v2_feature_names() -> tuple[str, ...]:
    names = [f"v2_{name}" for name in _V2_META_NAMES]
    for window in WINDOWS:
        names.extend(f"v2_w{window}_{name}" for name in _V2_WINDOW_NAMES)
    for summary in ("current", "decayed_max", "running_max"):
        names.extend(f"v2_{summary}_{name}" for name in _V2_FAMILY_NAMES)
    result = tuple(names)
    if len(result) != V2_FEATURE_COUNT:
        raise RuntimeError(
            f"v2 feature-name schema changed: {len(result)} != {V2_FEATURE_COUNT}"
        )
    return result


def _build_v3_supplement_schema() -> tuple[tuple[str, ...], tuple[int, ...]]:
    names: list[str] = []
    source_indices: list[int] = []
    raw_count = len(v3.ALL_STAT_NAMES)
    residual5_count = len(v3.RESIDUAL5_STAT_INDICES)
    for window_index, window in enumerate(WINDOWS):
        block_start = (
            v3.META_FEATURE_COUNT + window_index * v3.PER_WINDOW_FEATURE_COUNT
        )
        names.append(f"v3_w{window}_log_null_blocks")
        source_indices.append(block_start)

        # The raw mean is omitted.  v2 already contains signed and absolute
        # mean statistics at every matching window.
        for stat_index, stat_name in enumerate(v3.ALL_STAT_NAMES[1:], start=1):
            names.append(f"v3_w{window}_raw_calibrated_{stat_name}")
            source_indices.append(block_start + 1 + stat_index)

        ar10_start = block_start + 1 + raw_count + residual5_count
        for stat_index, stat_name in enumerate(v3.ALL_STAT_NAMES):
            names.append(f"v3_w{window}_ar10_calibrated_{stat_name}")
            source_indices.append(ar10_start + stat_index)

    names.append("v3_analytic_top8")
    source_indices.append(v3.FEATURE_COUNT - 1)
    result_names = tuple(names)
    result_indices = tuple(source_indices)
    if len(result_names) != V3_SUPPLEMENT_FEATURE_COUNT:
        raise RuntimeError("v3 supplemental name schema has the wrong width")
    if len(result_indices) != V3_SUPPLEMENT_FEATURE_COUNT:
        raise RuntimeError("v3 supplemental index schema has the wrong width")
    return result_names, result_indices


V2_FEATURE_NAMES = _build_v2_feature_names()
V3_SUPPLEMENT_FEATURE_NAMES, V3_SUPPLEMENT_SOURCE_INDICES = (
    _build_v3_supplement_schema()
)
V3_SUPPLEMENT_SOURCE_INDEX_ARRAY = np.asarray(
    V3_SUPPLEMENT_SOURCE_INDICES, dtype=np.intp
)
FEATURE_NAMES = V2_FEATURE_NAMES + V3_SUPPLEMENT_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
if len(FEATURE_NAMES) != FEATURE_COUNT or len(FEATURE_INDEX) != FEATURE_COUNT:
    raise RuntimeError("v4 feature names must be complete and unique")


class StreamingFeaturesV4:
    """Causal dual-state hybrid for one ADIA series."""

    def __init__(self, historical: Sequence[float] | np.ndarray):
        historical_array = np.array(historical, dtype=np.float64, copy=True)
        # Pass the same immutable sequence to both states.  The source states
        # make their own derived arrays and do not mutate this input.
        historical_array.setflags(write=False)
        self.v2_state = v2.StreamingFeaturesV2(historical_array)
        self.v3_state = v3.StreamingFeaturesV3(historical_array)

    def update(self, raw_value: float) -> np.ndarray:
        v2_row = self.v2_state.update(raw_value)
        v3_row = self.v3_state.update(raw_value)
        if v2_row.size != V2_FEATURE_COUNT:
            raise RuntimeError(
                f"v2 feature width changed: {v2_row.size} != {V2_FEATURE_COUNT}"
            )
        if v3_row.size != v3.FEATURE_COUNT:
            raise RuntimeError(
                f"v3 feature width changed: {v3_row.size} != {v3.FEATURE_COUNT}"
            )

        result = np.empty(FEATURE_COUNT, dtype=np.float32)
        result[:V2_FEATURE_COUNT] = v2_row
        np.take(
            v3_row,
            V3_SUPPLEMENT_SOURCE_INDEX_ARRAY,
            out=result[V2_FEATURE_COUNT:],
        )
        return result


def iter_features(
    historical: Sequence[float] | np.ndarray,
    online: Iterable[float],
) -> Iterator[np.ndarray]:
    """Single causal feature path shared by training and inference."""

    state = StreamingFeaturesV4(historical)
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
    """Assign one of five time residues deterministically to each series."""

    return int((-17 * int(dataset_id)) % SAMPLE_MODULUS)


def _keep_training_row(dataset_id: int, time_online: int) -> bool:
    """Uniform deterministic 20% sample over every online time region."""

    return int(time_online) % SAMPLE_MODULUS == _sample_residue(dataset_id)


def _sampled_count(dataset_id: int, length: int) -> int:
    first = _sample_residue(dataset_id)
    if first >= length:
        return 0
    return 1 + (length - 1 - first) // SAMPLE_MODULUS


def _validate_positive_ramp(positive_ramp: int) -> int:
    value = int(positive_ramp)
    if value not in ALLOWED_POSITIVE_RAMPS:
        raise ValueError(
            f"positive_ramp must be one of {ALLOWED_POSITIVE_RAMPS}, got {value}"
        )
    return value


def build_training_matrix(datasets, positive_ramp: int = DEFAULT_POSITIVE_RAMP):
    """Preallocate the sampled matrix and apply exact metric pair weights."""

    ramp = _validate_positive_ramp(positive_ramp)
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
            if label and ramp > 0:
                age = time_online - int(tau) + 1
                weight *= min(1.0, age / ramp)

            matrix[row_index] = feature_row
            target[row_index] = label
            sample_weight[row_index] = weight
            row_index += 1

    if row_index == 0:
        raise ValueError("No trainable rows remain after deterministic sampling")
    matrix = matrix[:row_index]
    target = target[:row_index]
    sample_weight = sample_weight[:row_index]
    sample_weight /= max(float(sample_weight.mean()), EPS)
    return matrix, target, sample_weight


def fit_booster(
    matrix: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    feature_fraction: float = DEFAULT_FEATURE_FRACTION,
    seed: int = DEFAULT_SEED,
) -> lgb.LGBMClassifier:
    """Fit the deterministic hybrid LightGBM with configurable column seed."""

    fraction = float(feature_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("feature_fraction must be in (0, 1]")
    random_seed = int(seed)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=10,
        min_child_samples=400,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=fraction,
        reg_alpha=0.20,
        reg_lambda=5.0,
        random_state=random_seed,
        bagging_seed=random_seed,
        feature_fraction_seed=random_seed,
        data_random_seed=random_seed,
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
    positive_ramp: int = DEFAULT_POSITIVE_RAMP,
    feature_fraction: float = DEFAULT_FEATURE_FRACTION,
    seed: int = DEFAULT_SEED,
):
    os.makedirs(model_directory_path, exist_ok=True)
    matrix, target, sample_weight = build_training_matrix(
        datasets, positive_ramp=positive_ramp
    )
    model = fit_booster(
        matrix,
        target,
        sample_weight,
        feature_fraction=feature_fraction,
        seed=seed,
    )
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
