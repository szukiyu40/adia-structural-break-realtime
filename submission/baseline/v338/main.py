"""ADIA V338: V335 with a conservative late balanced-detector specialist."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
BASE_ROOT = CODE_ROOT / "base"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from base import v13_heterogeneous_stack_runtime as base_runtime
from postprocess_runtime import CausalPostprocessState, apply_postprocess, validate_resources as validate_postprocess_resources
from v140_runtime import CausalV140State, load_resources as load_v140_resources, validate_resources as validate_v140_resources
from v147_runtime import CausalV147State, validate_resources as validate_v147_resources
from v152_runtime import CausalV152State, validate_resources as validate_v152_resources
from v256_runtime import load_resources as load_v256_resources
from v256_runtime import validate_resources as validate_v256_resources
from v317_runtime import CausalDetectorMemory
from v317_runtime import validate_resources as validate_v317_resources
from v329_runtime import CausalPeriodLagState, load_resources as load_v329_resources, validate_resources as validate_v329_resources
from v335_runtime import load_resources as load_v335_resources, validate_resources as validate_v335_resources
from v338_runtime import EARLY_SCALE, LATE_SCALE, LATE_START, MIDDLE_SCALE, MIDDLE_START, validate_resources as validate_v338_resources


def train(datasets, model_directory_path: str):
    """Validate the immutable all-training-data resources."""

    base_runtime.train_impl(datasets, model_directory_path)
    validate_postprocess_resources(model_directory_path)
    validate_v140_resources(model_directory_path)
    validate_v147_resources(model_directory_path)
    validate_v152_resources(model_directory_path)
    validate_v256_resources(model_directory_path)
    validate_v317_resources(model_directory_path)
    validate_v329_resources(model_directory_path)
    validate_v335_resources(model_directory_path)
    validate_v338_resources(model_directory_path)


def infer(datasets, model_directory_path: str):
    """Yield one deterministic, current/past-only V338 score per tick."""

    base_resources = base_runtime.load_resources(model_directory_path)
    postprocess_resources = validate_postprocess_resources(model_directory_path)
    v140_resources = load_v140_resources(model_directory_path)
    v147_resources = validate_v147_resources(model_directory_path)
    v152_resources = validate_v152_resources(model_directory_path)
    v256_resources = load_v256_resources(model_directory_path)
    v329_resources = load_v329_resources(model_directory_path)
    v335_resources = load_v335_resources(model_directory_path)
    validate_v338_resources(model_directory_path)
    base_runtime.feature_runtime._validate_feature_runtime(base_runtime.FEATURE_VIEWS)  # noqa: SLF001
    yield
    for historical, online in datasets:
        feature_state = base_runtime.SinglePassFeatureState(historical, base_runtime.FEATURE_VIEWS)
        postprocess_state = CausalPostprocessState(historical, postprocess_resources)
        v140_state = CausalV140State(historical, v140_resources)
        v147_state = CausalV147State(historical, v147_resources)
        v152_state = CausalV152State(historical, v152_resources)
        detector_memory = CausalDetectorMemory()
        period_lag_state = CausalPeriodLagState(historical, v329_resources)
        for online_tick, point in enumerate(online):
            value = float(point)
            signals = postprocess_state.update(value)
            v140_state.update(value)
            v147_state.update(value)
            v152_state.update(value)
            rows = feature_state.update(value)
            base_probabilities = base_runtime.predict_base_probabilities(base_resources, rows)
            raw_base = float(base_resources.meta.predict(base_probabilities))
            v35_probability = apply_postprocess(raw_base, signals)
            v140_probability = v140_state.apply(v35_probability)
            v147_probability = v147_state.apply(v35_probability, v140_probability)
            v152_probability = v152_state.apply(v147_probability)

            v4_row = np.asarray(rows["v4_seq18"][: v256_resources.feature_count], dtype=np.float32)
            if (
                v4_row.shape != (v256_resources.feature_count,)
                or v335_resources.feature_count != v256_resources.feature_count
                or not np.isfinite(v4_row).all()
            ):
                raise RuntimeError("V338 V4 feature row changed")
            row = v4_row.reshape(1, -1)
            old_predictions = []
            for booster in v256_resources.boosters:
                prediction = np.asarray(booster.predict(row, raw_score=False, num_threads=1), dtype=np.float64)
                if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
                    raise RuntimeError("V338 V256 prediction is invalid")
                old_predictions.append(float(prediction[0]))
            detector_probability = float(np.mean(old_predictions))
            new_predictions = []
            for booster in v335_resources.boosters:
                prediction = np.asarray(booster.predict(row, raw_score=False, num_threads=1), dtype=np.float64)
                if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
                    raise RuntimeError("V338 weighted detector prediction is invalid")
                new_predictions.append(float(prediction[0]))
            if len(new_predictions) != 2:
                raise RuntimeError("V338 weighted detector inventory changed")
            v317_probability = detector_memory.update(v152_probability, detector_probability)
            if online_tick < MIDDLE_START:
                detector_delta = 0.5 * (new_predictions[0] + new_predictions[1]) - detector_probability
                scale = EARLY_SCALE
            elif online_tick < LATE_START:
                detector_delta = new_predictions[1] - detector_probability
                scale = MIDDLE_SCALE
            else:
                detector_delta = new_predictions[1] - detector_probability
                scale = LATE_SCALE
            adjusted_probability = float(np.clip(
                v317_probability + scale * detector_delta,
                1e-7,
                1.0 - 1e-7,
            ))
            yield period_lag_state.update(value, adjusted_probability)


# @crunch/keep:on
INFER_PARALLELISM = 4
