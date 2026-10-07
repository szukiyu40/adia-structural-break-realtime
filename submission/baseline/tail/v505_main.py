"""ADIA V505 fast production transport of locally audited V504."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
V338_ROOT = CODE_ROOT / "v338"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(V338_ROOT) not in sys.path:
    sys.path.insert(0, str(V338_ROOT))

from v338 import main as v338_main


def train(datasets, model_directory_path: str):
    """No-op: immutable resources are authenticated inside inference."""

    del datasets, model_directory_path


def infer(datasets, model_directory_path: str):
    """Yield deterministic V504 probabilities one current tick at a time."""

    from v377_runtime import load_resources as load_v377_resources
    from v399_runtime import V399State, load_resources as load_v399_resources
    from v441_prequential_pit_geometry import PrequentialPITGeometry
    from v496_correction_runtime import (
        EPSILON,
        PIT_FEATURE_INDEX,
        SymmetricMemoryState,
        _logit,
        _sigmoid,
        fisher_energy,
        load_resources,
    )
    from v505_plateau_runtime import PlateauState, validate_resources

    base_resources = v338_main.base_runtime.load_resources(model_directory_path)
    postprocess_resources = v338_main.validate_postprocess_resources(model_directory_path)
    v140_resources = v338_main.load_v140_resources(model_directory_path)
    v147_resources = v338_main.validate_v147_resources(model_directory_path)
    v152_resources = v338_main.validate_v152_resources(model_directory_path)
    v256_resources = v338_main.load_v256_resources(model_directory_path)
    v329_resources = v338_main.load_v329_resources(model_directory_path)
    v335_resources = v338_main.load_v335_resources(model_directory_path)
    v338_main.validate_v338_resources(model_directory_path)
    v377_resources = load_v377_resources(model_directory_path)
    v399_resources = load_v399_resources(model_directory_path)
    correction_resources = load_resources(model_directory_path)
    validate_resources(model_directory_path)
    v338_main.base_runtime.feature_runtime._validate_feature_runtime(  # noqa: SLF001
        v338_main.base_runtime.FEATURE_VIEWS
    )
    yield
    for historical, online in datasets:
        feature_state = v338_main.base_runtime.SinglePassFeatureState(
            historical, v338_main.base_runtime.FEATURE_VIEWS
        )
        postprocess_state = v338_main.CausalPostprocessState(
            historical, postprocess_resources
        )
        v140_state = v338_main.CausalV140State(historical, v140_resources)
        v147_state = v338_main.CausalV147State(historical, v147_resources)
        v152_state = v338_main.CausalV152State(historical, v152_resources)
        detector_memory = v338_main.CausalDetectorMemory()
        period_lag_state = v338_main.CausalPeriodLagState(historical, v329_resources)
        v399_state = V399State(historical, v399_resources, v377_resources)
        pit_state = PrequentialPITGeometry(historical)
        memory_state = SymmetricMemoryState()
        plateau_state = PlateauState()
        for online_tick, point in enumerate(online):
            value = float(point)
            pit_row = pit_state.update(value)
            signals = postprocess_state.update(value)
            v140_state.update(value)
            v147_state.update(value)
            v152_state.update(value)
            rows = feature_state.update(value)
            base_probabilities = v338_main.base_runtime.predict_base_probabilities(
                base_resources, rows
            )
            raw_base = float(base_resources.meta.predict(base_probabilities))
            v35_probability = v338_main.apply_postprocess(raw_base, signals)
            v140_probability = v140_state.apply(v35_probability)
            v147_probability = v147_state.apply(v35_probability, v140_probability)
            v152_probability = v152_state.apply(v147_probability)
            full_v4_row = np.asarray(rows["v4_seq18"], dtype=np.float32)
            v4_row = full_v4_row[: v256_resources.feature_count]
            if (
                full_v4_row.shape != (440,)
                or v4_row.shape != (422,)
                or not np.isfinite(full_v4_row).all()
            ):
                raise RuntimeError("V505 V4 feature row changed")
            row = v4_row.reshape(1, -1)
            old_predictions = []
            for booster in v256_resources.boosters:
                prediction = np.asarray(
                    booster.predict(row, raw_score=False, num_threads=1),
                    dtype=np.float64,
                )
                if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
                    raise RuntimeError("V505 V256 prediction failed")
                old_predictions.append(float(prediction[0]))
            detector_probability = float(np.mean(old_predictions))
            new_predictions = []
            for booster in v335_resources.boosters:
                prediction = np.asarray(
                    booster.predict(row, raw_score=False, num_threads=1),
                    dtype=np.float64,
                )
                if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
                    raise RuntimeError("V505 V335 prediction failed")
                new_predictions.append(float(prediction[0]))
            v317_probability = detector_memory.update(v152_probability, detector_probability)
            if online_tick < v338_main.MIDDLE_START:
                detector_delta = 0.5 * (new_predictions[0] + new_predictions[1]) - detector_probability
                scale = v338_main.EARLY_SCALE
            elif online_tick < v338_main.LATE_START:
                detector_delta = new_predictions[1] - detector_probability
                scale = v338_main.MIDDLE_SCALE
            else:
                detector_delta = new_predictions[1] - detector_probability
                scale = v338_main.LATE_SCALE
            adjusted = float(
                np.clip(
                    v317_probability + scale * detector_delta,
                    EPSILON,
                    1.0 - EPSILON,
                )
            )
            v338_probability = period_lag_state.update(value, adjusted)
            v399_probability = v399_state.update(value, v4_row, v338_probability)
            pre_memory_margin = (
                _logit(v399_probability)
                + correction_resources.pit.increment(
                    online_tick, float(pit_row[PIT_FEATURE_INDEX])
                )
                + correction_resources.fisher.increment(
                    online_tick, fisher_energy(pit_row)
                )
            )
            post_memory_margin = memory_state.update(pre_memory_margin, online_tick)
            margin = plateau_state.update(
                pre_memory_margin, post_memory_margin, online_tick
            )
            result = _sigmoid(margin)
            if not math.isfinite(result):
                raise RuntimeError("V505 final prediction is non-finite")
            yield min(max(result, EPSILON), 1.0 - EPSILON)


# @crunch/keep:on
INFER_PARALLELISM = 1

