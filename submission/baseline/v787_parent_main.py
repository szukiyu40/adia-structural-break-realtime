"""ADIA V787: production-faithful frozen V780 with one full-fit resource each."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
V338_ROOT = CODE_ROOT / "v338"
TAIL_ROOT = CODE_ROOT / "tail"
for _path in (TAIL_ROOT, V338_ROOT, CODE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from v338 import main as v338_main  # noqa: E402
from tail.v441_prequential_pit_geometry import PrequentialPITGeometry  # noqa: E402
from tail.v496_correction_runtime import (  # noqa: E402
    PIT_FEATURE_INDEX,
    SymmetricMemoryState,
    fisher_energy,
    load_resources as load_correction_resources,
)
from tail.v505_plateau_runtime import (  # noqa: E402
    PlateauState,
    validate_resources as validate_plateau_resources,
)
from v784_v780_component_runtime import (  # noqa: E402
    LeanCurvatureState,
    LeanPositionalState,
    NLAREProcessState,
    ScorePathState,
)
from v780_runtime import (  # noqa: E402
    EPSILON,
    OPEN_SCALE,
    OCCUPATION_SCALE,
    POSITIONAL_SHARE,
    POSITIONAL_THRESHOLD,
    V643_BOUNDARY,
    V643_SCALE,
    booster_probability,
    curvature_correction,
    eprocess_increment,
    load_resources,
    logit,
    near_iid_responsibility,
    pairrank_correction,
    sigmoid,
    validate_resources,
)


def train(datasets, model_directory_path: str):
    """Authenticate immutable base, tail, and V780 full-fit resources."""

    v338_main.train(datasets, model_directory_path)
    load_correction_resources(model_directory_path)
    validate_plateau_resources(model_directory_path)
    validate_resources(model_directory_path)


def _scorepath_margin(
    tail_probability: float,
    tick: int,
    state: ScorePathState,
    eprocess: float,
) -> float:
    open_energy, occupation = state.update(tail_probability, tick)
    margin = logit(tail_probability)
    if tick >= 128:
        margin += OPEN_SCALE * (2.0 * open_energy - 1.0)
    uncertainty = 4.0 * tail_probability * (1.0 - tail_probability)
    margin += OCCUPATION_SCALE * uncertainty * (2.0 * occupation - 1.0)
    return margin + eprocess


def _tail_probability(
    anchor: float,
    curvature: float,
    pit_increment: float,
    fisher_increment: float,
    tick: int,
    memory: SymmetricMemoryState,
    plateau: PlateauState,
) -> float:
    pre_memory = logit(anchor) + curvature + pit_increment + fisher_increment
    post_memory = memory.update(pre_memory, tick)
    return sigmoid(plateau.update(pre_memory, post_memory, tick))


def infer(datasets, model_directory_path: str):
    """Yield one deterministic current/past-only V780 score per online tick."""

    base_resources = v338_main.base_runtime.load_resources(model_directory_path)
    postprocess_resources = v338_main.validate_postprocess_resources(model_directory_path)
    v140_resources = v338_main.load_v140_resources(model_directory_path)
    v147_resources = v338_main.validate_v147_resources(model_directory_path)
    v152_resources = v338_main.validate_v152_resources(model_directory_path)
    v256_resources = v338_main.load_v256_resources(model_directory_path)
    v329_resources = v338_main.load_v329_resources(model_directory_path)
    v335_resources = v338_main.load_v335_resources(model_directory_path)
    v338_main.validate_v338_resources(model_directory_path)
    correction_resources = load_correction_resources(model_directory_path)
    resources = load_resources(model_directory_path)
    v338_main.base_runtime.feature_runtime._validate_feature_runtime(  # noqa: SLF001
        v338_main.base_runtime.FEATURE_VIEWS
    )
    yield

    for historical, online in datasets:
        history = np.asarray(historical, dtype=np.float64)
        responsibility = near_iid_responsibility(history, resources)
        confidence = float(
            np.clip(
                (2.0 * responsibility - 1.0)
                / max(responsibility, np.finfo(np.float64).tiny),
                0.0,
                1.0,
            )
        )
        positional_active = responsibility >= POSITIONAL_THRESHOLD

        feature_state = v338_main.base_runtime.SinglePassFeatureState(
            history, v338_main.base_runtime.FEATURE_VIEWS
        )
        postprocess_state = v338_main.CausalPostprocessState(
            history, postprocess_resources
        )
        v140_state = v338_main.CausalV140State(history, v140_resources)
        v147_state = v338_main.CausalV147State(history, v147_resources)
        v152_state = v338_main.CausalV152State(history, v152_resources)
        detector_memory = v338_main.CausalDetectorMemory()
        period_lag_state = v338_main.CausalPeriodLagState(history, v329_resources)

        positional_state = LeanPositionalState(history)
        curvature_state = LeanCurvatureState(history)
        nlar_state = NLAREProcessState(history)
        pit_state = PrequentialPITGeometry(history)
        free_memory = SymmetricMemoryState()
        free_plateau = PlateauState()
        free_scorepath = ScorePathState()
        positional_memory = SymmetricMemoryState() if positional_active else None
        positional_plateau = PlateauState() if positional_active else None
        positional_scorepath = ScorePathState() if positional_active else None

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

            v4_row = np.asarray(
                rows["v4_seq18"][: v256_resources.feature_count], dtype=np.float32
            )
            if (
                v4_row.shape != (422,)
                or v335_resources.feature_count != v256_resources.feature_count
                or not np.isfinite(v4_row).all()
            ):
                raise RuntimeError("V787 V4 feature row changed")
            row = v4_row.reshape(1, -1)
            old_predictions = []
            for booster in v256_resources.boosters:
                prediction = np.asarray(
                    booster.predict(row, raw_score=False, num_threads=1),
                    dtype=np.float64,
                )
                if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
                    raise RuntimeError("V787 V256 prediction failed")
                old_predictions.append(float(prediction[0]))
            detector_probability = float(np.mean(old_predictions))

            new_predictions = []
            for booster in v335_resources.boosters:
                prediction = np.asarray(
                    booster.predict(row, raw_score=False, num_threads=1),
                    dtype=np.float64,
                )
                if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
                    raise RuntimeError("V787 V335 prediction failed")
                new_predictions.append(float(prediction[0]))
            if len(new_predictions) != 2:
                raise RuntimeError("V787 V335 inventory changed")

            v317_probability = detector_memory.update(
                v152_probability, detector_probability
            )
            if online_tick < v338_main.MIDDLE_START:
                detector_delta = 0.5 * (new_predictions[0] + new_predictions[1]) - detector_probability
                detector_scale = v338_main.EARLY_SCALE
            elif online_tick < v338_main.LATE_START:
                detector_delta = new_predictions[1] - detector_probability
                detector_scale = v338_main.MIDDLE_SCALE
            else:
                detector_delta = new_predictions[1] - detector_probability
                detector_scale = v338_main.LATE_SCALE
            adjusted = float(
                np.clip(
                    v317_probability + detector_scale * detector_delta,
                    EPSILON,
                    1.0 - EPSILON,
                )
            )
            compact_anchor = period_lag_state.update(value, adjusted)

            positional_row = positional_state.update(value)
            positional_probability = booster_probability(
                resources.positional, positional_row
            )
            v360_anchor = float(
                np.clip(
                    compact_anchor
                    + POSITIONAL_SHARE
                    * responsibility
                    * (positional_probability - compact_anchor),
                    EPSILON,
                    1.0 - EPSILON,
                )
            )
            curvature_row = curvature_state.update(value)
            curvature = curvature_correction(
                resources, curvature_row, v360_anchor, online_tick
            )
            pit_increment = correction_resources.pit.increment(
                online_tick, float(pit_row[PIT_FEATURE_INDEX])
            )
            fisher_increment = correction_resources.fisher.increment(
                online_tick, fisher_energy(pit_row)
            )
            raw_eprocess = nlar_state.update(value)
            eprocess = eprocess_increment(resources, raw_eprocess, online_tick)

            free_tail = _tail_probability(
                compact_anchor,
                curvature,
                pit_increment,
                fisher_increment,
                online_tick,
                free_memory,
                free_plateau,
            )
            free_margin = _scorepath_margin(
                free_tail, online_tick, free_scorepath, eprocess
            )
            if positional_active:
                if positional_memory is None or positional_plateau is None or positional_scorepath is None:
                    raise RuntimeError("V787 positional path state missing")
                positional_tail = _tail_probability(
                    v360_anchor,
                    curvature,
                    pit_increment,
                    fisher_increment,
                    online_tick,
                    positional_memory,
                    positional_plateau,
                )
                positional_margin = _scorepath_margin(
                    positional_tail,
                    online_tick,
                    positional_scorepath,
                    eprocess,
                )
                final_margin = free_margin + confidence * (
                    positional_margin - free_margin
                )
            else:
                final_margin = free_margin

            if online_tick >= V643_BOUNDARY:
                full_row = np.asarray(rows["v4_seq18"], dtype=np.float32)
                final_margin += V643_SCALE * pairrank_correction(
                    resources, full_row
                )
            result = sigmoid(final_margin)
            if not math.isfinite(result):
                raise RuntimeError("V787 final prediction is non-finite")
            yield min(max(result, EPSILON), 1.0 - EPSILON)


# @crunch/keep:on
INFER_PARALLELISM = 4
