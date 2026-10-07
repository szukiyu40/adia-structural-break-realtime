"""Authenticated lightweight runtime resources for the frozen V780 formula."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tail.history_process_features_runtime import extract_history_process_features


EPSILON = 1e-7
NEAR_IID_CODE = 7
POSITIONAL_SHARE = 0.08
POSITIONAL_THRESHOLD = 0.5
CURVATURE_START = 128
V646_BOUNDARY = 128
V643_BOUNDARY = 256
V643_SCALE = 0.10
OPEN_SCALE = 0.08
OCCUPATION_SCALE = 0.04
CANDIDATE_ID = "v787_v780_submission_candidate"
MANIFEST_NAME = "v787_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def logit(value: float) -> float:
    probability = float(np.clip(value, EPSILON, 1.0 - EPSILON))
    return math.log(probability) - math.log1p(-probability)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponent = math.exp(value)
    return float(exponent / (1.0 + exponent))


@dataclass(frozen=True)
class Resources:
    positional: Any
    pairrank: Any
    process_lower: np.ndarray
    process_upper: np.ndarray
    process_center: np.ndarray
    process_scale: np.ndarray
    process_centroids: np.ndarray
    semantic_code_by_raw: np.ndarray
    process_temperature: float
    curvature_beta: np.ndarray
    curvature_center: np.ndarray
    curvature_scale: np.ndarray
    eprocess_values: np.ndarray
    eprocess_offsets: np.ndarray
    eprocess_boundary: int
    eprocess_scale: float


def _validated_payload(model_directory_path: str | Path) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    before = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if before != _sha256(manifest_path):
        raise RuntimeError("V787 manifest changed while reading")
    contract = manifest.get("contract", {})
    if (
        manifest.get("status") != "complete_pending_runtime_gates"
        or manifest.get("candidate_id") != CANDIDATE_ID
        or contract.get("formula") != "frozen_v780"
        or contract.get("future_ticks_used") is not False
        or contract.get("peer_test_series_used") is not False
        or contract.get("final_online_length_used") is not False
        or contract.get("deterministic") is not True
    ):
        raise RuntimeError("V787 manifest contract changed")
    code_root = Path(__file__).resolve().parent
    for relative, digest in manifest.get("source_files", {}).items():
        source = (code_root / relative).resolve()
        if code_root not in source.parents or not source.is_file() or _sha256(source) != digest:
            raise RuntimeError(f"V787 source changed: {relative}")
    for filename, digest in manifest.get("resource_files", {}).items():
        path = root / filename
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"V787 resource changed: {filename}")
    audit = json.loads((root / "v786_v785_fullfit_resources_independent_audit.json").read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or audit.get("decision", {}).get("advance_to_submission_candidate_assembly") is not True
        or audit.get("contract", {}).get("v785_builder_imported") is not False
    ):
        raise RuntimeError("V787 independent resource audit changed")
    v785 = json.loads((root / "v785_fullfit_resources_manifest.json").read_text(encoding="utf-8"))
    if v785.get("status") != "complete_pending_independent_v786_audit":
        raise RuntimeError("V787 V785 manifest changed")

    with np.load(root / "v377_parameters.npz", allow_pickle=False) as archive:
        process = {name: archive[name].copy() for name in archive.files}
    expected_process = {
        "process_lower": (67,), "process_upper": (67,), "process_center": (67,),
        "process_scale": (67,), "process_centroids": (8, 67),
        "semantic_code_by_raw": (8,), "process_temperature": (1,),
    }
    for name, shape in expected_process.items():
        if name not in process or process[name].shape != shape or not np.isfinite(process[name]).all():
            raise RuntimeError(f"V787 process schema changed: {name}")

    with np.load(root / "v748_curvature_fullfit.npz", allow_pickle=False) as archive:
        curvature = {name: archive[name].copy() for name in archive.files}
    if (
        curvature.get("beta", np.empty(0)).shape != (128,)
        or curvature.get("center", np.empty(0)).shape != (32,)
        or curvature.get("scale", np.empty(0)).shape != (32,)
        or not np.array_equal(
            curvature.get("selected_feature_indices", np.empty(0, dtype=np.int64)),
            np.asarray((*range(8, 16), *range(56, 80)), dtype=np.int64),
        )
        or int(curvature.get("start_tick", np.asarray([-1]))[0]) != CURVATURE_START
    ):
        raise RuntimeError("V787 curvature resource schema changed")

    with np.load(root / "v646_exact_time_ecdf.npz", allow_pickle=False) as archive:
        eprocess = {name: archive[name].copy() for name in archive.files}
    values = eprocess.get("sorted_values", np.empty(0))
    offsets = eprocess.get("time_offsets", np.empty(0, dtype=np.int64)).astype(np.int64)
    if (
        values.dtype != np.float32
        or values.ndim != 1
        or offsets.ndim != 1
        or offsets.size < 2
        or int(offsets[-1]) != values.size
        or np.any(np.diff(offsets[V646_BOUNDARY:]) <= 0)
        or int(eprocess.get("boundary", np.asarray([-1]))[0]) != V646_BOUNDARY
        or float(eprocess.get("scale", np.asarray([math.nan]))[0]) != 0.04
    ):
        raise RuntimeError("V787 V646 resource schema changed")
    return root, process, {**curvature, **{f"eprocess_{key}": value for key, value in eprocess.items()}}


def validate_resources(model_directory_path: str | Path) -> None:
    _validated_payload(model_directory_path)


def load_resources(model_directory_path: str | Path) -> Resources:
    root, process, arrays = _validated_payload(model_directory_path)
    import lightgbm as lgb

    positional = lgb.Booster(model_file=str(root / "v377_positional_model.txt"))
    pairrank = lgb.Booster(model_file=str(root / "v643_mature_pairrank_model.txt"))
    if positional.num_feature() != 94 or positional.num_trees() != 500:
        raise RuntimeError("V787 positional model schema changed")
    if pairrank.num_feature() != 440 or pairrank.num_trees() != 180:
        raise RuntimeError("V787 PairRank model schema changed")
    return Resources(
        positional=positional,
        pairrank=pairrank,
        process_lower=process["process_lower"].astype(np.float64),
        process_upper=process["process_upper"].astype(np.float64),
        process_center=process["process_center"].astype(np.float64),
        process_scale=process["process_scale"].astype(np.float64),
        process_centroids=process["process_centroids"].astype(np.float64),
        semantic_code_by_raw=process["semantic_code_by_raw"].astype(np.int8),
        process_temperature=float(process["process_temperature"][0]),
        curvature_beta=arrays["beta"].astype(np.float64),
        curvature_center=arrays["center"].astype(np.float64),
        curvature_scale=arrays["scale"].astype(np.float64),
        eprocess_values=arrays["eprocess_sorted_values"].astype(np.float32, copy=False),
        eprocess_offsets=arrays["eprocess_time_offsets"].astype(np.int64),
        eprocess_boundary=int(arrays["eprocess_boundary"][0]),
        eprocess_scale=float(arrays["eprocess_scale"][0]),
    )


def near_iid_responsibility(
    historical: Sequence[float] | np.ndarray, resources: Resources
) -> float:
    feature = extract_history_process_features(np.asarray(historical, dtype=np.float64))
    clipped = np.clip(feature, resources.process_lower, resources.process_upper)
    standardized = (clipped - resources.process_center) / resources.process_scale
    squared = np.sum((standardized[None, :] - resources.process_centroids) ** 2, axis=1)
    logits = -(squared - float(np.min(squared))) / resources.process_temperature
    logits -= float(np.max(logits))
    raw = np.exp(logits)
    raw /= float(np.sum(raw))
    semantic = np.zeros(8, dtype=np.float64)
    for raw_code, semantic_code in enumerate(resources.semantic_code_by_raw):
        semantic[int(semantic_code)] = raw[raw_code]
    result = float(semantic[NEAR_IID_CODE])
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError("V787 near-IID responsibility is invalid")
    return result


def booster_probability(model: Any, row: np.ndarray) -> float:
    result = np.asarray(
        model.predict(row.reshape(1, -1), raw_score=False, num_threads=1),
        dtype=np.float64,
    )
    if result.shape != (1,) or not math.isfinite(float(result[0])):
        raise RuntimeError("V787 positional prediction failed")
    return float(result[0])


def curvature_correction(
    resources: Resources, raw: np.ndarray, anchor_probability: float, tick: int
) -> float:
    if int(tick) < CURVATURE_START:
        return 0.0
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise RuntimeError("V787 curvature row changed")
    standardized = np.clip(
        (values.astype(np.float64) - resources.curvature_center)
        / resources.curvature_scale,
        -4.0,
        4.0,
    ).astype(np.float32)
    positive = np.maximum(standardized, 0.0)
    gate = np.float32(np.clip(logit(anchor_probability) / 3.0, -1.0, 1.0))
    design = np.concatenate(
        (standardized, positive, standardized * gate, positive * gate)
    ).astype(np.float32)
    result = float(design @ resources.curvature_beta)
    if not math.isfinite(result):
        raise RuntimeError("V787 curvature correction is invalid")
    return result


def eprocess_increment(resources: Resources, feature_value: float, tick: int) -> float:
    tick = int(tick)
    if tick < resources.eprocess_boundary:
        return 0.0
    reference_tick = min(tick, resources.eprocess_offsets.size - 2)
    first = int(resources.eprocess_offsets[reference_tick])
    last = int(resources.eprocess_offsets[reference_tick + 1])
    reference = resources.eprocess_values[first:last]
    if reference.size == 0:
        raise RuntimeError("V787 V646 exact-time reference is empty")
    value = np.float32(feature_value)
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    ecdf = (left + right + 1.0) / (2.0 * (reference.size + 1.0))
    return resources.eprocess_scale * (2.0 * ecdf - 1.0)


def pairrank_correction(resources: Resources, row: np.ndarray) -> float:
    values = np.asarray(row, dtype=np.float32)
    if values.shape != (440,) or not np.isfinite(values).all():
        raise RuntimeError("V787 PairRank row changed")
    prediction = np.asarray(
        resources.pairrank.predict(
            values.reshape(1, -1), raw_score=True, num_threads=1
        ),
        dtype=np.float64,
    )
    if prediction.shape != (1,) or not math.isfinite(float(prediction[0])):
        raise RuntimeError("V787 PairRank prediction failed")
    return float(prediction[0])


__all__ = [
    "CURVATURE_START", "EPSILON", "OPEN_SCALE", "OCCUPATION_SCALE",
    "POSITIONAL_SHARE", "POSITIONAL_THRESHOLD", "Resources", "V643_BOUNDARY",
    "V643_SCALE", "booster_probability", "curvature_correction",
    "eprocess_increment", "load_resources", "logit", "near_iid_responsibility",
    "pairrank_correction", "sigmoid", "validate_resources",
]
