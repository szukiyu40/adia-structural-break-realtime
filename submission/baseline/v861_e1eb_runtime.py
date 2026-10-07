"""Frozen causal E1e-b checkpoint-ranker correction for the V844 parent.

The runtime keeps only the current series' history and its first 512 observed
online values.  A correction is recomputed at ticks 63, 127, 255, and 511,
then held until the next checkpoint.  It never observes the online horizon,
future values, or another test series.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from v201_features_runtime import FEATURE_NAMES, checkpoint_features


COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
CANDIDATE_ID = "v861_e1eb_checkpoint63_mean1_v844"
FULLFIT_CANDIDATE_ID = "v861_e1eb_checkpoint63_mean1_fullfit"
PARENT_CANDIDATE_ID = "v844_v826_parallel10_transport"
SCHEMA_VERSION = "adia-v861-e1eb-checkpoint63-mean1-v1"

MANIFEST_FILENAME = "v861_manifest.json"
MODEL_FILENAME = "v861_e1eb_ranker.json"
CALIBRATION_FILENAME = "v861_e1eb_calibration.npz"
FULLFIT_FILENAME = "v861_e1eb_fullfit_report.json"
AUTHORIZATION_FILENAME = "v861_e1eb_submission_authorization.json"
SOURCE_PATHS = frozenset(
    {"main.py", "v861_e1eb_runtime.py", "v201_features_runtime.py"}
)

SELECTED_FEATURES = (
    "regime_trend_strength",
    "interaction_location_episode_x_trend_strength",
    "interaction_trend_episode_x_trend_strength",
    "operator_shape_episode_log",
    "interaction_shape_episode_x_histogram_entropy",
    "operator_shape_suffix_log",
)
CHECKPOINTS = np.asarray((63, 127, 255, 511), dtype=np.int32)
ALPHA = 0.02
CORRECTION_CLIP = 4.0
# This is identical to ``time_ranker_runner._logit`` used for E1e-b OOF.
PROBABILITY_CLIP = 1e-12
MAX_RETAINED_ONLINE_VALUES = 512


def _contract() -> dict[str, Any]:
    """Return the exact immutable deployment contract stored in the manifest."""

    return {
        "features": list(SELECTED_FEATURES),
        "model_feature_count": len(SELECTED_FEATURES),
        "checkpoints": CHECKPOINTS.tolist(),
        "qid": "time_online",
        "objective": "rank:pairwise",
        "lambdarank_pair_method": "mean",
        "lambdarank_num_pair_per_sample": 1,
        "n_estimators": 160,
        "ecdf": "training_exact_time_right_midrank",
        "alpha": ALPHA,
        "correction_clip": CORRECTION_CLIP,
        "long_series_policy": "hold_latest_checkpoint_state",
        "maximum_online_values_retained": MAX_RETAINED_ONLINE_VALUES,
        "same_series_history_only": True,
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_test_series_values_used": False,
        "all_training_labels_used_for_frozen_fullfit_only": True,
        "fullfit_labels_used_for_model_selection": False,
        "reduced_or_public_labels_used": False,
        "deterministic": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ValidatedResources:
    model_path: Path
    checkpoints: np.ndarray
    feature_indices: np.ndarray
    sorted_values: tuple[np.ndarray, ...]
    model_center: float
    model_scale: float


@dataclass(frozen=True)
class LoadedResources:
    model: Any
    checkpoints: np.ndarray
    feature_indices: np.ndarray
    sorted_values: tuple[np.ndarray, ...]
    model_center: float
    model_scale: float


def _validate_calibration(path: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    float,
    float,
]:
    expected_arrays = {
        "checkpoints",
        "feature_indices",
        "model_center",
        "model_scale",
        *(f"sorted_{index}" for index in range(CHECKPOINTS.size)),
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_arrays:
            raise ValueError("V861 calibration inventory changed")
        checkpoints = archive["checkpoints"].astype(np.int32)
        feature_indices = archive["feature_indices"].astype(np.int32)
        center = archive["model_center"].astype(np.float64)
        scale = archive["model_scale"].astype(np.float64)
        sorted_values = tuple(
            archive[f"sorted_{index}"].astype(np.float32)
            for index in range(CHECKPOINTS.size)
        )

    expected_indices = np.asarray(
        [FEATURE_NAMES.index(name) for name in SELECTED_FEATURES], dtype=np.int32
    )
    if (
        not np.array_equal(checkpoints, CHECKPOINTS)
        or not np.array_equal(feature_indices, expected_indices)
        or center.shape != (1,)
        or scale.shape != (1,)
        or not np.isfinite(center[0])
        or not np.isfinite(scale[0])
        or float(scale[0]) <= 1e-8
    ):
        raise RuntimeError("V861 calibration header is invalid")
    for position, reference in enumerate(sorted_values):
        if (
            reference.ndim != 2
            or reference.shape[0] <= 0
            or reference.shape[1] != len(SELECTED_FEATURES)
            or not np.isfinite(reference).all()
            or np.any(reference[1:] < reference[:-1])
        ):
            raise RuntimeError(
                f"V861 exact-time ECDF reference is invalid: {position}"
            )
        reference.flags.writeable = False
    checkpoints.flags.writeable = False
    feature_indices.flags.writeable = False
    return (
        checkpoints,
        feature_indices,
        sorted_values,
        float(center[0]),
        float(scale[0]),
    )


def validate_resources(model_directory_path: str | Path) -> ValidatedResources:
    """Authenticate the frozen model, ECDF calibration, and provenance files."""

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
        "parent_candidate_id",
        "model_file",
        "model_sha256",
        "model_size_bytes",
        "calibration_file",
        "calibration_sha256",
        "calibration_size_bytes",
        "fullfit_report_file",
        "fullfit_report_sha256",
        "authorization_file",
        "authorization_sha256",
        "selected_features",
        "contract",
        "source_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("V861 manifest inventory changed")
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("parent_candidate_id", PARENT_CANDIDATE_ID),
        ("model_file", MODEL_FILENAME),
        ("calibration_file", CALIBRATION_FILENAME),
        ("fullfit_report_file", FULLFIT_FILENAME),
        ("authorization_file", AUTHORIZATION_FILENAME),
        ("selected_features", list(SELECTED_FEATURES)),
        ("contract", _contract()),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"V861 manifest {key} changed")

    source_root = Path(__file__).resolve().parent
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(SOURCE_PATHS):
        raise ValueError("V861 source inventory changed")
    for name in SOURCE_PATHS:
        digest = source_hashes.get(name)
        source_path = source_root / name
        if (
            not source_path.is_file()
            or not _is_sha256(digest)
            or _sha256_file(source_path) != digest
        ):
            raise RuntimeError(f"V861 source hash mismatch: {name}")

    for filename, hash_field, size_field in (
        (MODEL_FILENAME, "model_sha256", "model_size_bytes"),
        (CALIBRATION_FILENAME, "calibration_sha256", "calibration_size_bytes"),
    ):
        path = root / filename
        if (
            not path.is_file()
            or not _is_sha256(manifest.get(hash_field))
            or _sha256_file(path) != manifest[hash_field]
            or path.stat().st_size != manifest.get(size_field)
        ):
            raise RuntimeError(f"V861 immutable resource changed: {filename}")

    for filename, hash_field in (
        (FULLFIT_FILENAME, "fullfit_report_sha256"),
        (AUTHORIZATION_FILENAME, "authorization_sha256"),
    ):
        path = root / filename
        if (
            not path.is_file()
            or not _is_sha256(manifest.get(hash_field))
            or _sha256_file(path) != manifest[hash_field]
        ):
            raise RuntimeError(f"V861 provenance resource changed: {filename}")

    fullfit = json.loads((root / FULLFIT_FILENAME).read_text(encoding="utf-8"))
    artifacts = fullfit.get("artifacts", {})
    if (
        fullfit.get("status") != "complete"
        or fullfit.get("candidate_id") != FULLFIT_CANDIDATE_ID
        or fullfit.get("contract") != _contract()
        or artifacts.get("model_sha256") != manifest["model_sha256"]
        or artifacts.get("calibration_sha256") != manifest["calibration_sha256"]
    ):
        raise RuntimeError("V861 full-fit report is not production eligible")
    authorization = json.loads(
        (root / AUTHORIZATION_FILENAME).read_text(encoding="utf-8")
    )
    external = authorization.get("external_action", {})
    frozen = authorization.get("fullfit_contract", {})
    if (
        authorization.get("status") != "authorized_by_user"
        or authorization.get("candidate_id") != CANDIDATE_ID
        or external.get("crunchdao_code_and_model_upload_authorized") is not True
        or external.get("maximum_new_submissions") != 1
        or external.get("create_cloud_run") is not True
        or frozen.get("model_and_hyperparameters_frozen_before_fullfit") is not True
        or frozen.get("fullfit_labels_may_not_be_used_for_further_model_selection")
        is not True
        or frozen.get("reduced_or_public_labels_may_not_be_used") is not True
        or frozen.get("raw_tournament_data_may_not_be_packaged_or_uploaded") is not True
    ):
        raise RuntimeError("V861 submission authorization is invalid")

    calibration_path = root / CALIBRATION_FILENAME
    checkpoints, indices, references, center, scale = _validate_calibration(
        calibration_path
    )
    return ValidatedResources(
        model_path=root / MODEL_FILENAME,
        checkpoints=checkpoints,
        feature_indices=indices,
        sorted_values=references,
        model_center=center,
        model_scale=scale,
    )


def load_resources(model_directory_path: str | Path) -> LoadedResources:
    validated = validate_resources(model_directory_path)
    xgboost = importlib.import_module("xgboost")
    model = xgboost.Booster()
    model.load_model(validated.model_path)
    if int(model.num_features()) != len(SELECTED_FEATURES):
        raise RuntimeError("V861 loaded model feature count changed")
    return LoadedResources(
        model=model,
        checkpoints=validated.checkpoints,
        feature_indices=validated.feature_indices,
        sorted_values=validated.sorted_values,
        model_center=validated.model_center,
        model_scale=validated.model_scale,
    )


def _logit(probability: float) -> float:
    value = float(probability)
    if not math.isfinite(value):
        raise RuntimeError("V861 parent produced a non-finite probability")
    clipped = min(max(value, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid(score: float) -> float:
    if score >= 0.0:
        value = 1.0 / (1.0 + math.exp(-score))
    else:
        exponent = math.exp(score)
        value = exponent / (1.0 + exponent)
    return value


def _clean_history(values: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float]:
    history = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if history.size == 0:
        history = np.asarray([0.0], dtype=np.float64)
    finite = np.isfinite(history)
    replacement = float(np.median(history[finite])) if finite.any() else 0.0
    history[~finite] = replacement
    return history, replacement


def _ecdf_transform(
    raw_features: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    values = np.asarray(raw_features, dtype=np.float32).reshape(-1)
    if values.shape != (len(SELECTED_FEATURES),) or not np.isfinite(values).all():
        raise RuntimeError("V861 selected feature row is invalid")
    transformed = np.empty(values.size, dtype=np.float32)
    denominator = reference.shape[0] + 1.0
    for column, query in enumerate(values):
        right = int(np.searchsorted(reference[:, column], query, side="right"))
        transformed[column] = 2.0 * (right + 0.5) / denominator - 1.0
    if not np.isfinite(transformed).all():
        raise RuntimeError("V861 ECDF transform produced a non-finite value")
    return transformed


def _predict_raw(model: Any, row: np.ndarray) -> float:
    matrix = np.ascontiguousarray(row.reshape(1, -1), dtype=np.float32)
    predicted = np.asarray(
        model.inplace_predict(matrix, validate_features=True), dtype=np.float64
    ).reshape(-1)
    if predicted.shape != (1,) or not np.isfinite(predicted[0]):
        raise RuntimeError("V861 ranker produced an invalid prediction")
    return float(predicted[0])


class CausalE1EBState:
    """Series-local checkpoint state for the frozen E1e-b correction."""

    def __init__(
        self,
        historical: Sequence[float] | np.ndarray,
        resources: LoadedResources,
    ):
        self.history, self._replacement = _clean_history(historical)
        self.resources = resources
        self.online: list[float] = []
        self.time_online = -1
        self.correction = 0.0

    def update(self, point: float) -> None:
        self.time_online += 1
        value = float(point)
        if not math.isfinite(value):
            value = self._replacement
        if len(self.online) < MAX_RETAINED_ONLINE_VALUES:
            self.online.append(value)

        position = int(
            np.searchsorted(self.resources.checkpoints, self.time_online, side="left")
        )
        if (
            position >= self.resources.checkpoints.size
            or int(self.resources.checkpoints[position]) != self.time_online
        ):
            return
        encoded = checkpoint_features(
            self.history,
            np.asarray(self.online, dtype=np.float64),
            (self.time_online,),
        )
        if self.time_online not in encoded:
            raise RuntimeError(
                f"V861 feature runtime omitted checkpoint {self.time_online}"
            )
        raw = encoded[self.time_online][self.resources.feature_indices]
        transformed = _ecdf_transform(
            raw,
            self.resources.sorted_values[position],
        )
        raw_prediction = _predict_raw(self.resources.model, transformed)
        self.correction = float(
            np.clip(
                (raw_prediction - self.resources.model_center)
                / self.resources.model_scale,
                -CORRECTION_CLIP,
                CORRECTION_CLIP,
            )
        )

    def apply(self, parent_probability: float) -> float:
        if self.time_online < 0:
            raise RuntimeError("V861 apply called before update")
        if self.time_online < int(self.resources.checkpoints[0]):
            value = float(parent_probability)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError("V861 parent probability is outside [0, 1]")
            return value
        result = _sigmoid(
            _logit(parent_probability) + ALPHA * self.correction
        )
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise RuntimeError("V861 produced an invalid probability")
        return result


__all__ = [
    "ALPHA",
    "AUTHORIZATION_FILENAME",
    "CALIBRATION_FILENAME",
    "CANDIDATE_ID",
    "CHECKPOINTS",
    "CausalE1EBState",
    "FULLFIT_FILENAME",
    "LoadedResources",
    "MANIFEST_FILENAME",
    "MAX_RETAINED_ONLINE_VALUES",
    "MODEL_FILENAME",
    "SELECTED_FEATURES",
    "ValidatedResources",
    "load_resources",
    "validate_resources",
]
