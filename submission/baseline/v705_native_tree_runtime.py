"""Authenticated native replacements for V704's frozen flat tree evaluators."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FEATURE_COUNT = 440
FILES = {
    "v705_extratrees_transport.txt": (
        "b153436143ceb9cab8e73cbb68e4d3d6397562d107016ed34a4f801c00f67964",
        50_722_176,
        400,
    ),
    "v705_xgboost_transport.txt": (
        "e2c423c462f022a9ac1ef456128102539aec2936fcee6b42ee14003a78d46cd0",
        2_541_391,
        600,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class NativeTreePredictor:
    """Expose the frozen one-row contract expected by the V13 base stack."""

    def __init__(self, booster) -> None:
        self.booster = booster

    def predict_one(self, features) -> float:
        row = np.asarray(features)
        if row.shape != (FEATURE_COUNT,) or row.dtype != np.dtype("float32"):
            raise RuntimeError("V705 native tree row changed")
        values = np.asarray(
            self.booster.predict(
                row.reshape(1, -1), raw_score=False, num_threads=1
            ),
            dtype=np.float64,
        )
        if values.shape != (1,) or not math.isfinite(float(values[0])):
            raise RuntimeError("V705 native tree prediction failed")
        probability = float(values[0])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("V705 native tree probability is outside [0, 1]")
        return probability


def load(model_directory_path: str):
    """Authenticate and load both target-blind translated tree models."""

    import lightgbm as lgb

    root = Path(model_directory_path).expanduser().resolve()
    loaded = {}
    for name, (expected_sha, expected_size, expected_trees) in FILES.items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_sha
        ):
            raise RuntimeError(f"V705 native tree resource changed: {name}")
        booster = lgb.Booster(model_file=str(path))
        if (
            booster.num_feature() != FEATURE_COUNT
            or booster.num_trees() != expected_trees
        ):
            raise RuntimeError(f"V705 native tree structure changed: {name}")
        loaded[name] = NativeTreePredictor(booster)
    return (
        loaded["v705_xgboost_transport.txt"],
        loaded["v705_extratrees_transport.txt"],
    )


def install_lightweight_parent_validation(base_runtime):
    """Keep parent hash/schema validation while avoiding obsolete array loads."""

    original_xgboost = base_runtime._validate_xgboost_component  # noqa: SLF001
    original_extratrees = base_runtime._validate_extratrees_component  # noqa: SLF001

    def validate_xgboost(root: Path, raw: Mapping[str, Any]) -> object:
        expected_fields = {
            "name", "kind", "feature_view", "feature_count", "tree_count",
            "model_file", "sha256", "size_bytes", "model_manifest_file",
            "model_manifest_sha256", "model_manifest_size_bytes",
            "base_margin", "arrays",
        }
        if set(raw) != expected_fields:
            raise ValueError("V705 XGBoost component field inventory changed")
        for key, expected in (
            ("name", "xgb"),
            ("kind", "flat_xgboost_binary_logistic"),
            ("feature_view", "v4_seq18"),
            ("feature_count", 440),
            ("tree_count", 600),
        ):
            base_runtime._require_exact(  # noqa: SLF001
                raw.get(key), expected, f"V705 XGBoost component {key}"
            )
        base_runtime._validate_file(  # noqa: SLF001
            root,
            relative=raw.get("model_file"),
            digest=raw.get("sha256"),
            size_bytes=raw.get("size_bytes"),
            description="V705 source flat XGBoost model",
        )
        manifest_path = base_runtime._validate_file(  # noqa: SLF001
            root,
            relative=raw.get("model_manifest_file"),
            digest=raw.get("model_manifest_sha256"),
            size_bytes=raw.get("model_manifest_size_bytes"),
            description="V705 source flat XGBoost manifest",
        )
        exporter_manifest, _ = base_runtime._read_stable_json(  # noqa: SLF001
            manifest_path, "V705 source flat XGBoost manifest"
        )
        output = base_runtime._require_mapping(  # noqa: SLF001
            exporter_manifest.get("output"), "V705 XGBoost manifest output"
        )
        model_contract = base_runtime._require_mapping(  # noqa: SLF001
            exporter_manifest.get("model"), "V705 XGBoost manifest model"
        )
        base_runtime._require_exact(  # noqa: SLF001
            output.get("arrays"), raw.get("arrays"), "V705 XGBoost arrays"
        )
        base_runtime._require_exact(  # noqa: SLF001
            model_contract.get("base_margin"),
            raw.get("base_margin"),
            "V705 XGBoost base margin",
        )
        array_records = base_runtime._require_mapping(  # noqa: SLF001
            raw.get("arrays"), "V705 XGBoost array records"
        )
        expected_shapes = {
            "tree_roots": (600,),
            "iteration_indptr": (601,),
            "children_left": (74_580,),
            "children_right": (74_580,),
            "split_index": (74_580,),
            "split_condition": (74_580,),
            "default_left": (74_580,),
            "leaf_weight": (74_580,),
            "base_margin": (1,),
        }
        if set(array_records) != set(expected_shapes):
            raise RuntimeError("V705 XGBoost array inventory changed")
        for name, shape in expected_shapes.items():
            base_runtime.xgboost_flat_runtime._validate_array_record(  # noqa: SLF001
                array_records[name], name=name, expected_shape=shape
            )
        return object()

    def validate_extratrees(root: Path, raw: Mapping[str, Any]) -> object:
        expected_fields = {
            "name", "kind", "feature_view", "feature_count", "tree_count",
            "model_file", "sha256", "size_bytes", "artifact_manifest_file",
            "artifact_manifest_sha256", "artifact_manifest_size_bytes", "arrays",
        }
        if set(raw) != expected_fields:
            raise ValueError("V705 ExtraTrees component field inventory changed")
        for key, expected in (
            ("name", "extra_trees"),
            ("kind", "flat_extratrees_probability"),
            ("feature_view", "v4_seq18"),
            ("feature_count", 440),
            ("tree_count", 400),
        ):
            base_runtime._require_exact(  # noqa: SLF001
                raw.get(key), expected, f"V705 ExtraTrees component {key}"
            )
        base_runtime._validate_file(  # noqa: SLF001
            root,
            relative=raw.get("model_file"),
            digest=raw.get("sha256"),
            size_bytes=raw.get("size_bytes"),
            description="V705 source flat ExtraTrees model",
        )
        artifact_path = base_runtime._validate_file(  # noqa: SLF001
            root,
            relative=raw.get("artifact_manifest_file"),
            digest=raw.get("artifact_manifest_sha256"),
            size_bytes=raw.get("artifact_manifest_size_bytes"),
            description="V705 source ExtraTrees artifact manifest",
        )
        artifact, _ = base_runtime._read_stable_json(  # noqa: SLF001
            artifact_path, "V705 source ExtraTrees artifact manifest"
        )
        for key, expected in (
            ("schema_version", 1),
            ("status", "complete"),
            ("competition", base_runtime.COMPETITION),
            ("candidate_id", "v13_extratrees_v4_seq18_fullfit"),
            ("feature_view", "v4_seq18"),
            ("feature_count", 440),
            ("metrics_computed", False),
        ):
            base_runtime._require_exact(  # noqa: SLF001
                artifact.get(key), expected, f"V705 ExtraTrees artifact {key}"
            )
        base_runtime._require_exact(  # noqa: SLF001
            artifact.get("flat_arrays"),
            raw.get("arrays"),
            "V705 ExtraTrees flat arrays",
        )
        artifact_model = base_runtime._require_mapping(  # noqa: SLF001
            artifact.get("flat_model"), "V705 ExtraTrees artifact flat model"
        )
        for key, expected in (
            ("sha256", raw.get("sha256")),
            ("size_bytes", raw.get("size_bytes")),
        ):
            base_runtime._require_exact(  # noqa: SLF001
                artifact_model.get(key),
                expected,
                f"V705 ExtraTrees artifact flat model {key}",
            )
        base_runtime.extratrees_runtime._validate_array_records(  # noqa: SLF001
            raw.get("arrays")
        )
        return object()

    base_runtime._validate_xgboost_component = validate_xgboost  # noqa: SLF001
    base_runtime._validate_extratrees_component = validate_extratrees  # noqa: SLF001
    return original_xgboost, original_extratrees


def restore_parent_validation(base_runtime, originals) -> None:
    original_xgboost, original_extratrees = originals
    base_runtime._validate_xgboost_component = original_xgboost  # noqa: SLF001
    base_runtime._validate_extratrees_component = original_extratrees  # noqa: SLF001


__all__ = [
    "NativeTreePredictor",
    "install_lightweight_parent_validation",
    "load",
    "restore_parent_validation",
]
