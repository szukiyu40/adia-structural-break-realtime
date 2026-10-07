"""Exact-call transport from eight frozen boosters to two multi-output boosters."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PAIR_NAME = "v791_pairrank_multiclass.txt"
DETECTOR_NAME = "v791_detector_multiclassova.txt"
MANIFEST_NAME = "v791_tree_resources_manifest.json"
AUDIT_NAME = "v792_v791_tree_transport_independent_audit.json"
_INSTALLED_FOR: Path | None = None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_resources(model_directory_path: str | Path) -> Path:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "adia-v791-multiclass-tree-resources-v1"
        or manifest.get("status") != "complete_pending_independent_audit"
        or manifest.get("target_loaded") is not False
    ):
        raise RuntimeError("V791 transport manifest changed")
    for name, expected in manifest.get("merged", {}).items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["size_bytes"])
            or _sha(path) != expected["sha256"]
        ):
            raise RuntimeError(f"V791 merged resource changed: {name}")
    audit_path = root / AUDIT_NAME
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or audit.get("builder_imported") is not False
        or audit.get("prototype_imported") is not False
        or audit.get("runtime_imported") is not False
        or audit.get("target_loaded") is not False
        or audit.get("pair_rebuild_byte_exact") is not True
        or audit.get("detector_rebuild_byte_exact") is not True
        or audit.get("pair_predictions_bit_exact") is not True
        or audit.get("detector_predictions_bit_exact") is not True
        or audit.get("decision", {}).get("authorize_candidate_assembly") is not True
    ):
        raise RuntimeError("V792 independent transport audit changed")
    return root


class _DetectorCoordinator:
    def __init__(self, booster: Any) -> None:
        self.booster = booster
        self.cached = np.empty(5, dtype=np.float64)
        self.expected_class = 0
        self.row_identity: int | None = None

    def predict(self, class_index: int, data: np.ndarray, **kwargs) -> np.ndarray:
        if set(kwargs) != {"raw_score", "num_threads"}:
            raise RuntimeError("V791 detector call signature changed")
        if kwargs["raw_score"] is not False or int(kwargs["num_threads"]) != 1:
            raise RuntimeError("V791 detector prediction mode changed")
        values = np.asarray(data)
        if values.shape != (1, 422) or values.dtype != np.float32:
            raise RuntimeError("V791 detector row changed")
        if class_index == 0:
            if self.expected_class != 0:
                raise RuntimeError("V791 detector sequence restarted early")
            prediction = np.asarray(
                self.booster.predict(
                    values, raw_score=False, num_threads=1
                ),
                dtype=np.float64,
            )
            if prediction.shape != (1, 5) or not np.isfinite(prediction).all():
                raise RuntimeError("V791 merged detector prediction failed")
            self.cached[:] = prediction[0]
            self.row_identity = id(data)
            self.expected_class = 1
        else:
            if (
                class_index != self.expected_class
                or self.row_identity != id(data)
            ):
                raise RuntimeError("V791 detector source order changed")
            self.expected_class += 1
        result = np.asarray((self.cached[class_index],), dtype=np.float64)
        if class_index == 4:
            self.expected_class = 0
            self.row_identity = None
        return result


class _DetectorProxy:
    def __init__(self, coordinator: _DetectorCoordinator, class_index: int) -> None:
        self.coordinator = coordinator
        self.class_index = int(class_index)

    def predict(self, data, **kwargs):
        return self.coordinator.predict(self.class_index, data, **kwargs)

    @staticmethod
    def num_feature() -> int:
        return 422

    @staticmethod
    def num_trees() -> int:
        return 500


def install(v787_parent_main: Any, model_directory_path: str | Path) -> None:
    global _INSTALLED_FOR
    root = validate_resources(model_directory_path)
    if _INSTALLED_FOR is not None:
        if _INSTALLED_FOR != root:
            raise RuntimeError("V791 transport installed for a different resource root")
        return

    import lightgbm as lgb

    pair_booster = lgb.Booster(model_file=str(root / PAIR_NAME))
    detector_booster = lgb.Booster(model_file=str(root / DETECTOR_NAME))
    if (
        pair_booster.num_feature() != 440
        or pair_booster.num_trees() != 900
        or pair_booster.current_iteration() != 300
        or detector_booster.num_feature() != 422
        or detector_booster.num_trees() != 2500
        or detector_booster.current_iteration() != 500
    ):
        raise RuntimeError("V791 merged LightGBM structure changed")

    base_runtime = v787_parent_main.v338_main.base_runtime
    original_runtime = base_runtime.original

    def load_base_resources(
        directory: str | Path, *, check_runtime: bool = True
    ):
        validated = original_runtime.validate_resources(
            directory, check_runtime=check_runtime
        )
        meta = base_runtime._validate_v738(Path(directory).expanduser().resolve())  # noqa: SLF001
        seeds = tuple(
            original_runtime.LoadedPairSeed(
                seed.seed, None, seed.calibration
            )
            for seed in validated.pairrank_seeds
        )
        return base_runtime.LoadedResources(
            pairrank_seeds=seeds,
            extratrees_model=validated.extratrees_model,
            meta=meta,
        )

    def predict_base_probabilities(resources: Any, rows: Mapping[str, np.ndarray]):
        if set(rows) != {"v4_seq18"}:
            raise RuntimeError("V791 feature-view inventory changed")
        row = np.asarray(rows["v4_seq18"], dtype=np.float32)
        if row.shape != (440,) or not np.isfinite(row).all():
            raise RuntimeError("V791 PairRank row changed")
        raw = np.asarray(
            pair_booster.predict(
                row.reshape(1, -1), raw_score=True, num_threads=1
            ),
            dtype=np.float64,
        )
        if raw.shape != (1, 3) or not np.isfinite(raw).all():
            raise RuntimeError("V791 merged PairRank prediction failed")
        pair_total = 0.0
        for index, seed in enumerate(resources.pairrank_seeds):
            pair_total += seed.calibration.apply(float(raw[0, index]))
        pairrank = pair_total / len(resources.pairrank_seeds)
        extra_trees = float(resources.extratrees_model.predict_one(row))
        result = (pairrank, extra_trees)
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in result
        ):
            raise RuntimeError("V791 base output is invalid")
        return result

    base_runtime.load_resources = load_base_resources
    base_runtime.predict_base_probabilities = predict_base_probabilities

    coordinator = _DetectorCoordinator(detector_booster)
    proxies = tuple(_DetectorProxy(coordinator, index) for index in range(5))
    v338_main = v787_parent_main.v338_main
    v256_module = __import__(v338_main.load_v256_resources.__module__)
    v335_module = __import__(v338_main.load_v335_resources.__module__)
    validate_v256 = v338_main.validate_v256_resources
    validate_v335 = v338_main.validate_v335_resources

    def load_v256(directory: str | Path):
        validated = validate_v256(directory)
        return v256_module.LoadedResources(
            boosters=proxies[:3], feature_count=validated.feature_count
        )

    def load_v335(directory: str | Path):
        validated = validate_v335(directory)
        return v335_module.LoadedResources(
            boosters=proxies[3:], feature_count=validated.feature_count
        )

    v338_main.load_v256_resources = load_v256
    v338_main.load_v335_resources = load_v335
    _INSTALLED_FOR = root


__all__ = ["install", "validate_resources"]

