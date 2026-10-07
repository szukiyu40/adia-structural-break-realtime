"""Authenticated same-tick transport for PairRank, ExtraTrees, and V643."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import v791_tree_transport_runtime as standard
from v803_fast_merged_lightgbm_runtime import FastMultiPredictor


MODEL_NAME = "v824_v4_mega_tree.txt"
MANIFEST_NAME = "v824_v4_mega_tree_manifest.json"
AUDIT_NAME = "v825_v824_mega_tree_independent_audit.json"
_INSTALLED_FOR: Path | None = None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_resources(model_directory_path: str | Path) -> Path:
    root = Path(model_directory_path).expanduser().resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    model = manifest.get("model", {})
    if (
        manifest.get("schema_version") != "adia-v824-v4-mega-tree-resource-v1"
        or manifest.get("status") != "complete_pending_independent_audit"
        or manifest.get("target_loaded") is not False
        or model.get("filename") != MODEL_NAME
        or model.get("features") != 440
        or model.get("outputs") != 5
        or model.get("iterations") != 400
        or model.get("trees") != 2000
        or model.get("raw_score") is not True
    ):
        raise RuntimeError("V824 resource manifest changed")
    model_path = root / MODEL_NAME
    if (
        not model_path.is_file()
        or model_path.stat().st_size != int(model["size_bytes"])
        or _sha(model_path) != model["sha256"]
    ):
        raise RuntimeError("V824 mega-tree resource changed")
    audit = json.loads((root / AUDIT_NAME).read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or audit.get("target_loaded") is not False
        or audit.get("prototype_imported") is not False
        or audit.get("builder_imported") is not False
        or audit.get("runtime_imported") is not False
        or audit.get("rebuild_byte_exact") is not True
        or audit.get("outputs_bit_exact") is not True
        or audit.get("fast_replay_bit_exact") is not True
        or audit.get("decision", {}).get("authorize_runtime") is not True
    ):
        raise RuntimeError("V825 independent audit changed")
    standard.validate_resources(root)
    return root


class _DetectorCoordinator:
    def __init__(self, predictor: FastMultiPredictor) -> None:
        self.predictor = predictor
        self.cached = np.empty(5, dtype=np.float64)
        self.expected_class = 0
        self.row_identity: int | None = None

    def predict(self, class_index: int, data: np.ndarray, **kwargs) -> np.ndarray:
        if (
            set(kwargs) != {"raw_score", "num_threads"}
            or kwargs["raw_score"] is not False
            or int(kwargs["num_threads"]) != 1
        ):
            raise RuntimeError("V824 detector call signature changed")
        values = np.asarray(data)
        if values.shape != (1, 422) or values.dtype != np.float32:
            raise RuntimeError("V824 detector row changed")
        if class_index == 0:
            if self.expected_class != 0:
                raise RuntimeError("V824 detector sequence restarted early")
            self.cached[:] = self.predictor.predict(values[0])
            self.row_identity = id(data)
            self.expected_class = 1
        else:
            if class_index != self.expected_class or self.row_identity != id(data):
                raise RuntimeError("V824 detector source order changed")
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


class _MegaCoordinator:
    def __init__(self, predictor: FastMultiPredictor) -> None:
        self.predictor = predictor
        self.cached = np.empty(5, dtype=np.float64)
        self.row_identity: int | None = None

    def base(self, resources: Any, rows: Mapping[str, np.ndarray]):
        if set(rows) != {"v4_seq18"}:
            raise RuntimeError("V824 feature-view inventory changed")
        row = np.asarray(rows["v4_seq18"], dtype=np.float32)
        if row.shape != (440,) or not np.isfinite(row).all():
            raise RuntimeError("V824 V4 row changed")
        self.cached[:] = self.predictor.predict(row)
        self.row_identity = id(row)
        pair_total = 0.0
        for index, seed in enumerate(resources.pairrank_seeds):
            pair_total += seed.calibration.apply(float(self.cached[index]))
        pairrank = pair_total / len(resources.pairrank_seeds)
        extra_trees = float(self.cached[3])
        result = (pairrank, extra_trees)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result):
            raise RuntimeError("V824 base output is invalid")
        return result

    def mature(self, row: np.ndarray) -> float:
        values = np.asarray(row, dtype=np.float32)
        if values.shape != (440,) or id(values) != self.row_identity:
            raise RuntimeError("V824 mature correction row changed")
        result = float(self.cached[4])
        if not math.isfinite(result):
            raise RuntimeError("V824 mature correction is invalid")
        self.row_identity = None
        return result


def install(v787_parent_main: Any, model_directory_path: str | Path) -> None:
    global _INSTALLED_FOR
    root = validate_resources(model_directory_path)
    if _INSTALLED_FOR is not None:
        if _INSTALLED_FOR != root:
            raise RuntimeError("V824 installed for a different resource root")
        return
    import lightgbm as lgb

    mega_booster = lgb.Booster(model_file=str(root / MODEL_NAME))
    detector_booster = lgb.Booster(model_file=str(root / standard.DETECTOR_NAME))
    if (
        mega_booster.num_feature() != 440
        or mega_booster.num_trees() != 2000
        or mega_booster.current_iteration() != 400
        or detector_booster.num_feature() != 422
        or detector_booster.num_trees() != 2500
        or detector_booster.current_iteration() != 500
    ):
        raise RuntimeError("V824 loaded model structure changed")
    mega = _MegaCoordinator(FastMultiPredictor(mega_booster, 440, 5, True))
    detector = _DetectorCoordinator(
        FastMultiPredictor(detector_booster, 422, 5, False)
    )

    base_runtime = v787_parent_main.v338_main.base_runtime
    original_runtime = base_runtime.original

    def load_base_resources(directory: str | Path, *, check_runtime: bool = True):
        validated = original_runtime.validate_resources(
            directory, check_runtime=check_runtime
        )
        meta = base_runtime._validate_v738(Path(directory).expanduser().resolve())  # noqa: SLF001
        seeds = tuple(
            original_runtime.LoadedPairSeed(seed.seed, None, seed.calibration)
            for seed in validated.pairrank_seeds
        )
        return base_runtime.LoadedResources(
            pairrank_seeds=seeds,
            extratrees_model=validated.extratrees_model,
            meta=meta,
        )

    base_runtime.load_resources = load_base_resources
    base_runtime.predict_base_probabilities = mega.base
    v787_parent_main.pairrank_correction = lambda _resources, row: mega.mature(row)

    proxies = tuple(_DetectorProxy(detector, index) for index in range(5))
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

