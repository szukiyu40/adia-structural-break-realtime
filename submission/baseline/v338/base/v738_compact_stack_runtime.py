"""Authenticated V4-only PairRank/ExtraTrees replacement for the V13 core."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import ORIGINAL_V13_RUNTIME as original


FEATURE_VIEWS = ("v4_seq18",)
PAIRRANK_SEEDS = original.PAIRRANK_SEEDS
SinglePassFeatureState = original.SinglePassFeatureState
feature_runtime = original.feature_runtime
EPSILON = 1e-7
MANIFEST_FILENAME = "v738_manifest.json"
META_FILENAME = "v738_compact_meta.json"
CANDIDATE_ID = "v738_two_expert_v4_core"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponential = math.exp(value)
    return float(exponential / (1.0 + exponential))


@dataclass(frozen=True)
class CompactMeta:
    alpha: tuple[float, float]
    intercept: float

    def predict(self, probabilities: Sequence[float]) -> float:
        if len(probabilities) != 2:
            raise RuntimeError("V738 compact meta requires exactly two experts")
        score = self.intercept
        for coefficient, raw in zip(self.alpha, probabilities, strict=True):
            probability = min(max(float(raw), EPSILON), 1.0 - EPSILON)
            if not math.isfinite(probability):
                raise RuntimeError("V738 expert probability is non-finite")
            score += coefficient * (math.log(probability) - math.log1p(-probability))
        result = _sigmoid(score)
        if not math.isfinite(result):
            raise RuntimeError("V738 compact meta output is non-finite")
        return min(max(result, EPSILON), 1.0 - EPSILON)


@dataclass(frozen=True)
class LoadedResources:
    pairrank_seeds: tuple[Any, ...]
    extratrees_model: Any
    meta: CompactMeta


def _validate_v738(root: Path) -> CompactMeta:
    manifest_path = root / MANIFEST_FILENAME
    before = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if before != _sha256(manifest_path) or not isinstance(manifest, Mapping):
        raise RuntimeError("V738 manifest is unstable or invalid")
    expected_contract = {
        "base_order": ["pairrank", "extra_trees"],
        "feature_views": ["v4_seq18"],
        "outer_pure_selection_source": "v737",
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_test_series_used": False,
        "deterministic": True,
        "infer_parallelism": 4,
    }
    if (
        manifest.get("schema_version") != "adia-v738-production-v1"
        or manifest.get("status") != "complete"
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("contract") != expected_contract
    ):
        raise RuntimeError("V738 production contract changed")
    code_root = Path(__file__).resolve().parents[1]
    for relative, digest in manifest.get("source_sha256", {}).items():
        path = (code_root / relative).resolve()
        if code_root not in path.parents or not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"V738 source changed: {relative}")
    for relative, digest in manifest.get("upstream_sha256", {}).items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"V738 upstream resource changed: {relative}")
    meta_path = root / META_FILENAME
    if _sha256(meta_path) != manifest.get("compact_meta_sha256"):
        raise RuntimeError("V738 compact meta resource changed")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        payload.get("base_order") != ["pairrank", "extra_trees"]
        or payload.get("probability_clip") != EPSILON
        or payload.get("metrics_computed") is not False
        or payload.get("complete_nested_selection_source") != "v737"
    ):
        raise RuntimeError("V738 compact meta schema changed")
    coefficients = payload.get("raw_logit_coef_by_base", {})
    alpha = (float(coefficients["pairrank"]), float(coefficients["extra_trees"]))
    intercept = float(payload["raw_logit_intercept"])
    if not all(math.isfinite(value) for value in (*alpha, intercept)):
        raise RuntimeError("V738 compact meta parameters are invalid")
    return CompactMeta(alpha, intercept)


def validate_resources(model_directory_path: str | Path, *, check_runtime: bool = True):
    root = Path(model_directory_path).expanduser().resolve()
    validated = original.validate_resources(root, check_runtime=check_runtime)
    _validate_v738(root)
    return validated


def load_resources(model_directory_path: str | Path, *, check_runtime: bool = True) -> LoadedResources:
    root = Path(model_directory_path).expanduser().resolve()
    validated = original.validate_resources(root, check_runtime=check_runtime)
    meta = _validate_v738(root)
    lightgbm = importlib.import_module("lightgbm")
    pairrank = []
    for seed in validated.pairrank_seeds:
        booster = lightgbm.Booster(model_file=str(seed.model_path))
        if booster.num_feature() != seed.feature_count or booster.num_trees() != seed.tree_count:
            raise RuntimeError("V738 PairRank booster structure changed")
        pairrank.append(original.LoadedPairSeed(seed.seed, booster, seed.calibration))
    return LoadedResources(tuple(pairrank), validated.extratrees_model, meta)


def predict_base_probabilities(resources: LoadedResources, rows: Mapping[str, np.ndarray]) -> tuple[float, float]:
    if set(rows) != {"v4_seq18"}:
        raise RuntimeError("V738 feature-view inventory changed")
    row = np.asarray(rows["v4_seq18"], dtype=np.float32)
    expected = feature_runtime.VIEW_FEATURE_COUNTS["v4_seq18"]
    if row.shape != (expected,) or not np.isfinite(row).all():
        raise RuntimeError("V738 V4 feature row changed")
    pair_total = 0.0
    for seed in resources.pairrank_seeds:
        raw = original._one_prediction(seed.booster, row, raw_score=True)  # noqa: SLF001
        pair_total += seed.calibration.apply(raw)
    pairrank = pair_total / len(resources.pairrank_seeds)
    extra_trees = float(resources.extratrees_model.predict_one(row))
    result = (pairrank, extra_trees)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result):
        raise RuntimeError("V738 expert output is invalid")
    return result


def train_impl(datasets: Any, model_directory_path: str, *, _check_runtime: bool = True) -> None:
    original.train_impl(datasets, model_directory_path, _check_runtime=_check_runtime)
    _validate_v738(Path(model_directory_path).expanduser().resolve())


__all__ = [
    "FEATURE_VIEWS",
    "SinglePassFeatureState",
    "feature_runtime",
    "load_resources",
    "predict_base_probabilities",
    "train_impl",
    "validate_resources",
]
