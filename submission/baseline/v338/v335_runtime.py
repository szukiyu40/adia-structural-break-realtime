"""Fail-closed resource loader for the V335 loss-weighted V4 detector bag."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FILENAME = "v335_manifest.json"
CANDIDATE_ID = "v335_dual_partition_loss_weighted_bag"
COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
PARTITIONS = ("canonical", "balanced")
FEATURE_COUNT = 422
TREE_COUNT = 500
REPLACEMENT_SHARE = 0.25
V317_DETECTOR_ALPHA = 0.04
REPLACEMENT_SCALE = 2.0 * V317_DETECTOR_ALPHA * REPLACEMENT_SHARE


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class ValidatedResources:
    model_paths: tuple[Path, ...]
    feature_count: int


@dataclass(frozen=True)
class LoadedResources:
    boosters: tuple[Any, ...]
    feature_count: int


def validate_resources(model_directory_path: str | Path) -> ValidatedResources:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "status", "competition", "candidate_id", "feature_count",
        "tree_count", "partitions", "models", "fullfit_report_file",
        "fullfit_report_sha256", "oof_evaluation_file", "oof_evaluation_sha256",
        "contract", "source_sha256", "upstream_v329_manifest_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("V335 manifest inventory changed")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("feature_count", FEATURE_COUNT),
        ("tree_count", TREE_COUNT),
        ("partitions", list(PARTITIONS)),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"V335 manifest {key} changed")
    expected_contract = {
        "canonical_detector_share": 0.5,
        "balanced_detector_share": 0.5,
        "v256_replacement_share": REPLACEMENT_SHARE,
        "v317_detector_alpha": V317_DETECTOR_ALPHA,
        "replacement_scale": REPLACEMENT_SCALE,
        "hard_row_weight": 0.0,
        "same_time_counterpart_weight": 1.0,
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_test_series_used": False,
        "deterministic": True,
        "determinism_tolerance": 1e-8,
    }
    if manifest.get("contract") != expected_contract:
        raise ValueError("V335 manifest contract changed")
    source = manifest.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != {"main.py", "v335_runtime.py"}:
        raise ValueError("V335 source inventory changed")
    code_root = Path(__file__).resolve().parent
    for name, digest in source.items():
        if not _is_sha256(digest) or _sha(code_root / name) != digest:
            raise RuntimeError(f"V335 source changed: {name}")
    for file_key, digest_key in (
        ("fullfit_report_file", "fullfit_report_sha256"),
        ("oof_evaluation_file", "oof_evaluation_sha256"),
    ):
        path = root / str(manifest[file_key])
        digest = manifest[digest_key]
        if not path.is_file() or not _is_sha256(digest) or _sha(path) != digest:
            raise RuntimeError(f"V335 provenance changed: {file_key}")
    fullfit = json.loads((root / manifest["fullfit_report_file"]).read_text(encoding="utf-8"))
    oof = json.loads((root / manifest["oof_evaluation_file"]).read_text(encoding="utf-8"))
    if (
        fullfit.get("status") != "complete"
        or fullfit.get("candidate_id") != CANDIDATE_ID
        or oof.get("status") != "pass"
        or oof.get("decision", {}).get("advance_to_fullfit_runtime_audit") is not True
    ):
        raise RuntimeError("V335 provenance is not production eligible")
    records = manifest.get("models")
    if not isinstance(records, list) or len(records) != len(PARTITIONS):
        raise ValueError("V335 model inventory changed")
    paths = []
    fullfit_models = {row["partition"]: row for row in fullfit["models"]}
    for partition, record in zip(PARTITIONS, records, strict=True):
        expected_file = f"v335_models/{partition}.txt"
        if (
            not isinstance(record, Mapping)
            or set(record) != {"partition", "file", "sha256", "size_bytes"}
            or record.get("partition") != partition
            or record.get("file") != expected_file
        ):
            raise ValueError(f"V335 model record changed: {partition}")
        path = root / expected_file
        digest = record.get("sha256")
        source_record = fullfit_models[partition]
        if (
            not path.is_file()
            or not _is_sha256(digest)
            or _sha(path) != digest
            or path.stat().st_size != record.get("size_bytes")
            or source_record["sha256"] != digest
            or source_record["feature_count"] != FEATURE_COUNT
            or source_record["tree_count"] != TREE_COUNT
        ):
            raise RuntimeError(f"V335 model changed: {partition}")
        paths.append(path)
    return ValidatedResources(tuple(paths), FEATURE_COUNT)


def load_resources(model_directory_path: str | Path) -> LoadedResources:
    validated = validate_resources(model_directory_path)
    import lightgbm as lgb

    boosters = tuple(lgb.Booster(model_file=str(path)) for path in validated.model_paths)
    if any(
        booster.num_feature() != FEATURE_COUNT or booster.num_trees() != TREE_COUNT
        for booster in boosters
    ):
        raise RuntimeError("V335 loaded LightGBM metadata changed")
    return LoadedResources(boosters=boosters, feature_count=FEATURE_COUNT)


__all__ = [
    "FEATURE_COUNT", "REPLACEMENT_SCALE", "load_resources", "validate_resources",
]
