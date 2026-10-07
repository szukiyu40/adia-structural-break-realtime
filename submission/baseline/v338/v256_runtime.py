"""Fail-closed resource loader for the frozen ADIA V256 detector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FILENAME = "v256_manifest.json"
CANDIDATE_ID = "v256_three_seed_equal_series_logtick_v152"
COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
SEEDS = (0, 1, 2)
FEATURE_COUNT = 422
TREE_COUNT = 500


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
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
        "schema_version",
        "status",
        "competition",
        "candidate_id",
        "feature_count",
        "tree_count",
        "seeds",
        "models",
        "fullfit_report_file",
        "fullfit_report_sha256",
        "transport_audit_file",
        "transport_audit_sha256",
        "contract",
        "source_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("V256 manifest inventory changed")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("feature_count", FEATURE_COUNT),
        ("tree_count", TREE_COUNT),
        ("seeds", list(SEEDS)),
        ("fullfit_report_file", "v256_equal_series_logtick_fullfit.json"),
        ("transport_audit_file", "v256_reduced100_transport_audit.json"),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"V256 manifest {key} changed")
    expected_contract = {
        "alpha": 0.04,
        "blend": "V152 + alpha * (2 * mean_seed_probability - 1)",
        "arithmetic_probability_mean": True,
        "series_equal_total_weight": True,
        "log_spaced_ticks_per_series_maximum": 40,
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_test_series_used": False,
        "deterministic": True,
        "determinism_tolerance": 1e-8,
    }
    if manifest.get("contract") != expected_contract:
        raise ValueError("V256 manifest contract changed")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != {
        "main.py",
        "v256_runtime.py",
    }:
        raise ValueError("V256 source inventory changed")
    code_root = Path(__file__).resolve().parent
    for name, digest in source_hashes.items():
        if not _is_sha256(digest) or sha256_file(code_root / name) != digest:
            raise RuntimeError(f"V256 source changed: {name}")
    report_path = root / manifest["fullfit_report_file"]
    report_digest = manifest.get("fullfit_report_sha256")
    if (
        not report_path.is_file()
        or not _is_sha256(report_digest)
        or sha256_file(report_path) != report_digest
    ):
        raise RuntimeError("V256 fullfit report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("candidate_id") != CANDIDATE_ID
        or report.get("runtime") != {"lightgbm": "4.6.0", "numpy": "2.3.3"}
        or report.get("contract", {}).get("seeds") != list(SEEDS)
        or report.get("source_evaluation", {}).get("positive_folds") != 5
    ):
        raise RuntimeError("V256 fullfit provenance is not production eligible")
    audit_path = root / manifest["transport_audit_file"]
    audit_digest = manifest.get("transport_audit_sha256")
    if (
        not audit_path.is_file()
        or not _is_sha256(audit_digest)
        or sha256_file(audit_path) != audit_digest
    ):
        raise RuntimeError("V256 reduced transport audit changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("decision", {}).get(
            "authorize_production_packaging_and_code_submission"
        )
        is not True
        or audit.get("contract", {}).get("target_used_for_fitting_tuning_or_calibration")
        is not False
    ):
        raise RuntimeError("V256 reduced transport audit did not pass")
    model_records = manifest.get("models")
    if not isinstance(model_records, list) or len(model_records) != len(SEEDS):
        raise ValueError("V256 model inventory changed")
    paths = []
    report_models = {int(row["seed"]): row for row in report["models"]}
    for seed, record in zip(SEEDS, model_records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "seed",
            "file",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("V256 model record changed")
        expected_file = f"v256_models/seed_{seed}.txt"
        if record.get("seed") != seed or record.get("file") != expected_file:
            raise ValueError("V256 model ordering changed")
        path = root / expected_file
        digest = record.get("sha256")
        if (
            not path.is_file()
            or not _is_sha256(digest)
            or sha256_file(path) != digest
            or path.stat().st_size != record.get("size_bytes")
            or report_models[seed]["sha256"] != digest
            or report_models[seed]["feature_count"] != FEATURE_COUNT
            or report_models[seed]["tree_count"] != TREE_COUNT
        ):
            raise RuntimeError(f"V256 seed {seed} model changed")
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
        raise RuntimeError("V256 loaded LightGBM metadata changed")
    return LoadedResources(boosters=boosters, feature_count=FEATURE_COUNT)


__all__ = ["load_resources", "sha256_file", "validate_resources"]
