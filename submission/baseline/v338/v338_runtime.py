"""Fail-closed manifest validation for the V338 late specialist router."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FILENAME = "v338_manifest.json"
CANDIDATE_ID = "v338_minimax_shrunk_late_specialist"
COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
MIDDLE_START = 128
LATE_START = 512
EARLY_SCALE = 0.02
MIDDLE_SCALE = 0.04
LATE_SCALE = 0.03


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


def validate_resources(model_directory_path: str | Path) -> Mapping[str, Any]:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "status", "competition", "candidate_id", "schedule",
        "contract", "evaluation_file", "evaluation_sha256", "source_sha256",
        "upstream_v335_manifest_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("V338 manifest inventory changed")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"V338 manifest {key} changed")
    expected_schedule = {
        "0-127": {"canonical_share": 0.5, "replacement_share": 0.25, "scale": EARLY_SCALE},
        "128-511": {"canonical_share": 0.0, "replacement_share": 0.5, "scale": MIDDLE_SCALE},
        "512+": {"canonical_share": 0.0, "replacement_share": 0.375, "scale": LATE_SCALE},
    }
    if manifest.get("schedule") != expected_schedule:
        raise ValueError("V338 schedule changed")
    expected_contract = {
        "selector": "time_online_only",
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_test_series_used": False,
        "deterministic": True,
        "determinism_tolerance": 1e-8,
    }
    if manifest.get("contract") != expected_contract:
        raise ValueError("V338 contract changed")
    source = manifest.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != {"main.py", "v338_runtime.py"}:
        raise ValueError("V338 source inventory changed")
    code_root = Path(__file__).resolve().parent
    for name, digest in source.items():
        if not _is_sha256(digest) or _sha(code_root / name) != digest:
            raise RuntimeError(f"V338 source changed: {name}")
    evaluation = root / str(manifest["evaluation_file"])
    if (
        not evaluation.is_file()
        or not _is_sha256(manifest["evaluation_sha256"])
        or _sha(evaluation) != manifest["evaluation_sha256"]
    ):
        raise RuntimeError("V338 evaluation provenance changed")
    report = json.loads(evaluation.read_text(encoding="utf-8"))
    if (
        report.get("status") != "pass"
        or report.get("decision", {}).get("advance_to_production") is not True
    ):
        raise RuntimeError("V338 evaluation is not production eligible")
    upstream = root / "v335_manifest.json"
    if (
        not upstream.is_file()
        or not _is_sha256(manifest["upstream_v335_manifest_sha256"])
        or _sha(upstream) != manifest["upstream_v335_manifest_sha256"]
    ):
        raise RuntimeError("V338 upstream V335 provenance changed")
    return manifest


__all__ = [
    "EARLY_SCALE", "LATE_SCALE", "LATE_START", "MIDDLE_SCALE", "MIDDLE_START",
    "validate_resources",
]
