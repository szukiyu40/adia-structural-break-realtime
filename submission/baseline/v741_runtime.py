"""Authenticated parent resources for the ADIA V741 production candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from tail.v496_correction_runtime import CorrectionResources, load_resources as load_corrections
from tail.v505_plateau_runtime import validate_resources as validate_plateau_resources


CANDIDATE_ID = "v741_v740_production_faithful_candidate"
MANIFEST_FILENAME = "v741_manifest.json"
MANIFEST_SCHEMA = "adia-v741-production-v1"

EXPECTED_CONTRACT = {
    "base": "v738_two_expert_v4_core",
    "structure": "v377_mature_absorption_gate",
    "tail": "v441_pit_plus_v483_fisher_plus_v495_memory_plus_v504_plateau",
    "future_ticks_used": False,
    "peer_test_series_used": False,
    "final_online_length_used": False,
    "series_identifier_used": False,
    "supervised_online_updates": False,
    "deterministic": True,
    "infer_parallelism": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_resources(model_directory_path: str | Path) -> CorrectionResources:
    """Validate V741 identity and load only its lightweight correction tables."""

    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    before = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if before != _sha256(manifest_path) or not isinstance(manifest, Mapping):
        raise RuntimeError("V741 manifest is unstable or invalid")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("contract") != EXPECTED_CONTRACT
    ):
        raise RuntimeError("V741 production identity changed")

    code_root = Path(__file__).resolve().parent
    for relative, digest in manifest.get("source_sha256", {}).items():
        source = (code_root / relative).resolve()
        if (
            code_root not in source.parents
            or not source.is_file()
            or _sha256(source) != digest
        ):
            raise RuntimeError(f"V741 source changed: {relative}")
    for filename, digest in manifest.get("parent_manifest_sha256", {}).items():
        parent = root / filename
        if not parent.is_file() or _sha256(parent) != digest:
            raise RuntimeError(f"V741 parent manifest changed: {filename}")

    corrections = load_corrections(root)
    validate_plateau_resources(root)
    return corrections


__all__ = ["EXPECTED_CONTRACT", "load_resources"]
