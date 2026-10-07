"""Fail-closed causal detector-memory state for ADIA V317."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FILENAME = "v317_manifest.json"
CANDIDATE_ID = "v317_v256_runtime_space_decayed_memory"
COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
PROBABILITY_CLIP = 1e-7
ALPHA = 0.04
DETECTOR_MEMORY_DECAY = 0.99


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _clip(value: float) -> float:
    if not math.isfinite(float(value)):
        raise RuntimeError("V317 non-finite probability")
    return min(max(float(value), PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)


class CausalDetectorMemory:
    """One scalar of same-series state; update after consuming the current tick."""

    __slots__ = ("memory",)

    def __init__(self) -> None:
        self.memory: float | None = None

    def update(self, v152_probability: float, detector_probability: float) -> float:
        current = _clip(detector_probability)
        self.memory = current if self.memory is None else max(current, DETECTOR_MEMORY_DECAY * self.memory)
        return _clip(float(v152_probability) + ALPHA * (2.0 * self.memory - 1.0))


def validate_resources(model_directory_path: str | Path) -> Mapping[str, Any]:
    root = Path(model_directory_path).expanduser().resolve()
    path = root / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("V317 manifest changed")
    for key, expected in (("schema_version", 1), ("status", "complete"), ("competition", COMPETITION), ("candidate_id", CANDIDATE_ID)):
        if manifest.get(key) != expected:
            raise ValueError(f"V317 manifest {key} changed")
    expected_contract = {
        "upstream_v256_models_unchanged": True,
        "detector_memory_decay": DETECTOR_MEMORY_DECAY,
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_test_series_used": False,
        "deterministic": True,
        "determinism_tolerance": 1e-8,
    }
    if manifest.get("contract") != expected_contract:
        raise ValueError("V317 manifest contract changed")
    source = manifest.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != {"main.py", "v256_runtime.py", "v317_runtime.py"}:
        raise ValueError("V317 source inventory changed")
    code_root = Path(__file__).resolve().parent
    for name, digest in source.items():
        if not _is_sha256(digest) or _sha(code_root / name) != digest:
            raise RuntimeError(f"V317 source changed: {name}")
    return manifest


__all__ = ["ALPHA", "CausalDetectorMemory", "DETECTOR_MEMORY_DECAY", "PROBABILITY_CLIP", "validate_resources"]

