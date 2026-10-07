"""Authenticated constant-memory V504 current-plateau correction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


CANDIDATE_ID = "v505_v504_plateau_fast_production"
MANIFEST_FILENAME = "v505_manifest.json"
MANIFEST_SCHEMA = "adia-v505-production-v1"
BOUNDARY = 128
CALIBRATION_START = 64
ALPHA = 0.04

EXPECTED_CONTRACT = {
    "parent": "v496_v495_fast_production",
    "formula": "V496_plus_0.04_centered_current_q75_run_fraction",
    "boundary": BOUNDARY,
    "calibration_start": CALIBRATION_START,
    "alpha": ALPHA,
    "future_ticks_used": False,
    "peer_test_series_used": False,
    "final_online_length_used": False,
    "deterministic": True,
    "infer_parallelism": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class PlateauState:
    """Series-local q75 calibration and current post-boundary run length."""

    def __init__(self) -> None:
        self.calibration: list[float] = []
        self.threshold: float | None = None
        self.elapsed = 0
        self.current_run = 0

    def update(self, pre_memory_margin: float, post_memory_margin: float, tick: int) -> float:
        before = float(pre_memory_margin)
        after = float(post_memory_margin)
        tick = int(tick)
        if not math.isfinite(before) or not math.isfinite(after):
            raise RuntimeError("V505 plateau input is non-finite")
        if tick < BOUNDARY:
            if tick >= CALIBRATION_START:
                self.calibration.append(before)
            if tick == BOUNDARY - 1:
                if len(self.calibration) != BOUNDARY - CALIBRATION_START:
                    raise RuntimeError("V505 plateau calibration count changed")
                self.threshold = float(
                    np.quantile(
                        np.asarray(self.calibration, dtype=np.float64),
                        0.75,
                        method="linear",
                    )
                )
            return after
        if self.threshold is None:
            raise RuntimeError("V505 plateau started before q75 froze")
        self.elapsed += 1
        self.current_run = self.current_run + 1 if before > self.threshold else 0
        # V503/V504 persist this target-free state as float32 before applying
        # the fixed correction.  Preserve that exact transport arithmetic.
        fraction = float(np.float32(self.current_run / self.elapsed))
        result = after + ALPHA * (2.0 * fraction - 1.0)
        if not math.isfinite(result):
            raise RuntimeError("V505 plateau output is non-finite")
        return result


def validate_resources(model_directory_path: str | Path) -> Mapping[str, object]:
    root = Path(model_directory_path).expanduser().resolve()
    path = root / MANIFEST_FILENAME
    before = _sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if before != _sha256(path) or not isinstance(manifest, Mapping):
        raise RuntimeError("V505 manifest is unstable or invalid")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("contract") != EXPECTED_CONTRACT
    ):
        raise RuntimeError("V505 production identity changed")
    code_root = Path(__file__).resolve().parent
    for relative, digest in manifest.get("source_files", {}).items():
        source = (code_root / relative).resolve()
        if code_root not in source.parents or not source.is_file() or _sha256(source) != digest:
            raise RuntimeError(f"V505 source changed: {relative}")
    parent = root / "v496_manifest.json"
    if _sha256(parent) != manifest.get("parent_v496_manifest_sha256"):
        raise RuntimeError("V505 parent V496 manifest changed")
    return manifest


__all__ = ["ALPHA", "BOUNDARY", "CALIBRATION_START", "PlateauState", "validate_resources"]
