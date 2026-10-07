"""Authenticated causal Fisher-energy and symmetric-memory correction for V496."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


CANDIDATE_ID = "v496_v495_fast_production"
MANIFEST_FILENAME = "v496_manifest.json"
MANIFEST_SCHEMA = "adia-v496-production-v1"
EPSILON = 1e-7
PIT_FEATURE_INDEX = 14
PIT_FEATURE_NAME = "pit_slow_ewma"
FISHER_FEATURE_NAME = "energy_fisher_mean"
PIT_ALPHA = 0.04
FISHER_ALPHA = 0.04
GATE_ZERO_END = 63
GATE_FULL_START = 127
ENERGY_COUNT = 5
P_FLOOR = 1e-6
MEMORY_BOUNDARY = 128
MEMORY_FLOOR_START = 64
MEMORY_TOP_COUNT = 32
MEMORY_ALPHA = 0.08

EXPECTED_CONTRACT = {
    "anchor": "v399_prediction_identical",
    "formula": "logit_v399_plus_v441_pit_plus_v483_fisher_plus_v495_symmetric_memory",
    "pit_feature": PIT_FEATURE_NAME,
    "pit_alpha": PIT_ALPHA,
    "fisher_feature": FISHER_FEATURE_NAME,
    "fisher_alpha": FISHER_ALPHA,
    "gate_zero_end": GATE_ZERO_END,
    "gate_full_start": GATE_FULL_START,
    "memory_boundary": MEMORY_BOUNDARY,
    "memory_floor_start": MEMORY_FLOOR_START,
    "memory_top_count": MEMORY_TOP_COUNT,
    "memory_alpha": MEMORY_ALPHA,
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


def _logit(value: float) -> float:
    clipped = min(max(float(value), EPSILON), 1.0 - EPSILON)
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exponential = math.exp(value)
    return float(exponential / (1.0 + exponential))


def fisher_energy(pit_row: np.ndarray) -> float:
    """Exact scalar form of V483 `energy_fisher_mean`."""

    values = np.asarray(pit_row[:ENERGY_COUNT], dtype=np.float64)
    if values.shape != (ENERGY_COUNT,) or not np.isfinite(values).all():
        raise RuntimeError("V496 energy feature schema changed")
    active = values > 0.0
    count = int(np.sum(active))
    if count == 0:
        return 0.0
    clipped = np.clip(values, 0.0, 1.0)
    p_value = np.where(active, np.clip(1.0 - clipped, P_FLOOR, 1.0), 1.0)
    result = float(np.sum(np.where(active, -np.log(p_value), 0.0)) / count)
    if not math.isfinite(result):
        raise RuntimeError("V496 Fisher energy is non-finite")
    return float(np.float32(np.clip(result, 0.0, 20.0)))


@dataclass(frozen=True)
class ExactTimeCalibration:
    name: str
    sorted_values: np.ndarray
    time_offsets: np.ndarray
    maximum_time: int
    alpha: float

    def increment(self, tick: int, feature_value: float) -> float:
        tick = int(tick)
        if tick <= GATE_ZERO_END:
            return 0.0
        reference_time = min(max(tick, 0), self.maximum_time)
        first = int(self.time_offsets[reference_time])
        last = int(self.time_offsets[reference_time + 1])
        reference = self.sorted_values[first:last]
        if reference.size == 0:
            raise RuntimeError(f"V496 {self.name} exact-time reference is empty")
        value = np.float32(feature_value)
        left = int(np.searchsorted(reference, value, side="left"))
        right = int(np.searchsorted(reference, value, side="right"))
        midrank = (left + right + 1.0) / (reference.size + 1.0) - 1.0
        gate = (
            1.0
            if tick >= GATE_FULL_START
            else (float(tick) - GATE_ZERO_END) / (GATE_FULL_START - GATE_ZERO_END)
        )
        return self.alpha * gate * midrank


class SymmetricMemoryState:
    def __init__(self) -> None:
        self.early_heap: list[float] = []
        self.early_total = 0.0
        self.floor_values: list[float] = []
        self.floor: float | None = None
        self.late_heap: list[float] = []
        self.late_total = 0.0

    @staticmethod
    def _push(heap: list[float], total: float, value: float) -> tuple[float, float]:
        if len(heap) < MEMORY_TOP_COUNT:
            heapq.heappush(heap, value)
            total += value
        elif value > heap[0]:
            removed = heapq.heapreplace(heap, value)
            total += value - removed
        return total, max(total / len(heap) - value, 0.0)

    def update(self, margin: float, tick: int) -> float:
        value = float(margin)
        tick = int(tick)
        if tick < MEMORY_BOUNDARY:
            self.early_total, drawdown = self._push(
                self.early_heap, self.early_total, value
            )
            if tick >= MEMORY_FLOOR_START:
                self.floor_values.append(drawdown)
            if tick == MEMORY_BOUNDARY - 1:
                if len(self.floor_values) != MEMORY_BOUNDARY - MEMORY_FLOOR_START:
                    raise RuntimeError("V496 memory floor sample count changed")
                self.floor = float(np.median(np.asarray(self.floor_values, dtype=np.float64)))
            return value
        if self.floor is None:
            raise RuntimeError("V496 memory started before its floor froze")
        self.late_total, drawdown = self._push(self.late_heap, self.late_total, value)
        result = value + MEMORY_ALPHA * (drawdown - self.floor)
        if not math.isfinite(result):
            raise RuntimeError("V496 memory result is non-finite")
        return result


@dataclass(frozen=True)
class CorrectionResources:
    pit: ExactTimeCalibration
    fisher: ExactTimeCalibration


def _load_calibration(
    root: Path, record: Mapping[str, object], name: str, feature_index: int, alpha: float
) -> ExactTimeCalibration:
    path = root / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size_bytes"])
        or _sha256(path) != str(record["sha256"])
    ):
        raise RuntimeError(f"V496 {name} calibration resource changed")
    with np.load(path, allow_pickle=False) as archive:
        sorted_values = archive["sorted_values"].copy()
        time_offsets = archive["time_offsets"].astype(np.int64)
        reference_counts = archive["reference_counts"].astype(np.int64)
        scalars = {
            "feature_index": int(archive["feature_index"]),
            "alpha": float(archive["alpha"]),
            "gate_zero_end": int(archive["gate_zero_end"]),
            "gate_full_start": int(archive["gate_full_start"]),
        }
    if (
        sorted_values.dtype != np.float32
        or sorted_values.ndim != 1
        or time_offsets.ndim != 1
        or reference_counts.ndim != 1
        or time_offsets.size != reference_counts.size + 1
        or int(time_offsets[-1]) != sorted_values.size
        or not np.array_equal(np.diff(time_offsets), reference_counts)
        or np.any(reference_counts <= 0)
        or not np.isfinite(sorted_values).all()
        or scalars
        != {
            "feature_index": feature_index,
            "alpha": alpha,
            "gate_zero_end": GATE_ZERO_END,
            "gate_full_start": GATE_FULL_START,
        }
    ):
        raise RuntimeError(f"V496 {name} calibration schema changed")
    return ExactTimeCalibration(
        name, sorted_values, time_offsets, reference_counts.size - 1, alpha
    )


def load_resources(model_directory_path: str | Path) -> CorrectionResources:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    before = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if before != _sha256(manifest_path) or not isinstance(manifest, Mapping):
        raise RuntimeError("V496 manifest is unstable or invalid")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("contract") != EXPECTED_CONTRACT
    ):
        raise RuntimeError("V496 production identity changed")
    code_root = Path(__file__).resolve().parent
    for relative, digest in manifest.get("source_files", {}).items():
        source = (code_root / relative).resolve()
        if code_root not in source.parents or not source.is_file() or _sha256(source) != digest:
            raise RuntimeError(f"V496 source changed: {relative}")
    pit = _load_calibration(
        root,
        manifest["pit_calibration"],
        PIT_FEATURE_NAME,
        PIT_FEATURE_INDEX,
        PIT_ALPHA,
    )
    fisher = _load_calibration(
        root,
        manifest["fisher_calibration"],
        FISHER_FEATURE_NAME,
        1,
        FISHER_ALPHA,
    )
    return CorrectionResources(pit, fisher)


__all__ = [
    "EPSILON",
    "PIT_FEATURE_INDEX",
    "CorrectionResources",
    "SymmetricMemoryState",
    "fisher_energy",
    "load_resources",
    "_logit",
    "_sigmoid",
]
