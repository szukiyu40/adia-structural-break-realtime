"""Deterministic causal runtime for the fixed corrected-local V147 layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
CANDIDATE_ID = "v147_fixed_correct_local_memory"
MANIFEST_FILENAME = "v147_manifest.json"
AUDIT_FILENAME = "v147_independent_audit.json"
FULLFIT_FILENAME = "v147_fullfit_report.json"
SOURCE_PATHS = ("main.py", "v147_runtime.py")
BLOCK = 64
MAD_SCALE = 1.4826
MEMORY_START_TICK = 128
MEMORY_MINIMUM = 1.0
ALPHA = 0.16
HISTORY_DISTANCE_Q50 = 1.6062169671058655
HISTORY_DISTANCE_Q95 = 3.6169520378112776
PROBABILITY_CLIP = 1e-7


def _sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class ValidatedResources:
    history_distance_q50: float
    history_distance_q95: float


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
        "audit_file",
        "audit_sha256",
        "fullfit_report_file",
        "fullfit_report_sha256",
        "contract",
        "source_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("v147 manifest inventory changed")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("audit_file", AUDIT_FILENAME),
        ("fullfit_report_file", FULLFIT_FILENAME),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"v147 manifest {key} changed")
    contract = {
        "block_size": BLOCK,
        "memory_start_tick": MEMORY_START_TICK,
        "memory_minimum": MEMORY_MINIMUM,
        "alpha": ALPHA,
        "history_distance_q50": HISTORY_DISTANCE_Q50,
        "history_distance_q95": HISTORY_DISTANCE_Q95,
        "corrected_immutable_block_state": True,
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_series_raw_values_used": False,
        "deterministic": True,
    }
    if manifest.get("contract") != contract:
        raise ValueError("v147 contract changed")
    source_hashes = manifest.get("source_sha256")
    source_root = Path(__file__).resolve().parent
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(SOURCE_PATHS):
        raise ValueError("v147 source inventory changed")
    for name in SOURCE_PATHS:
        digest = source_hashes.get(name)
        if not _is_sha256(digest) or _sha256_file(source_root / name) != digest:
            raise RuntimeError(f"v147 source hash mismatch: {name}")
    audit_path = root / AUDIT_FILENAME
    if (
        not audit_path.is_file()
        or not _is_sha256(manifest.get("audit_sha256"))
        or _sha256_file(audit_path) != manifest["audit_sha256"]
    ):
        raise RuntimeError("v147 audit resource changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("checks", {}).get("deterministic_exact_30_percent") is not True
        or audit.get("checks", {}).get("future_ticks_used") is not False
        or audit.get("checks", {}).get("peer_series_current_values_used") is not False
        or float(audit.get("checks", {}).get("score_max_abs_difference", math.inf)) > 1e-12
    ):
        raise RuntimeError("v147 independent audit is not production eligible")
    fullfit_path = root / FULLFIT_FILENAME
    if (
        not fullfit_path.is_file()
        or not _is_sha256(manifest.get("fullfit_report_sha256"))
        or _sha256_file(fullfit_path) != manifest["fullfit_report_sha256"]
    ):
        raise RuntimeError("v147 fullfit resource changed")
    fullfit = json.loads(fullfit_path.read_text(encoding="utf-8"))
    if (
        fullfit.get("status") != "complete"
        or fullfit.get("promotion_passed") is not True
        or fullfit.get("fixed_config", {}).get("memory_minimum") != MEMORY_MINIMUM
        or fullfit.get("fixed_config", {}).get("alpha") != ALPHA
        or fullfit.get("fullfit_thresholds", {}).get("q50") != HISTORY_DISTANCE_Q50
        or fullfit.get("fullfit_thresholds", {}).get("q95") != HISTORY_DISTANCE_Q95
    ):
        raise RuntimeError("v147 fullfit report is not production eligible")
    return ValidatedResources(
        history_distance_q50=HISTORY_DISTANCE_Q50,
        history_distance_q95=HISTORY_DISTANCE_Q95,
    )


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(output)
    replacement = float(np.median(output[finite])) if finite.any() else 0.0
    output[~finite] = replacement
    return output


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)
    return math.log(clipped) - math.log1p(-clipped)


def _sigmoid(score: float) -> float:
    if score >= 0.0:
        value = 1.0 / (1.0 + math.exp(-score))
    else:
        exponential = math.exp(score)
        value = exponential / (1.0 + exponential)
    return min(max(value, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)


def _states(blocks: np.ndarray) -> np.ndarray:
    source = np.array(blocks, dtype=np.float64, copy=True)
    mean = source.mean(axis=1)
    log_std = np.log(np.maximum(source.std(axis=1, ddof=1), 1e-12))
    left = source[:, :-1] - source[:, :-1].mean(axis=1, keepdims=True)
    right = source[:, 1:] - source[:, 1:].mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    lag1 = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.zeros(source.shape[0], dtype=np.float64),
        where=denominator > 1e-12,
    )
    tail = np.mean(np.abs(source) > 2.0, axis=1)
    return np.column_stack((mean, log_std, lag1, tail))


def _history_reference(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = history.size // BLOCK
    if count < 2:
        raise RuntimeError("v147 requires at least two complete history blocks")
    states = _states(history[-count * BLOCK :].reshape(count, BLOCK))
    center = np.median(states, axis=0)
    robust = MAD_SCALE * np.median(np.abs(states - center), axis=0)
    floors = np.asarray(
        (
            1.0 / math.sqrt(BLOCK),
            1.0 / math.sqrt(2.0 * (BLOCK - 1)),
            1.0 / math.sqrt(BLOCK),
            math.sqrt(0.05 * 0.95 / BLOCK),
        ),
        dtype=np.float64,
    )
    return center, np.maximum(robust, floors)


class CausalV147State:
    def __init__(self, historical: Sequence[float] | np.ndarray, resources: ValidatedResources):
        history = _finite(historical)
        self.center, self.scale = _history_reference(history)
        self.resources = resources
        self.online_block: list[float] = []
        self.previous_distance: float | None = None
        self.local_support = 0.0
        self.top3_logits: list[float] = []
        self.time_online = -1

    def update(self, value: float) -> None:
        self.time_online += 1
        self.online_block.append(float(value))
        if len(self.online_block) == BLOCK:
            state = _states(np.asarray(self.online_block, dtype=np.float64).reshape(1, BLOCK))[0]
            distance = float(np.linalg.norm((state - self.center) / self.scale))
            recent_two = distance if self.previous_distance is None else min(self.previous_distance, distance)
            lower = self.resources.history_distance_q50
            upper = self.resources.history_distance_q95
            self.local_support = min(max((recent_two - lower) / (upper - lower), 0.0), 1.0)
            self.previous_distance = distance
            self.online_block.clear()

    def apply(self, v35_probability: float, v140_probability: float) -> float:
        if self.time_online < 0:
            raise RuntimeError("v147 apply called before update")
        memory = 0.0
        if self.time_online >= MEMORY_START_TICK:
            current = _logit(v35_probability)
            self.top3_logits.append(current)
            self.top3_logits.sort(reverse=True)
            if len(self.top3_logits) > 3:
                self.top3_logits.pop()
            memory = max(float(np.mean(self.top3_logits)) - current, 0.0)
        score = _logit(v140_probability)
        if memory >= MEMORY_MINIMUM:
            score += ALPHA * self.local_support * memory
        return _sigmoid(score)
