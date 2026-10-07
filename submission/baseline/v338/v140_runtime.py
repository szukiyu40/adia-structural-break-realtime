"""Validated causal runtime for the frozen ADIA v140 sparse rank correction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
CANDIDATE_ID = "v140_sparse_causal_time_gate"
MANIFEST_FILENAME = "v140_manifest.json"
MODEL_FILENAME = "v140_sparse_ranker.json"
REPORT_FILENAME = "v140_fullfit_report.json"
SOURCE_PATHS = ("main.py", "v140_runtime.py")
SELECTED_INDICES = (196, 55, 159, 45, 278, 201, 175, 147)
SELECTED_NAMES = (
    "adaptive_f30_std_diff",
    "global_std",
    "adaptive_f10_tail_ratio_diff",
    "tail_ratio_post",
    "adaptive_f60_var_ratio",
    "adaptive_f30_q90_diff",
    "adaptive_f10_q99_ratio",
    "adaptive_f10_q90_diff",
)
MODEL_FEATURE_COUNT = 10
CHECKPOINTS = (255, 511, 767)
START_TICK = 256
ALPHA = 0.08
CORRECTION_CLIP = 4.0
PROBABILITY_CLIP = 1e-7
TIME_DENOMINATOR = math.log1p(768.0)
EPSILON = 1e-9


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
    model_path: Path
    center: float
    scale: float


@dataclass(frozen=True)
class LoadedResources:
    model: Any
    center: float
    scale: float


def validate_resources(model_directory_path: str | Path) -> ValidatedResources:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "status", "competition", "candidate_id", "model_file",
        "model_sha256", "model_size_bytes", "fullfit_report_file",
        "fullfit_report_sha256", "center", "scale", "selected_indices",
        "selected_names", "contract", "source_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise ValueError("v140 manifest inventory changed")
    for key, expected in (
        ("schema_version", 1), ("status", "complete"),
        ("competition", COMPETITION), ("candidate_id", CANDIDATE_ID),
        ("model_file", MODEL_FILENAME), ("fullfit_report_file", REPORT_FILENAME),
        ("selected_indices", list(SELECTED_INDICES)), ("selected_names", list(SELECTED_NAMES)),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"v140 manifest {key} changed")
    contract = {
        "model_feature_count": MODEL_FEATURE_COUNT,
        "checkpoint_ticks": list(CHECKPOINTS),
        "start_tick": START_TICK,
        "alpha": ALPHA,
        "correction_clip": CORRECTION_CLIP,
        "future_ticks_used": False,
        "final_online_length_used": False,
        "peer_series_raw_values_used": False,
        "deterministic": True,
    }
    if manifest.get("contract") != contract:
        raise ValueError("v140 contract changed")
    source_hashes = manifest.get("source_sha256")
    source_root = Path(__file__).resolve().parent
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(SOURCE_PATHS):
        raise ValueError("v140 source inventory changed")
    for name in SOURCE_PATHS:
        digest = source_hashes.get(name)
        if not _is_sha256(digest) or _sha256_file(source_root / name) != digest:
            raise RuntimeError(f"v140 source hash mismatch: {name}")
    model_path = root / MODEL_FILENAME
    if (
        not model_path.is_file()
        or not _is_sha256(manifest.get("model_sha256"))
        or _sha256_file(model_path) != manifest["model_sha256"]
        or model_path.stat().st_size != manifest.get("model_size_bytes")
    ):
        raise RuntimeError("v140 model resource changed")
    report_path = root / REPORT_FILENAME
    if (
        not report_path.is_file()
        or not _is_sha256(manifest.get("fullfit_report_sha256"))
        or _sha256_file(report_path) != manifest["fullfit_report_sha256"]
    ):
        raise RuntimeError("v140 fullfit report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("fixed_config_oof", {}).get("promotion_passed") is not True
        or report.get("contract", {}).get("training_folds") != [0, 1, 2, 3, 4]
        or report.get("artifacts", {}).get("model_sha256") != manifest["model_sha256"]
        or report.get("fullfit", {}).get("screen", {}).get("selected_features") != list(SELECTED_NAMES)
    ):
        raise RuntimeError("v140 fullfit report is not production eligible")
    center = float(manifest.get("center"))
    scale = float(manifest.get("scale"))
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("v140 calibration is invalid")
    return ValidatedResources(model_path=model_path, center=center, scale=scale)


def load_resources(model_directory_path: str | Path) -> LoadedResources:
    validated = validate_resources(model_directory_path)
    xgboost = importlib.import_module("xgboost")
    model = xgboost.XGBRanker()
    model.load_model(validated.model_path)
    if int(model.n_features_in_) != MODEL_FEATURE_COUNT:
        raise RuntimeError("v140 loaded feature count changed")
    return LoadedResources(model=model, center=validated.center, scale=validated.scale)


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


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(output)
    replacement = float(np.median(output[finite])) if finite.any() else 0.0
    output[~finite] = replacement
    return output


def _std_var(values: np.ndarray) -> tuple[float, float]:
    if values.size <= 1:
        return 0.0, 0.0
    variance = float(np.var(values, ddof=1))
    return math.sqrt(max(variance, 0.0)), max(variance, 0.0)


def _tail_ratio(values: np.ndarray) -> float:
    q05, q25, q75, q95 = np.quantile(
        values, (0.05, 0.25, 0.75, 0.95), method="linear"
    )
    return float(q95 - q05) / (float(q75 - q25) + EPSILON)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0.0 else 0.0


class CausalV140State:
    def __init__(self, historical: Sequence[float] | np.ndarray, resources: LoadedResources):
        self.history = _finite(historical)
        self.history_std, self.history_var = _std_var(self.history)
        self.history_q90, self.history_q99 = np.quantile(
            self.history, (0.90, 0.99), method="linear"
        )
        self.history_tail_ratio = _tail_ratio(self.history)
        self.online: list[float] = []
        self.resources = resources
        self.time_online = -1
        self.pending: np.ndarray | None = None
        self.correction = 0.0

    def _selected_features(self) -> np.ndarray:
        prefix = _finite(self.online)
        global_values = np.concatenate((self.history, prefix))
        global_std, _ = _std_var(global_values)
        width10 = max(1, int(math.ceil(0.10 * prefix.size)))
        width30 = max(1, int(math.ceil(0.30 * prefix.size)))
        width60 = max(1, int(math.ceil(0.60 * prefix.size)))
        tail10 = prefix[-width10:]
        tail30 = prefix[-width30:]
        tail60 = prefix[-width60:]
        tail30_std, _ = _std_var(tail30)
        _, tail60_var = _std_var(tail60)
        tail30_q90 = float(np.quantile(tail30, 0.90, method="linear"))
        tail10_q90, tail10_q99 = np.quantile(
            tail10, (0.90, 0.99), method="linear"
        )
        return np.asarray(
            (
                tail30_std - self.history_std,
                global_std,
                _tail_ratio(tail10) - self.history_tail_ratio,
                _tail_ratio(prefix),
                _safe_ratio(tail60_var, self.history_var),
                tail30_q90 - float(self.history_q90),
                _safe_ratio(float(tail10_q99), float(self.history_q99)),
                float(tail10_q90) - float(self.history_q90),
            ),
            dtype=np.float32,
        )

    def update(self, value: float) -> None:
        self.time_online += 1
        self.online.append(float(value))
        self.pending = None
        if self.time_online in CHECKPOINTS:
            self.pending = self._selected_features()

    def apply(self, v35_probability: float) -> float:
        base_logit = _logit(v35_probability)
        if self.pending is not None:
            model_row = np.r_[
                self.pending,
                np.float32(base_logit),
                np.float32(math.log1p(self.time_online) / TIME_DENOMINATOR),
            ].astype(np.float32, copy=False)
            raw = np.asarray(
                self.resources.model.predict(model_row.reshape(1, -1)), dtype=np.float64
            )
            if raw.shape != (1,) or not math.isfinite(float(raw[0])):
                raise RuntimeError("v140 ranker produced an invalid correction")
            standardized = (float(raw[0]) - self.resources.center) / self.resources.scale
            self.correction = min(max(standardized, -CORRECTION_CLIP), CORRECTION_CLIP)
        score = base_logit
        if self.time_online >= START_TICK:
            score += ALPHA * self.correction
        return _sigmoid(score)
