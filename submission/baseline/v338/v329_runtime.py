"""Causal period/LAG correction runtime for ADIA V329."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import v329_feature_core as core


MANIFEST_FILENAME = "v329_manifest.json"
CALIBRATION_FILENAME = "v329_fullfit_period_lag_calibration.npz"
CANDIDATE_ID = "v329_fixed_shrunk_period_lag"
COMPETITION = "ADIA Lab x CrunchDAO Structural Break Challenge 2026"
STABLE_ALPHA = 0.04
WEAK_ALPHA = 0.12
START_TICK = 31
CADENCE = 8
STABLE_FEATURES = (
    "acf_vector_shift_64",
    "acf_vector_shift_128",
    "adaptive_period_acf_loss",
)
WEAK_FEATURES = (
    "multilag_point_mean",
    "multilag_point_top2",
    "lag_energy_shift_4",
    "lag_energy_shift_64",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class V329Resources:
    stable_threshold: float
    weak_threshold: float
    cadence_times: np.ndarray
    stable_references: Mapping[int, tuple[np.ndarray, ...]]
    weak_references: Mapping[int, tuple[np.ndarray, ...]]


def _load_calibration(path: Path) -> V329Resources:
    with np.load(path, allow_pickle=False) as archive:
        cadence_times = archive["cadence_times"].astype(np.int32)
        stable_names = tuple(str(value) for value in archive["stable_feature_names"])
        weak_names = tuple(str(value) for value in archive["weak_feature_names"])
        if stable_names != STABLE_FEATURES or weak_names != WEAK_FEATURES:
            raise ValueError("V329 calibration feature inventory changed")
        stable: dict[int, tuple[np.ndarray, ...]] = {}
        weak: dict[int, tuple[np.ndarray, ...]] = {}
        for tick_value in cadence_times:
            tick = int(tick_value)
            stable[tick] = tuple(
                archive[f"s_{tick}_{index}"].astype(np.float64)
                for index in range(len(STABLE_FEATURES))
            )
            weak[tick] = tuple(
                archive[f"w_{tick}_{index}"].astype(np.float64)
                for index in range(len(WEAK_FEATURES))
            )
        resources = V329Resources(
            stable_threshold=float(archive["stable_threshold"][0]),
            weak_threshold=float(archive["weak_threshold"][0]),
            cadence_times=cadence_times,
            stable_references=stable,
            weak_references=weak,
        )
    if resources.cadence_times.ndim != 1 or resources.cadence_times.size == 0:
        raise ValueError("V329 cadence inventory changed")
    if np.any(np.diff(resources.cadence_times) != CADENCE) or int(resources.cadence_times[0]) != START_TICK:
        raise ValueError("V329 calibration cadence changed")
    for collection in (resources.stable_references, resources.weak_references):
        for references in collection.values():
            for values in references:
                if values.size == 0 or not np.isfinite(values).all() or np.any(values[1:] < values[:-1]):
                    raise ValueError("V329 ECDF reference changed")
    return resources


def validate_resources(model_directory_path: str | Path) -> Mapping[str, Any]:
    root = Path(model_directory_path).expanduser().resolve()
    manifest_path = root / MANIFEST_FILENAME
    calibration_path = root / CALIBRATION_FILENAME
    if not manifest_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else calibration_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in (("schema_version", 1), ("status", "complete"), ("competition", COMPETITION), ("candidate_id", CANDIDATE_ID)):
        if manifest.get(key) != expected:
            raise ValueError(f"V329 manifest {key} changed")
    if manifest.get("calibration_sha256") != _sha(calibration_path):
        raise RuntimeError("V329 calibration changed")
    source = manifest.get("source_sha256")
    expected_names = {"main.py", "v329_runtime.py", "v329_feature_core.py"}
    if not isinstance(source, Mapping) or set(source) != expected_names:
        raise ValueError("V329 source inventory changed")
    code_root = Path(__file__).resolve().parent
    for name, digest in source.items():
        if not _is_sha256(digest) or _sha(code_root / name) != digest:
            raise RuntimeError(f"V329 source changed: {name}")
    return manifest


def load_resources(model_directory_path: str | Path) -> V329Resources:
    root = Path(model_directory_path).expanduser().resolve()
    validate_resources(root)
    return _load_calibration(root / CALIBRATION_FILENAME)


def _mid_ecdf(reference: np.ndarray, value: float) -> float:
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    return (left + right) / (2.0 * reference.size) - 0.5


class CausalPeriodLagState:
    """Same-series-only V329 state with cadence-8 feature updates."""

    __slots__ = (
        "center", "correction", "history_size", "history_z", "online_z",
        "resources", "scale", "stable_gate", "static", "tick", "weak_gate",
    )

    def __init__(self, history: Any, resources: V329Resources) -> None:
        history_clean = core._clean(np.asarray(history, dtype=np.float64))  # noqa: SLF001
        center, scale = core._robust_location_scale(history_clean)  # noqa: SLF001
        history_z = np.clip((history_clean - center) / scale, -core.CLIP, core.CLIP)
        static = core._history_resources(history_z)  # noqa: SLF001
        self.center = float(center)
        self.scale = float(scale)
        self.history_z = history_z
        self.history_size = int(history_z.size)
        self.static = static
        self.resources = resources
        self.stable_gate = float(static["period_stability"]) >= resources.stable_threshold
        self.weak_gate = float(static["period_strength"]) <= resources.weak_threshold
        self.online_z: list[float] = []
        self.tick = -1
        self.correction = 0.0

    def _features(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        values = np.concatenate((self.history_z, np.asarray(self.online_z, dtype=np.float64)))
        lag_null = self.static["lag_null"]
        point_scores = []
        for lag in core.FIXED_LAGS:
            lag_center, lag_scale, _ = lag_null[lag]
            raw = abs((values[-1] - values[-1 - lag] - lag_center) / lag_scale)
            point_scores.append(float(np.log1p(min(raw, 30.0))))
        ordered = np.sort(np.asarray(point_scores, dtype=np.float64))
        periodic = core._period_features(values, self.history_size, self.static)  # noqa: SLF001
        stable = (
            core._acf_vector_shift(values, self.static, 64),  # noqa: SLF001
            core._acf_vector_shift(values, self.static, 128),  # noqa: SLF001
            float(periodic[4]),
        )
        weak = (
            float(np.mean(ordered)),
            float(np.mean(ordered[-2:])),
            core._lag_energy_shift(values, 4, float(lag_null[4][2])),  # noqa: SLF001
            core._lag_energy_shift(values, 64, float(lag_null[64][2])),  # noqa: SLF001
        )
        return stable, weak

    def update(self, point: float, v317_probability: float) -> float:
        self.tick += 1
        value = float(point)
        if not math.isfinite(value):
            value = self.center
        self.online_z.append(float(np.clip((value - self.center) / self.scale, -core.CLIP, core.CLIP)))
        if self.tick >= START_TICK and (self.tick - START_TICK) % CADENCE == 0:
            calibration_tick = int(self.resources.cadence_times[min(
                np.searchsorted(self.resources.cadence_times, self.tick, side="right") - 1,
                self.resources.cadence_times.size - 1,
            )])
            stable_values, weak_values = self._features()
            correction = 0.0
            if self.stable_gate:
                correction += STABLE_ALPHA * float(np.mean([
                    _mid_ecdf(reference, value)
                    for reference, value in zip(self.resources.stable_references[calibration_tick], stable_values, strict=True)
                ]))
            if self.weak_gate:
                time_weight = float(np.clip((self.tick - 63.0) / 448.0, 0.0, 1.0))
                correction += WEAK_ALPHA * time_weight * float(np.mean([
                    _mid_ecdf(reference, value)
                    for reference, value in zip(self.resources.weak_references[calibration_tick], weak_values, strict=True)
                ]))
            self.correction = correction
        probability = min(max(float(v317_probability), 1e-7), 1.0 - 1e-7)
        raw = math.log(probability) - math.log1p(-probability) + self.correction
        return 1.0 / (1.0 + math.exp(-raw))


__all__ = [
    "CADENCE", "CANDIDATE_ID", "CausalPeriodLagState", "START_TICK",
    "STABLE_ALPHA", "V329Resources", "WEAK_ALPHA", "load_resources",
    "validate_resources",
]
