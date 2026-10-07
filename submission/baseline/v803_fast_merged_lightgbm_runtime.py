"""Private exact FastConfig transport for V791's two merged boosters."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import v791_tree_transport_runtime as standard


_INSTALLED_FOR: Path | None = None


class FastMultiPredictor:
    def __init__(self, booster: Any, width: int, outputs: int, raw_score: bool) -> None:
        from lightgbm.basic import _LIB, _safe_call

        self.booster = booster
        self.width = int(width)
        self.outputs = int(outputs)
        self.lib = _LIB
        self.safe_call = _safe_call
        self.handle = ctypes.c_void_p()
        self.out_len = ctypes.c_int64()
        self.output = np.empty(self.outputs, dtype=np.float64)
        self.output_pointer = self.output.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        _safe_call(
            _LIB.LGBM_BoosterPredictForMatSingleRowFastInit(
                booster._handle,  # noqa: SLF001
                ctypes.c_int(1 if raw_score else 0),
                ctypes.c_int(0),
                ctypes.c_int(-1),
                ctypes.c_int(0),
                ctypes.c_int32(self.width),
                ctypes.c_char_p(b"num_threads=1"),
                ctypes.byref(self.handle),
            )
        )
        if not self.handle:
            raise RuntimeError("V803 empty LightGBM FastConfig")

    def predict(self, row: np.ndarray) -> np.ndarray:
        values = np.asarray(row)
        if values.shape != (self.width,) or values.dtype != np.dtype("float32"):
            raise RuntimeError("V803 fast multi-output row changed")
        if not np.isfinite(values).all():
            raise RuntimeError("V803 fast multi-output row is non-finite")
        if not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)
        self.out_len.value = 0
        self.safe_call(
            self.lib.LGBM_BoosterPredictForMatSingleRowFast(
                self.handle,
                ctypes.c_void_p(values.ctypes.data),
                ctypes.byref(self.out_len),
                self.output_pointer,
            )
        )
        if self.out_len.value != self.outputs or not np.isfinite(self.output).all():
            raise RuntimeError("V803 fast multi-output prediction failed")
        return self.output


class _DetectorCoordinator:
    def __init__(self, predictor: FastMultiPredictor) -> None:
        self.predictor = predictor
        self.cached = np.empty(5, dtype=np.float64)
        self.expected_class = 0
        self.row_identity: int | None = None

    def predict(self, class_index: int, data: np.ndarray, **kwargs) -> np.ndarray:
        if (
            set(kwargs) != {"raw_score", "num_threads"}
            or kwargs["raw_score"] is not False
            or int(kwargs["num_threads"]) != 1
        ):
            raise RuntimeError("V803 detector call signature changed")
        values = np.asarray(data)
        if values.shape != (1, 422) or values.dtype != np.float32:
            raise RuntimeError("V803 detector matrix changed")
        if class_index == 0:
            if self.expected_class != 0:
                raise RuntimeError("V803 detector sequence restarted early")
            self.cached[:] = self.predictor.predict(values[0])
            self.row_identity = id(data)
            self.expected_class = 1
        else:
            if class_index != self.expected_class or self.row_identity != id(data):
                raise RuntimeError("V803 detector source order changed")
            self.expected_class += 1
        result = np.asarray((self.cached[class_index],), dtype=np.float64)
        if class_index == 4:
            self.expected_class = 0
            self.row_identity = None
        return result


class _DetectorProxy:
    def __init__(self, coordinator: _DetectorCoordinator, class_index: int) -> None:
        self.coordinator = coordinator
        self.class_index = int(class_index)

    def predict(self, data, **kwargs):
        return self.coordinator.predict(self.class_index, data, **kwargs)

    @staticmethod
    def num_feature() -> int:
        return 422

    @staticmethod
    def num_trees() -> int:
        return 500


def install(v787_parent_main: Any, model_directory_path: str | Path) -> None:
    global _INSTALLED_FOR
    root = standard.validate_resources(model_directory_path)
    if _INSTALLED_FOR is not None:
        if _INSTALLED_FOR != root:
            raise RuntimeError("V803 installed for a different resource root")
        return
    import lightgbm as lgb

    pair_booster = lgb.Booster(model_file=str(root / standard.PAIR_NAME))
    detector_booster = lgb.Booster(model_file=str(root / standard.DETECTOR_NAME))
    if (
        pair_booster.num_feature() != 440
        or pair_booster.num_trees() != 900
        or pair_booster.current_iteration() != 300
        or detector_booster.num_feature() != 422
        or detector_booster.num_trees() != 2500
        or detector_booster.current_iteration() != 500
    ):
        raise RuntimeError("V803 merged model structure changed")
    pair_predictor = FastMultiPredictor(pair_booster, 440, 3, True)
    detector_predictor = FastMultiPredictor(detector_booster, 422, 5, False)

    base_runtime = v787_parent_main.v338_main.base_runtime
    original_runtime = base_runtime.original

    def load_base_resources(directory: str | Path, *, check_runtime: bool = True):
        validated = original_runtime.validate_resources(
            directory, check_runtime=check_runtime
        )
        meta = base_runtime._validate_v738(Path(directory).expanduser().resolve())  # noqa: SLF001
        seeds = tuple(
            original_runtime.LoadedPairSeed(seed.seed, None, seed.calibration)
            for seed in validated.pairrank_seeds
        )
        return base_runtime.LoadedResources(
            pairrank_seeds=seeds,
            extratrees_model=validated.extratrees_model,
            meta=meta,
        )

    def predict_base_probabilities(resources: Any, rows: Mapping[str, np.ndarray]):
        if set(rows) != {"v4_seq18"}:
            raise RuntimeError("V803 feature-view inventory changed")
        row = np.asarray(rows["v4_seq18"], dtype=np.float32)
        if row.shape != (440,) or not np.isfinite(row).all():
            raise RuntimeError("V803 PairRank row changed")
        raw = pair_predictor.predict(row)
        pair_total = 0.0
        for index, seed in enumerate(resources.pairrank_seeds):
            pair_total += seed.calibration.apply(float(raw[index]))
        pairrank = pair_total / len(resources.pairrank_seeds)
        extra_trees = float(resources.extratrees_model.predict_one(row))
        result = (pairrank, extra_trees)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result):
            raise RuntimeError("V803 base output is invalid")
        return result

    base_runtime.load_resources = load_base_resources
    base_runtime.predict_base_probabilities = predict_base_probabilities

    coordinator = _DetectorCoordinator(detector_predictor)
    proxies = tuple(_DetectorProxy(coordinator, index) for index in range(5))
    v338_main = v787_parent_main.v338_main
    v256_module = __import__(v338_main.load_v256_resources.__module__)
    v335_module = __import__(v338_main.load_v335_resources.__module__)
    validate_v256 = v338_main.validate_v256_resources
    validate_v335 = v338_main.validate_v335_resources

    def load_v256(directory: str | Path):
        validated = validate_v256(directory)
        return v256_module.LoadedResources(
            boosters=proxies[:3], feature_count=validated.feature_count
        )

    def load_v335(directory: str | Path):
        validated = validate_v335(directory)
        return v335_module.LoadedResources(
            boosters=proxies[3:], feature_count=validated.feature_count
        )

    v338_main.load_v256_resources = load_v256
    v338_main.load_v335_resources = load_v335
    _INSTALLED_FOR = root


__all__ = ["FastMultiPredictor", "install"]

