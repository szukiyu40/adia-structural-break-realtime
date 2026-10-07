"""Authenticated V705 ExtraTrees translation with a private fast-row handle."""

from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path

import numpy as np


FEATURE_COUNT = 440
TREE_COUNT = 400
RESOURCE_NAME = "v705_extratrees_transport.txt"
RESOURCE_SHA256 = "b153436143ceb9cab8e73cbb68e4d3d6397562d107016ed34a4f801c00f67964"
RESOURCE_SIZE = 50_722_176


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class NativeExtraTreesFastPredictor:
    def __init__(self, booster) -> None:
        from lightgbm.basic import _LIB, _safe_call

        # The C fast-config handle borrows the booster; retain its owner for
        # exactly as long as this predictor can be called.
        self.booster = booster
        self.lib = _LIB
        self.safe_call = _safe_call
        self.handle = ctypes.c_void_p()
        self.out_len = ctypes.c_int64()
        self.output = np.empty(1, dtype=np.float64)
        self.output_pointer = self.output.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        _safe_call(
            _LIB.LGBM_BoosterPredictForMatSingleRowFastInit(
                booster._handle,  # noqa: SLF001
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(-1),
                ctypes.c_int(0),
                ctypes.c_int32(FEATURE_COUNT),
                ctypes.c_char_p(b"num_threads=1"),
                ctypes.byref(self.handle),
            )
        )
        if not self.handle:
            raise RuntimeError("V795 empty LightGBM FastConfig")

    def predict_one(self, features) -> float:
        row = np.asarray(features)
        if row.shape != (FEATURE_COUNT,) or row.dtype != np.dtype("float32"):
            raise RuntimeError("V795 native ExtraTrees row changed")
        if not np.isfinite(row).all():
            raise RuntimeError("V795 native ExtraTrees row is non-finite")
        values = row if row.flags.c_contiguous else np.ascontiguousarray(row)
        self.out_len.value = 0
        self.safe_call(
            self.lib.LGBM_BoosterPredictForMatSingleRowFast(
                self.handle,
                ctypes.c_void_p(values.ctypes.data),
                ctypes.byref(self.out_len),
                self.output_pointer,
            )
        )
        if self.out_len.value != 1:
            raise RuntimeError("V795 native ExtraTrees output changed")
        probability = float(self.output[0])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("V795 native ExtraTrees probability is invalid")
        return probability


def load(model_directory_path: str | Path) -> NativeExtraTreesFastPredictor:
    import lightgbm as lgb

    root = Path(model_directory_path).expanduser().resolve()
    path = root / RESOURCE_NAME
    if (
        not path.is_file()
        or path.stat().st_size != RESOURCE_SIZE
        or _sha(path) != RESOURCE_SHA256
    ):
        raise RuntimeError("V795 native ExtraTrees resource changed")
    booster = lgb.Booster(model_file=str(path))
    if booster.num_feature() != FEATURE_COUNT or booster.num_trees() != TREE_COUNT:
        raise RuntimeError("V795 native ExtraTrees structure changed")
    return NativeExtraTreesFastPredictor(booster)


__all__ = ["NativeExtraTreesFastPredictor", "load"]
