"""Full-five-fold V4 model under public equal-series/log-tick training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import v4_hybrid  # noqa: E402
from scripts import run_v138_causal303_nested_rank_correction as v138  # noqa: E402
from scripts.cv_v3_cached import _pair_weight, _score_arrays  # noqa: E402
from scripts.v3_cache_common import (  # noqa: E402
    load_feature_cache,
    sha256_file,
    target_and_age,
    v4_feature_config,
)


FOLDS = (0, 1, 2, 3, 4)
SAMPLES_PER_SERIES = 40
ALPHAS = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16)
PRIMARY_ALPHA = 0.04
PREREGISTRATION = ROOT / "reports/v255_equal_series_logtick_v4_preregistration.md"
CACHE = ROOT / "cache/v4_full"
V152 = ROOT / "cache/v152_bayesian_ar16_complete_nested_predictions.npz"
OUTPUT = ROOT / "reports/v255_equal_series_logtick_v4_evaluation.json"
PREDICTIONS = ROOT / "cache/v255_equal_series_logtick_v4_predictions.npz"
PUBLIC_COMMIT = "bcdcf16b8a32566bb33124de8ad844bb5548c6fe"

MODEL_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 0,
    "bagging_seed": 0,
    "feature_fraction_seed": 0,
    "data_random_seed": 0,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 8,
    "verbosity": -1,
}


def _sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _log_steps(length: int, count: int = SAMPLES_PER_SERIES) -> np.ndarray:
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.expm1(
        np.linspace(0.0, np.log1p(length - 1), count)
    )).astype(np.int64))


def _training_rows(metadata: Any, outer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    selected_counts = []
    for series_index, dataset_id in enumerate(metadata.series_ids):
        if int(dataset_id) % 5 == outer:
            continue
        length = int(metadata.lengths[series_index])
        tau = int(metadata.taus[series_index])
        times = _log_steps(length)
        count = int(times.size)
        rows.append(int(metadata.offsets[series_index]) + times)
        labels.append(((tau >= 0) & (times >= tau)).astype(np.uint8))
        weights.append(np.full(count, 1.0 / count, dtype=np.float64))
        selected_counts.append(count)
    row_index = np.concatenate(rows).astype(np.int64, copy=False)
    target = np.concatenate(labels).astype(np.uint8, copy=False)
    sample_weight = np.concatenate(weights)
    sample_weight /= float(sample_weight.mean())
    if np.any(np.diff(row_index) <= 0) or np.unique(target).size != 2:
        raise RuntimeError("V255 training sample is invalid")
    per_series = np.add.reduceat(sample_weight, np.r_[0, np.cumsum(selected_counts)[:-1]])
    if float(np.max(per_series) - np.min(per_series)) > 1e-10:
        raise RuntimeError("V255 series totals are unequal")
    audit = {
        "series": len(selected_counts),
        "rows": int(row_index.size),
        "minimum_rows_per_series": int(min(selected_counts)),
        "maximum_rows_per_series": int(max(selected_counts)),
        "positive_rate": float(target.mean()),
        "series_weight_range": float(np.max(per_series) - np.min(per_series)),
    }
    return row_index, target, sample_weight.astype(np.float32), audit


def _validation_identity(metadata: Any, outer: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    series_indices = np.flatnonzero(np.mod(metadata.series_ids, 5) == outer)
    rows = []
    identifiers = []
    times = []
    targets = []
    for series_index in series_indices:
        length = int(metadata.lengths[series_index])
        start = int(metadata.offsets[series_index])
        dataset_id = int(metadata.series_ids[series_index])
        target, _ = target_and_age(length, int(metadata.taus[series_index]))
        rows.append(np.arange(start, start + length, dtype=np.int64))
        identifiers.append(np.full(length, dataset_id, dtype=np.int64))
        times.append(np.arange(length, dtype=np.int32))
        targets.append(target)
    return np.concatenate(rows), {
        "id": np.concatenate(identifiers),
        "time": np.concatenate(times),
        "target": np.concatenate(targets),
    }


def _predict_batches(model: lgb.LGBMClassifier, features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    result = np.empty(rows.size, dtype=np.float64)
    batch = 65_536
    for first in range(0, rows.size, batch):
        last = min(first + batch, rows.size)
        matrix = np.asarray(features[rows[first:last]], dtype=np.float32)
        result[first:last] = model.predict_proba(matrix)[:, 1]
    return np.clip(result, 1e-7, 1.0 - 1e-7)


def _fold_score(base: dict[str, np.ndarray], fold: int, values: np.ndarray) -> tuple[float, int]:
    selected = base["fold"] == fold
    return (
        float(_score_arrays(base["time"][selected], base["target"][selected], values[selected])),
        int(_pair_weight(base["time"][selected], base["target"][selected])),
    )


def run(*, overwrite: bool = False) -> dict[str, Any]:
    for path in (PREREGISTRATION, V152, CACHE / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not overwrite and (OUTPUT.exists() or PREDICTIONS.exists()):
        raise FileExistsError("V255 output exists; pass --overwrite")
    started = time.perf_counter()
    cache = load_feature_cache(
        CACHE,
        candidate_path=Path(v4_hybrid.__file__).resolve(),
        candidate_config=v4_feature_config(v4_hybrid),
        expected_feature_count=v4_hybrid.FEATURE_COUNT,
    )
    base = v138._base_arrays()  # noqa: SLF001
    with np.load(V152, allow_pickle=False) as archive:
        v152_score = archive["candidate_score"].astype(np.float64)
        for key, expected in (("fold", base["fold"]), ("id", base["id"]), ("time_online", base["time"]), ("target", base["target"])):
            if not np.array_equal(archive[key], expected):
                raise RuntimeError(f"V255 V152 alignment failed: {key}")
    standalone = np.empty(base["id"].size, dtype=np.float64)
    fold_reports = []
    for outer in FOLDS:
        fold_started = time.perf_counter()
        train_rows, train_target, train_weight, sample_audit = _training_rows(cache.metadata, outer)
        training_matrix = np.asarray(cache.features[train_rows], dtype=np.float32)
        model = lgb.LGBMClassifier(**MODEL_PARAMETERS)
        fit_started = time.perf_counter()
        model.fit(training_matrix, train_target, sample_weight=train_weight)
        fit_seconds = time.perf_counter() - fit_started
        validation_rows, identity = _validation_identity(cache.metadata, outer)
        prediction = _predict_batches(model, cache.features, validation_rows)
        selected = base["fold"] == outer
        if not (
            np.array_equal(base["id"][selected], identity["id"])
            and np.array_equal(base["time"][selected], identity["time"])
            and np.array_equal(base["target"][selected], identity["target"])
        ):
            raise RuntimeError(f"V255 validation alignment failed: fold {outer}")
        standalone[selected] = prediction
        score = float(_score_arrays(identity["time"], identity["target"], prediction))
        booster_text = model.booster_.model_to_string().encode("utf-8")
        fold_reports.append({
            "outer_fold": outer,
            "train_folds": [fold for fold in FOLDS if fold != outer],
            "sample": sample_audit,
            "standalone_ts_auc": score,
            "booster_sha256": hashlib.sha256(booster_text).hexdigest(),
            "fit_seconds": fit_seconds,
            "fold_seconds": time.perf_counter() - fold_started,
        })
        print(f"V255 outer={outer} standalone={score:.9f} fit={fit_seconds:.1f}s", flush=True)
        del model, training_matrix, train_rows, train_target, train_weight
        gc.collect()

    pair_masses = [_fold_score(base, fold, standalone)[1] for fold in FOLDS]
    standalone_folds = [_fold_score(base, fold, standalone)[0] for fold in FOLDS]
    standalone_score = float(np.average(standalone_folds, weights=pair_masses))
    v152_folds = [_fold_score(base, fold, v152_score)[0] for fold in FOLDS]
    v152_total = float(np.average(v152_folds, weights=pair_masses))
    correction = 2.0 * standalone - 1.0
    diagnostic_grid = []
    for alpha in ALPHAS:
        grid_prediction = v152_score + alpha * correction
        grid_folds = [_fold_score(base, fold, grid_prediction)[0] for fold in FOLDS]
        grid_deltas = [grid_folds[fold] - v152_folds[fold] for fold in FOLDS]
        diagnostic_grid.append({
            "alpha": alpha,
            "pair_weighted_ts_auc": float(np.average(grid_folds, weights=pair_masses)),
            "delta_over_v152": float(np.average(grid_deltas, weights=pair_masses)),
            "positive_folds": int(sum(delta > 0.0 for delta in grid_deltas)),
            "minimum_fold_delta": float(min(grid_deltas)),
            "folds": grid_folds,
            "fold_deltas": grid_deltas,
            "promotion_evidence": alpha == PRIMARY_ALPHA,
        })
    blend = v152_score + PRIMARY_ALPHA * correction

    blend_folds = [_fold_score(base, fold, blend)[0] for fold in FOLDS]
    blend_score = float(np.average(blend_folds, weights=pair_masses))
    deltas = [blend_folds[fold] - v152_folds[fold] for fold in FOLDS]
    v35_report = v138._baseline_report(base)  # noqa: SLF001
    v35_folds = {int(row["fold"]): float(row["ts_auc"]) for row in v35_report["folds"]}
    every_above_v35 = all(blend_folds[fold] > v35_folds[fold] for fold in FOLDS)
    promotion = bool(
        blend_score - v152_total >= 0.00010
        and sum(delta > 0.0 for delta in deltas) >= 4
        and min(deltas) >= -0.00005
        and every_above_v35
    )
    np.savez_compressed(
        PREDICTIONS,
        standalone_probability=standalone,
        candidate_score=blend,
        v152_score=v152_score,
        fold=base["fold"], id=base["id"], time_online=base["time"], target=base["target"],
    )
    report = {
        "schema_version": "adia-v255-equal-series-logtick-v4-full5-v1",
        "competition": "ADIA Lab x CrunchDAO Structural Break Challenge 2026",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "external_prior": {
            "repository": "https://github.com/mq1234567/ADIA_Lab_Structural_Break_Challenge",
            "commit": PUBLIC_COMMIT,
            "used_for": "fixed row sampler, weighting, and model profile only",
        },
        "contract": {
            "whole_series_outer_folds": True,
            "blend_strength_selected_from_v255_scores": False,
            "primary_blend_alpha": PRIMARY_ALPHA,
            "future_ticks_used_as_features": False,
            "final_online_length_used_as_feature": False,
            "peer_test_series_used": False,
            "deterministic": True,
        },
        "model_parameters": MODEL_PARAMETERS,
        "standalone": {
            "pair_weighted_ts_auc": standalone_score,
            "folds": standalone_folds,
            "prediction_sha256": _sha(standalone),
        },
        "baseline_v152": {"pair_weighted_ts_auc": v152_total, "folds": v152_folds},
        "candidate": {
            "pair_weighted_ts_auc": blend_score,
            "delta_over_v152": blend_score - v152_total,
            "folds": blend_folds,
            "fold_deltas": deltas,
            "positive_folds": int(sum(delta > 0.0 for delta in deltas)),
            "minimum_fold_delta": float(min(deltas)),
            "every_fold_above_v35": every_above_v35,
            "prediction_sha256": _sha(blend),
        },
        "outer_models": fold_reports,
        "diagnostic_alpha_grid": diagnostic_grid,
        "promotion": {"advance_to_three_seed_fullfit": promotion},
        "artifacts": {
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "cache_manifest_sha256": sha256_file(CACHE / "manifest.json"),
            "predictions": str(PREDICTIONS),
        },
        "timing_seconds": time.perf_counter() - started,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "standalone": standalone_score,
        "baseline_v152": v152_total,
        "candidate": blend_score,
        "delta_over_v152": blend_score - v152_total,
        "fold_deltas": deltas,
        "promotion": report["promotion"],
        "timing_seconds": report["timing_seconds"],
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
