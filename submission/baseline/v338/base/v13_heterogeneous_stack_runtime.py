"""Fail-closed causal runtime for the authorized ADIA v13 six-base stack.

The runtime performs no fitting.  ``train_impl`` only authenticates immutable
resources and deliberately does not import LightGBM or construct a feature
state before Crunch forks its inference workers.  ``infer_impl`` constructs
one shared causal feature state per series, consumes each online point once,
evaluates the six bases in their frozen order, and applies the fixed final
raw-logit affine meta model.

Neither this module nor its resource contract contains a Stage-3 prediction or
score reader.  Stage-3 authorization has already been reduced to the compact
final-meta config and immutable full-fit resource hashes by offline tooling.
"""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Repository/package import.
    from . import extratrees_runtime
    from . import production_runtime as feature_runtime
    from . import xgboost_flat_runtime
except ImportError:  # Top-level imports after files are copied into a bundle.
    import extratrees_runtime  # type: ignore[no-redef]
    import production_runtime as feature_runtime  # type: ignore[no-redef]
    import xgboost_flat_runtime  # type: ignore[no-redef]


COMPETITION = "ADIA Structural Break Challenge: Real-Time Edition"
CANDIDATE_ID = "v13_heterogeneous_stack"
MANIFEST_FILENAME = "production_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
BASE_ORDER = ("v2", "v3", "v4seq", "pairrank", "xgb", "extra_trees")
LGB_BASE_ORDER = ("v2", "v3", "v4seq")
PAIRRANK_SEEDS = (42, 1337, 2024)
FEATURE_VIEWS = ("v2", "v3_full", "v4_seq18")
FEATURE_VIEW_BY_BASE = {
    "v2": "v2",
    "v3": "v3_full",
    "v4seq": "v4_seq18",
    "pairrank": "v4_seq18",
    "xgb": "v4_seq18",
    "extra_trees": "v4_seq18",
}
FEATURE_COUNT_BY_BASE = {
    base: feature_runtime.VIEW_FEATURE_COUNTS[view]
    for base, view in FEATURE_VIEW_BY_BASE.items()
}
EXPECTED_TREE_COUNT = {
    "v2": 700,
    "v3": 475,
    "v4seq": 700,
    "pairrank": 300,
    "xgb": 600,
    "extra_trees": 400,
}

PROBABILITY_CLIP = 1e-7
PAIRRANK_Z_CLIP = (-40.0, 40.0)
PAIRRANK_CALIBRATION_VERSION = "train-raw-linear-quantile-iqr-sigmoid-v1"
PAIRRANK_ENSEMBLE = "arithmetic_mean_seed_probabilities"
META_EVALUATION = "raw_logit_affine_fixed_base_order_float64_left_to_right"
FEATURE_UPDATE_CONTRACT = "one_shared_state_update_per_online_tick"
TRAIN_MODE = "validation-only-no-op"
EXPECTED_INFER_PARALLELISM = 4
REQUIRED_PACKAGES = {"lightgbm": "4.6.0", "numpy": "2.3.3"}
CANONICAL_REQUIREMENTS_SHA256 = (
    "b1b54bb65f8928926774e357a55b16c2a95349ae16850eb20b5c985c73e65862"
)
EXPECTED_PLAN_SHA256 = (
    "4f7db267175f72a1db99654aed187e38985e19e85c9c297b3a0e72d15e73ffcd"
)
SHA256_LENGTH = 64

SOURCE_PATHS = frozenset(
    {
        "main.py",
        "v13_heterogeneous_stack_runtime.py",
        "production_runtime.py",
        "extratrees_runtime.py",
        "xgboost_flat_runtime.py",
        "experiments/__init__.py",
        "experiments/v2_candidate.py",
        "experiments/v3_candidate.py",
        "experiments/v4_hybrid.py",
        "experiments/v5_sequential_evidence.py",
    }
)

TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "competition",
    "candidate_id",
    "base_order",
    "meta",
    "runtime",
    "components",
    "source_sha256",
    "fullfit_provenance",
}
META_FIELDS = {
    "probability_clip",
    "raw_logit_alpha_by_base",
    "raw_intercept",
    "final_meta_artifact_file",
    "final_meta_artifact_sha256",
    "final_meta_artifact_size_bytes",
}
RUNTIME_FIELDS = {
    "packages",
    "requirements_sha256",
    "infer_parallelism",
    "train_mode",
    "prediction_clip",
    "feature_views",
    "feature_state_update",
    "future_ticks_used",
    "supervised_online_updates",
    "meta_evaluation",
    "xgboost_runtime_dependency",
    "sklearn_runtime_dependency",
}
FINAL_META_CONFIG_FIELDS = {
    "schema_version",
    "status",
    "authorized_by_stage3",
    "competition",
    "candidate_id",
    "base_order",
    "blend_space",
    "probability_clip",
    "weighted_logit_mean",
    "weighted_logit_scale",
    "standardized_coef_by_base",
    "standardized_intercept",
    "raw_logit_coef_by_base",
    "raw_logit_intercept",
    "algebraic_flattening_passed",
    "stage3_report_sha256",
    "plan_sha256",
    "metrics_computed_by_finalizer",
    "fullfit_base_models_required_before_submission",
}
RESEARCH_FINAL_META_CONFIG_FIELDS = FINAL_META_CONFIG_FIELDS | {
    "authorization_mode",
    "original_stage3_passed",
    "research_submission_authorization_sha256",
    "research_submission_purpose",
}
RESEARCH_AUTHORIZATION_MODE = "one_off_public_ts_auc_measurement"
RESEARCH_SUBMISSION_PURPOSE = (
    "measure_public_ts_auc_once_for_frozen_v13_heterogeneous_stack"
)

# Alias the audited feature implementation; this module contains no second
# feature implementation.
SinglePassFeatureState = feature_runtime.SinglePassFeatureState


@dataclass(frozen=True)
class TextModelMetadata:
    feature_count: int
    tree_count: int
    objective: str


@dataclass(frozen=True)
class ValidatedLGBComponent:
    name: str
    view: str
    model_path: Path
    feature_count: int
    tree_count: int


@dataclass(frozen=True)
class MarginCalibration:
    median: float
    scale: float
    z_min: float
    z_max: float

    def apply(self, raw_margin: float) -> float:
        margin = float(raw_margin)
        if not math.isfinite(margin):
            raise RuntimeError("PairRank produced a non-finite raw margin")
        z = (margin - self.median) / self.scale
        z = min(max(z, self.z_min), self.z_max)
        return _sigmoid(z)


@dataclass(frozen=True)
class ValidatedPairSeed:
    seed: int
    model_path: Path
    feature_count: int
    tree_count: int
    calibration: MarginCalibration


@dataclass(frozen=True)
class RawLogitMeta:
    alpha: tuple[float, ...]
    intercept: float
    probability_clip: float

    def predict(self, base_probabilities: Sequence[float]) -> float:
        if len(base_probabilities) != len(BASE_ORDER):
            raise ValueError("meta model requires exactly six base probabilities")
        score = float(self.intercept)
        for base, coefficient, raw_probability in zip(
            BASE_ORDER, self.alpha, base_probabilities, strict=True
        ):
            probability = float(raw_probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError(f"{base} produced an invalid probability")
            clipped = min(
                max(probability, self.probability_clip),
                1.0 - self.probability_clip,
            )
            logit = math.log(clipped) - math.log1p(-clipped)
            score += coefficient * logit
        result = _sigmoid(score)
        return min(
            max(result, self.probability_clip),
            1.0 - self.probability_clip,
        )


@dataclass(frozen=True)
class ValidatedResources:
    lgb_components: tuple[ValidatedLGBComponent, ...]
    pairrank_seeds: tuple[ValidatedPairSeed, ...]
    xgboost_model: Any
    extratrees_model: Any
    meta: RawLogitMeta
    candidate_id: str


@dataclass(frozen=True)
class LoadedLGBComponent:
    name: str
    view: str
    booster: Any


@dataclass(frozen=True)
class LoadedPairSeed:
    seed: int
    booster: Any
    calibration: MarginCalibration


@dataclass(frozen=True)
class LoadedResources:
    lgb_components: tuple[LoadedLGBComponent, ...]
    pairrank_seeds: tuple[LoadedPairSeed, ...]
    xgboost_model: Any
    extratrees_model: Any
    meta: RawLogitMeta
    candidate_id: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact(actual: Any, expected: Any, description: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{description} changed: expected {expected!r}, got {actual!r}"
        )


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _safe_relative_path(value: Any, *, description: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"unsafe {description}: {value!r}")
    return path


def _resolved_child(root: Path, value: Any, *, description: str) -> Path:
    relative = _safe_relative_path(value, description=description)
    resolved_root = root.expanduser().resolve()
    path = (resolved_root / Path(*relative.parts)).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"{description} escapes its resource root")
    return path


def _read_stable_json(path: Path, description: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    before = sha256_file(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"{description} contains non-finite constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"{description} changed while being read")
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value, before


def _source_path(relative_name: str) -> Path:
    code_root = Path(__file__).resolve().parent
    for candidate in (code_root / relative_name, code_root.parent / relative_name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing packaged source: {relative_name}")


def _validate_source_files(manifest: Mapping[str, Any]) -> None:
    records = manifest.get("source_sha256")
    if not isinstance(records, Mapping) or set(records) != set(SOURCE_PATHS):
        raise ValueError("runtime source SHA-256 inventory changed")
    for relative_name in sorted(SOURCE_PATHS):
        expected = records.get(relative_name)
        if not _is_sha256(expected):
            raise ValueError(f"invalid runtime source hash: {relative_name}")
        if sha256_file(_source_path(relative_name)) != expected:
            raise RuntimeError(f"packaged source SHA-256 mismatch: {relative_name}")


def _validate_runtime_versions() -> None:
    # Metadata lookup avoids importing LightGBM in Crunch's pre-fork parent.
    installed = {
        name: importlib_metadata.version(name) for name in REQUIRED_PACKAGES
    }
    mismatches = {
        name: (installed[name], required)
        for name, required in REQUIRED_PACKAGES.items()
        if installed[name] != required
    }
    if mismatches:
        raise RuntimeError(f"cloud runtime pin mismatch: {mismatches}")


def _validate_file(
    root: Path,
    *,
    relative: Any,
    digest: Any,
    size_bytes: Any,
    description: str,
) -> Path:
    path = _resolved_child(root, relative, description=description)
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    if not _is_sha256(digest):
        raise ValueError(f"{description} SHA-256 is invalid")
    if sha256_file(path) != digest:
        raise RuntimeError(f"{description} SHA-256 mismatch")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError(f"{description} size is invalid")
    if int(path.stat().st_size) != size_bytes:
        raise RuntimeError(f"{description} size mismatch")
    return path


def _sigmoid(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("affine score is non-finite")
    if number >= 0.0:
        return 1.0 / (1.0 + math.exp(-number))
    exponential = math.exp(number)
    return exponential / (1.0 + exponential)


def _float_map(value: Any, description: str) -> dict[str, float]:
    mapping = _require_mapping(value, description)
    if set(mapping) != set(BASE_ORDER):
        raise ValueError(f"{description} base inventory changed")
    result: dict[str, float] = {}
    for base in BASE_ORDER:
        raw = mapping[base]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{description}.{base} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{description}.{base} must be finite")
        result[base] = number
    return result


def _validate_final_meta_config(
    config: Mapping[str, Any], meta_manifest: Mapping[str, Any]
) -> RawLogitMeta:
    fields = set(config)
    research_override = fields == RESEARCH_FINAL_META_CONFIG_FIELDS
    if fields not in (FINAL_META_CONFIG_FIELDS, RESEARCH_FINAL_META_CONFIG_FIELDS):
        raise ValueError("final-meta config field inventory changed")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("authorized_by_stage3", not research_override),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("base_order", list(BASE_ORDER)),
        ("blend_space", "standardized clipped base logits"),
        ("probability_clip", PROBABILITY_CLIP),
        ("algebraic_flattening_passed", True),
        ("plan_sha256", EXPECTED_PLAN_SHA256),
        ("metrics_computed_by_finalizer", False),
        ("fullfit_base_models_required_before_submission", True),
    ):
        _require_exact(config.get(key), expected, f"final-meta {key}")
    if research_override:
        for key, expected in (
            ("authorization_mode", RESEARCH_AUTHORIZATION_MODE),
            ("original_stage3_passed", False),
            ("research_submission_purpose", RESEARCH_SUBMISSION_PURPOSE),
        ):
            _require_exact(config.get(key), expected, f"final-meta {key}")
        if not _is_sha256(config.get("research_submission_authorization_sha256")):
            raise ValueError("research-submission authorization SHA-256 is invalid")
    if not _is_sha256(config.get("stage3_report_sha256")):
        raise ValueError("final-meta Stage-3 report hash is invalid")

    mean = _float_map(config.get("weighted_logit_mean"), "weighted logit mean")
    scale = _float_map(config.get("weighted_logit_scale"), "weighted logit scale")
    standardized = _float_map(
        config.get("standardized_coef_by_base"), "standardized coefficient"
    )
    raw = _float_map(config.get("raw_logit_coef_by_base"), "raw coefficient")
    if any(value <= 0.0 for value in scale.values()):
        raise ValueError("weighted logit scale must be positive")
    for base in BASE_ORDER:
        expected = standardized[base] / scale[base]
        if not math.isclose(raw[base], expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"raw/standardized coefficient mismatch: {base}")

    standardized_intercept = config.get("standardized_intercept")
    raw_intercept = config.get("raw_logit_intercept")
    for description, value in (
        ("standardized intercept", standardized_intercept),
        ("raw intercept", raw_intercept),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"final-meta {description} must be finite")
    expected_intercept = float(standardized_intercept)
    for base in BASE_ORDER:
        expected_intercept -= standardized[base] * mean[base] / scale[base]
    if not math.isclose(
        float(raw_intercept), expected_intercept, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError("raw/standardized intercept mismatch")

    manifest_alpha = _float_map(
        meta_manifest.get("raw_logit_alpha_by_base"), "manifest meta alpha"
    )
    if manifest_alpha != raw:
        raise ValueError("manifest/final-meta raw coefficients differ")
    _require_exact(
        meta_manifest.get("raw_intercept"),
        raw_intercept,
        "manifest/final-meta raw intercept",
    )
    _require_exact(
        meta_manifest.get("probability_clip"),
        PROBABILITY_CLIP,
        "manifest meta probability clip",
    )
    return RawLogitMeta(
        alpha=tuple(raw[base] for base in BASE_ORDER),
        intercept=float(raw_intercept),
        probability_clip=PROBABILITY_CLIP,
    )


def _inspect_lgb_text_model(
    path: Path, *, expected_objective: str
) -> TextModelMetadata:
    version: str | None = None
    num_class: int | None = None
    trees_per_iteration: int | None = None
    max_feature_index: int | None = None
    objective: str | None = None
    declared_tree_sizes: int | None = None
    declared_feature_names: int | None = None
    tree_count = 0
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if line_number == 1 and line != "tree":
                raise ValueError(f"invalid LightGBM text-model header: {path}")
            if line.startswith("version="):
                version = line.split("=", 1)[1]
            elif line.startswith("num_class="):
                num_class = int(line.split("=", 1)[1])
            elif line.startswith("num_tree_per_iteration="):
                trees_per_iteration = int(line.split("=", 1)[1])
            elif line.startswith("max_feature_idx="):
                max_feature_index = int(line.split("=", 1)[1])
            elif line.startswith("objective="):
                objective = line.split("=", 1)[1]
            elif line.startswith("feature_names="):
                declared_feature_names = len(line.split("=", 1)[1].split())
            elif line.startswith("tree_sizes="):
                declared_tree_sizes = len(line.split("=", 1)[1].split())
            elif line.startswith("Tree="):
                index = int(line.split("=", 1)[1])
                if index != tree_count:
                    raise ValueError("LightGBM tree indices are not contiguous")
                tree_count += 1
    if version != "v4" or num_class != 1 or trees_per_iteration != 1:
        raise ValueError("unsupported LightGBM text-model structure")
    if max_feature_index is None or max_feature_index < 0:
        raise ValueError("LightGBM model lacks max_feature_idx")
    feature_count = max_feature_index + 1
    if declared_feature_names != feature_count:
        raise ValueError("LightGBM feature-name width changed")
    if tree_count <= 0 or declared_tree_sizes != tree_count:
        raise ValueError("LightGBM tree-count structure changed")
    if expected_objective == "binary":
        if objective is None:
            raise ValueError("LightGBM model lacks an objective")
        if not objective.startswith("binary "):
            raise ValueError("LightGBM probability model objective changed")
    elif expected_objective == "custom":
        # LightGBM 4.6 omits ``objective=`` when a Booster was trained with a
        # Python callable objective.  The PairRank caller authenticates the
        # exact model bytes and the separately hash-bound artifact objective
        # before reaching this custom-only branch.  Ordinary probability
        # models above still require an explicit binary objective line.
        if objective not in (None, "custom"):
            raise ValueError("PairRank LightGBM objective changed")
        if objective is None:
            objective = "custom"
    else:  # pragma: no cover - internal programming error
        raise RuntimeError("unknown expected LightGBM objective")
    return TextModelMetadata(feature_count, tree_count, objective)


def _validate_lgb_artifact(
    path: Path,
    *,
    model_sha256: str,
    feature_count: int,
    tree_count: int,
    base: str,
) -> None:
    value, _ = _read_stable_json(path, f"{base} artifact manifest")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
    ):
        _require_exact(value.get(key), expected, f"{base} artifact {key}")
    model = _require_mapping(value.get("model"), f"{base} artifact model")
    for key, expected in (
        ("sha256", model_sha256),
        ("num_features", feature_count),
        ("num_trees", tree_count),
    ):
        _require_exact(model.get(key), expected, f"{base} artifact model.{key}")
    runtime = _require_mapping(value.get("runtime"), f"{base} artifact runtime")
    for key, expected in (
        ("required_packages", REQUIRED_PACKAGES),
        ("installed_packages", REQUIRED_PACKAGES),
        ("is_exact_match", True),
        ("override_used", False),
    ):
        _require_exact(runtime.get(key), expected, f"{base} artifact runtime.{key}")


def _validate_lgb_component(
    root: Path, raw: Mapping[str, Any], base: str
) -> ValidatedLGBComponent:
    expected_fields = {
        "name",
        "kind",
        "feature_view",
        "feature_count",
        "num_trees",
        "model_file",
        "sha256",
        "size_bytes",
        "artifact_manifest_file",
        "artifact_manifest_sha256",
        "artifact_manifest_size_bytes",
    }
    if set(raw) != expected_fields:
        raise ValueError(f"{base} component field inventory changed")
    expected_view = FEATURE_VIEW_BY_BASE[base]
    expected_features = FEATURE_COUNT_BY_BASE[base]
    expected_trees = EXPECTED_TREE_COUNT[base]
    for key, expected in (
        ("name", base),
        ("kind", "lightgbm_probability"),
        ("feature_view", expected_view),
        ("feature_count", expected_features),
        ("num_trees", expected_trees),
    ):
        _require_exact(raw.get(key), expected, f"{base} component {key}")
    model_path = _validate_file(
        root,
        relative=raw.get("model_file"),
        digest=raw.get("sha256"),
        size_bytes=raw.get("size_bytes"),
        description=f"{base} model",
    )
    metadata = _inspect_lgb_text_model(model_path, expected_objective="binary")
    if (
        metadata.feature_count != expected_features
        or metadata.tree_count != expected_trees
    ):
        raise RuntimeError(f"{base} LightGBM model structure changed")
    artifact_path = _validate_file(
        root,
        relative=raw.get("artifact_manifest_file"),
        digest=raw.get("artifact_manifest_sha256"),
        size_bytes=raw.get("artifact_manifest_size_bytes"),
        description=f"{base} artifact manifest",
    )
    _validate_lgb_artifact(
        artifact_path,
        model_sha256=str(raw["sha256"]),
        feature_count=expected_features,
        tree_count=expected_trees,
        base=base,
    )
    return ValidatedLGBComponent(
        base, expected_view, model_path, expected_features, expected_trees
    )


def _validate_calibration(value: Mapping[str, Any]) -> MarginCalibration:
    required_fields = {
        "version",
        "fit_rows",
        "fit_scope",
        "source",
        "labels_used",
        "validation_values_used",
        "quantile_method",
        "q25",
        "median",
        "q75",
        "iqr",
        "scale",
        "zero_iqr_fallback",
        "zero_iqr_fallback_scale",
        "z_clip",
        "full_train_folds",
        "synthetic_heldout_identity_used",
    }
    if set(value) != required_fields:
        raise ValueError("PairRank calibration field inventory changed")
    for key, expected in (
        ("version", PAIRRANK_CALIBRATION_VERSION),
        ("fit_scope", "all_series_full_train_pairrank_rows_raw_score_only"),
        ("source", "all_series_full_train_pairrank_rows_raw_score_only"),
        ("labels_used", False),
        ("validation_values_used", False),
        ("quantile_method", "numpy-linear"),
        ("zero_iqr_fallback_scale", 1.0),
        ("z_clip", list(PAIRRANK_Z_CLIP)),
        ("full_train_folds", [0, 1, 2, 3, 4]),
        ("synthetic_heldout_identity_used", False),
    ):
        _require_exact(value.get(key), expected, f"PairRank calibration {key}")
    if type(value.get("fit_rows")) is not int or value["fit_rows"] <= 0:
        raise ValueError("PairRank calibration fit_rows is invalid")
    numbers: dict[str, float] = {}
    for key in ("q25", "median", "q75", "iqr", "scale"):
        raw = value.get(key)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ValueError(f"PairRank calibration {key} is invalid")
        numbers[key] = float(raw)
    if not numbers["q25"] <= numbers["median"] <= numbers["q75"]:
        raise ValueError("PairRank calibration quantiles are not ordered")
    expected_iqr = numbers["q75"] - numbers["q25"]
    if not math.isclose(numbers["iqr"], expected_iqr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("PairRank calibration IQR changed")
    fallback = value.get("zero_iqr_fallback")
    if type(fallback) is not bool:
        raise ValueError("PairRank calibration fallback flag is invalid")
    if fallback:
        if numbers["scale"] != 1.0 or numbers["iqr"] > 1e-12:
            raise ValueError("PairRank fallback scale contract changed")
    elif numbers["scale"] != numbers["iqr"] or numbers["scale"] <= 1e-12:
        raise ValueError("PairRank calibration scale contract changed")
    return MarginCalibration(
        median=numbers["median"],
        scale=numbers["scale"],
        z_min=PAIRRANK_Z_CLIP[0],
        z_max=PAIRRANK_Z_CLIP[1],
    )


def _validate_pairrank_component(
    root: Path, raw: Mapping[str, Any]
) -> tuple[ValidatedPairSeed, ...]:
    expected_fields = {
        "name",
        "kind",
        "feature_view",
        "feature_count",
        "seeds",
        "ensemble",
        "artifact_manifest_file",
        "artifact_manifest_sha256",
        "artifact_manifest_size_bytes",
    }
    if set(raw) != expected_fields:
        raise ValueError("PairRank component field inventory changed")
    for key, expected in (
        ("name", "pairrank"),
        ("kind", "pairrank_probability_mean"),
        ("feature_view", "v4_seq18"),
        ("feature_count", 440),
        ("ensemble", PAIRRANK_ENSEMBLE),
    ):
        _require_exact(raw.get(key), expected, f"PairRank component {key}")
    artifact_path = _validate_file(
        root,
        relative=raw.get("artifact_manifest_file"),
        digest=raw.get("artifact_manifest_sha256"),
        size_bytes=raw.get("artifact_manifest_size_bytes"),
        description="PairRank artifact manifest",
    )
    artifact, _ = _read_stable_json(artifact_path, "PairRank artifact manifest")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", "v13_pairrank_v4_seq18_fullfit"),
        ("feature_view", "v4_seq18"),
        ("feature_count", 440),
        ("train_folds", [0, 1, 2, 3, 4]),
        ("synthetic_heldout_identity_used", False),
        ("metrics_computed", False),
    ):
        _require_exact(artifact.get(key), expected, f"PairRank artifact {key}")
    artifact_model = _require_mapping(
        artifact.get("model"), "PairRank artifact model contract"
    )
    _require_exact(
        artifact_model.get("objective"),
        "PairwiseLogisticObjective",
        "PairRank artifact custom objective",
    )

    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(PAIRRANK_SEEDS):
        raise ValueError("PairRank seed inventory changed")
    validated: list[ValidatedPairSeed] = []
    calibration_documents: list[dict[str, Any]] = []
    for expected_seed, seed_raw in zip(PAIRRANK_SEEDS, seeds, strict=True):
        seed_record = _require_mapping(seed_raw, "PairRank seed")
        if set(seed_record) != {
            "seed",
            "model_file",
            "sha256",
            "size_bytes",
            "num_trees",
            "calibration_file",
            "calibration_sha256",
            "calibration_size_bytes",
        }:
            raise ValueError("PairRank seed field inventory changed")
        _require_exact(seed_record.get("seed"), expected_seed, "PairRank seed order")
        _require_exact(
            seed_record.get("num_trees"),
            EXPECTED_TREE_COUNT["pairrank"],
            "PairRank tree count",
        )
        model_path = _validate_file(
            root,
            relative=seed_record.get("model_file"),
            digest=seed_record.get("sha256"),
            size_bytes=seed_record.get("size_bytes"),
            description=f"PairRank seed {expected_seed} model",
        )
        metadata = _inspect_lgb_text_model(model_path, expected_objective="custom")
        if metadata.feature_count != 440 or metadata.tree_count != 300:
            raise RuntimeError("PairRank LightGBM model structure changed")
        calibration_path = _validate_file(
            root,
            relative=seed_record.get("calibration_file"),
            digest=seed_record.get("calibration_sha256"),
            size_bytes=seed_record.get("calibration_size_bytes"),
            description=f"PairRank seed {expected_seed} calibration",
        )
        calibration_json, _ = _read_stable_json(
            calibration_path, f"PairRank seed {expected_seed} calibration"
        )
        calibration = _validate_calibration(calibration_json)
        calibration_documents.append(calibration_json)
        validated.append(
            ValidatedPairSeed(expected_seed, model_path, 440, 300, calibration)
        )
    artifact_seeds = artifact.get("seeds")
    if not isinstance(artifact_seeds, list) or len(artifact_seeds) != 3:
        raise ValueError("PairRank artifact seed inventory changed")
    for expected_seed, source, packaged, checked, calibration_document in zip(
        PAIRRANK_SEEDS,
        artifact_seeds,
        seeds,
        validated,
        calibration_documents,
        strict=True,
    ):
        record = _require_mapping(source, "PairRank artifact seed")
        packaged_record = _require_mapping(packaged, "PairRank packaged seed")
        _require_exact(record.get("seed"), expected_seed, "PairRank artifact seed")
        artifact_model = _require_mapping(
            record.get("model"), "PairRank artifact model"
        )
        artifact_calibration = _require_mapping(
            record.get("calibration"), "PairRank artifact calibration"
        )
        for key, expected in (
            ("sha256", sha256_file(checked.model_path)),
            ("size_bytes", packaged_record.get("size_bytes")),
        ):
            _require_exact(
                artifact_model.get(key), expected, f"PairRank artifact model {key}"
            )
        for key, expected in (
            ("sha256", packaged_record.get("calibration_sha256")),
            ("size_bytes", packaged_record.get("calibration_size_bytes")),
        ):
            _require_exact(
                artifact_calibration.get(key),
                expected,
                f"PairRank artifact calibration {key}",
            )
        _require_exact(
            record.get("calibration_values"),
            calibration_document,
            "PairRank artifact calibration values",
        )
        _require_exact(record.get("tree_count"), 300, "PairRank artifact tree count")
    return tuple(validated)


def _validate_xgboost_component(root: Path, raw: Mapping[str, Any]) -> Any:
    expected_fields = {
        "name",
        "kind",
        "feature_view",
        "feature_count",
        "tree_count",
        "model_file",
        "sha256",
        "size_bytes",
        "model_manifest_file",
        "model_manifest_sha256",
        "model_manifest_size_bytes",
        "base_margin",
        "arrays",
    }
    if set(raw) != expected_fields:
        raise ValueError("XGBoost component field inventory changed")
    for key, expected in (
        ("name", "xgb"),
        ("kind", "flat_xgboost_binary_logistic"),
        ("feature_view", "v4_seq18"),
        ("feature_count", 440),
        ("tree_count", 600),
    ):
        _require_exact(raw.get(key), expected, f"XGBoost component {key}")
    model_path = _validate_file(
        root,
        relative=raw.get("model_file"),
        digest=raw.get("sha256"),
        size_bytes=raw.get("size_bytes"),
        description="flat XGBoost model",
    )
    manifest_path = _validate_file(
        root,
        relative=raw.get("model_manifest_file"),
        digest=raw.get("model_manifest_sha256"),
        size_bytes=raw.get("model_manifest_size_bytes"),
        description="flat XGBoost manifest",
    )
    exporter_manifest, _ = _read_stable_json(
        manifest_path, "flat XGBoost manifest"
    )
    output = _require_mapping(
        exporter_manifest.get("output"), "flat XGBoost manifest output"
    )
    model_contract = _require_mapping(
        exporter_manifest.get("model"), "flat XGBoost manifest model"
    )
    _require_exact(output.get("arrays"), raw.get("arrays"), "XGBoost arrays")
    _require_exact(
        model_contract.get("base_margin"), raw.get("base_margin"), "XGBoost base margin"
    )
    model = xgboost_flat_runtime.load_export(model_path, manifest_path)
    if model.feature_count != 440 or model.tree_count != 600:
        raise RuntimeError("flat XGBoost model structure changed")
    if float(model.base_margin) != float(raw["base_margin"]):
        raise RuntimeError("flat XGBoost base margin changed")
    return model


def _validate_extratrees_component(root: Path, raw: Mapping[str, Any]) -> Any:
    expected_fields = {
        "name",
        "kind",
        "feature_view",
        "feature_count",
        "tree_count",
        "model_file",
        "sha256",
        "size_bytes",
        "artifact_manifest_file",
        "artifact_manifest_sha256",
        "artifact_manifest_size_bytes",
        "arrays",
    }
    if set(raw) != expected_fields:
        raise ValueError("ExtraTrees component field inventory changed")
    for key, expected in (
        ("name", "extra_trees"),
        ("kind", "flat_extratrees_probability"),
        ("feature_view", "v4_seq18"),
        ("feature_count", 440),
        ("tree_count", 400),
    ):
        _require_exact(raw.get(key), expected, f"ExtraTrees component {key}")
    model_path = _validate_file(
        root,
        relative=raw.get("model_file"),
        digest=raw.get("sha256"),
        size_bytes=raw.get("size_bytes"),
        description="flat ExtraTrees model",
    )
    artifact_path = _validate_file(
        root,
        relative=raw.get("artifact_manifest_file"),
        digest=raw.get("artifact_manifest_sha256"),
        size_bytes=raw.get("artifact_manifest_size_bytes"),
        description="ExtraTrees artifact manifest",
    )
    artifact, _ = _read_stable_json(artifact_path, "ExtraTrees artifact manifest")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", "v13_extratrees_v4_seq18_fullfit"),
        ("feature_view", "v4_seq18"),
        ("feature_count", 440),
        ("metrics_computed", False),
    ):
        _require_exact(artifact.get(key), expected, f"ExtraTrees artifact {key}")
    _require_exact(
        artifact.get("flat_arrays"), raw.get("arrays"), "ExtraTrees flat arrays"
    )
    artifact_model = _require_mapping(
        artifact.get("flat_model"), "ExtraTrees artifact flat model"
    )
    for key, expected in (
        ("sha256", raw.get("sha256")),
        ("size_bytes", raw.get("size_bytes")),
    ):
        _require_exact(
            artifact_model.get(key), expected, f"ExtraTrees artifact flat model {key}"
        )
    records = extratrees_runtime._validate_array_records(raw.get("arrays"))  # noqa: SLF001
    forest = extratrees_runtime._load_flat_forest(model_path, records)  # noqa: SLF001
    if forest.tree_count != 400:
        raise RuntimeError("flat ExtraTrees tree count changed")
    return forest


def _validate_runtime_contract(manifest: Mapping[str, Any]) -> None:
    runtime = _require_mapping(manifest.get("runtime"), "runtime contract")
    if set(runtime) != RUNTIME_FIELDS:
        raise ValueError("runtime contract field inventory changed")
    for key, expected in (
        ("packages", REQUIRED_PACKAGES),
        ("requirements_sha256", CANONICAL_REQUIREMENTS_SHA256),
        ("infer_parallelism", EXPECTED_INFER_PARALLELISM),
        ("train_mode", TRAIN_MODE),
        ("prediction_clip", PROBABILITY_CLIP),
        ("feature_views", list(FEATURE_VIEWS)),
        ("feature_state_update", FEATURE_UPDATE_CONTRACT),
        ("future_ticks_used", False),
        ("supervised_online_updates", False),
        ("meta_evaluation", META_EVALUATION),
        ("xgboost_runtime_dependency", False),
        ("sklearn_runtime_dependency", False),
    ):
        _require_exact(runtime.get(key), expected, f"runtime {key}")


def _validate_fullfit_provenance(root: Path, manifest: Mapping[str, Any]) -> None:
    provenance = _require_mapping(
        manifest.get("fullfit_provenance"), "full-fit provenance"
    )
    if set(provenance) != {"manifest_file", "manifest_sha256", "manifest_size_bytes"}:
        raise ValueError("full-fit provenance field inventory changed")
    # Deliberately hash-bound but not JSON-parsed: no nested Stage-3 report,
    # score, or prediction identity is consumed by the production runtime.
    _validate_file(
        root,
        relative=provenance.get("manifest_file"),
        digest=provenance.get("manifest_sha256"),
        size_bytes=provenance.get("manifest_size_bytes"),
        description="full-fit provenance manifest",
    )


def _validate_bundle(
    model_directory_path: str | Path,
    *,
    check_runtime: bool,
) -> ValidatedResources:
    root = Path(model_directory_path).expanduser().resolve()
    manifest, _ = _read_stable_json(root / MANIFEST_FILENAME, "production manifest")
    if set(manifest) != TOP_LEVEL_FIELDS:
        raise ValueError("production manifest field inventory changed")
    for key, expected in (
        ("schema_version", MANIFEST_SCHEMA_VERSION),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("base_order", list(BASE_ORDER)),
    ):
        _require_exact(manifest.get(key), expected, f"production {key}")
    _validate_runtime_contract(manifest)
    if check_runtime:
        _validate_runtime_versions()
    _validate_source_files(manifest)
    _validate_fullfit_provenance(root, manifest)

    meta_manifest = _require_mapping(manifest.get("meta"), "meta contract")
    if set(meta_manifest) != META_FIELDS:
        raise ValueError("meta contract field inventory changed")
    config_path = _validate_file(
        root,
        relative=meta_manifest.get("final_meta_artifact_file"),
        digest=meta_manifest.get("final_meta_artifact_sha256"),
        size_bytes=meta_manifest.get("final_meta_artifact_size_bytes"),
        description="final-meta config",
    )
    config, _ = _read_stable_json(config_path, "final-meta config")
    meta = _validate_final_meta_config(config, meta_manifest)

    raw_components = manifest.get("components")
    if not isinstance(raw_components, list) or len(raw_components) != len(BASE_ORDER):
        raise ValueError("production component inventory changed")
    names = tuple(
        item.get("name") if isinstance(item, Mapping) else None
        for item in raw_components
    )
    if names != BASE_ORDER:
        raise ValueError("production component order changed")
    lgb_components = tuple(
        _validate_lgb_component(
            root,
            _require_mapping(raw_components[index], f"{base} component"),
            base,
        )
        for index, base in enumerate(LGB_BASE_ORDER)
    )
    pairrank = _validate_pairrank_component(
        root, _require_mapping(raw_components[3], "PairRank component")
    )
    xgboost = _validate_xgboost_component(
        root, _require_mapping(raw_components[4], "XGBoost component")
    )
    extra_trees = _validate_extratrees_component(
        root, _require_mapping(raw_components[5], "ExtraTrees component")
    )
    return ValidatedResources(
        lgb_components=lgb_components,
        pairrank_seeds=pairrank,
        xgboost_model=xgboost,
        extratrees_model=extra_trees,
        meta=meta,
        candidate_id=CANDIDATE_ID,
    )


def validate_resources(
    model_directory_path: str | Path,
    *,
    check_runtime: bool = True,
) -> ValidatedResources:
    """Authenticate every immutable resource without importing LightGBM."""

    return _validate_bundle(model_directory_path, check_runtime=check_runtime)


def load_resources(
    model_directory_path: str | Path,
    *,
    check_runtime: bool = True,
) -> LoadedResources:
    validated = validate_resources(
        model_directory_path, check_runtime=check_runtime
    )
    # The only LightGBM import occurs in an inference worker, after Crunch has
    # forked.  Flat XGBoost/ExtraTrees models require NumPy only.
    lgb = importlib.import_module("lightgbm")
    loaded_lgb: list[LoadedLGBComponent] = []
    for component in validated.lgb_components:
        booster = lgb.Booster(model_file=str(component.model_path))
        if (
            int(booster.num_feature()) != component.feature_count
            or int(booster.num_trees()) != component.tree_count
        ):
            raise RuntimeError(f"loaded {component.name} LightGBM structure changed")
        loaded_lgb.append(
            LoadedLGBComponent(component.name, component.view, booster)
        )
    loaded_pair: list[LoadedPairSeed] = []
    for seed in validated.pairrank_seeds:
        booster = lgb.Booster(model_file=str(seed.model_path))
        if (
            int(booster.num_feature()) != seed.feature_count
            or int(booster.num_trees()) != seed.tree_count
        ):
            raise RuntimeError(f"loaded PairRank seed {seed.seed} structure changed")
        loaded_pair.append(LoadedPairSeed(seed.seed, booster, seed.calibration))
    return LoadedResources(
        lgb_components=tuple(loaded_lgb),
        pairrank_seeds=tuple(loaded_pair),
        xgboost_model=validated.xgboost_model,
        extratrees_model=validated.extratrees_model,
        meta=validated.meta,
        candidate_id=validated.candidate_id,
    )


def _one_prediction(
    booster: Any,
    row: np.ndarray,
    *,
    raw_score: bool,
) -> float:
    values = np.asarray(
        booster.predict(
            row.reshape(1, -1),
            raw_score=raw_score,
            num_threads=1,
        ),
        dtype=np.float64,
    )
    if values.shape != (1,) or not math.isfinite(float(values[0])):
        raise RuntimeError("LightGBM produced an invalid one-row prediction")
    return float(values[0])


def predict_base_probabilities(
    resources: LoadedResources,
    rows: Mapping[str, np.ndarray],
) -> tuple[float, ...]:
    if tuple(component.name for component in resources.lgb_components) != LGB_BASE_ORDER:
        raise RuntimeError("loaded LightGBM base order changed")
    if tuple(seed.seed for seed in resources.pairrank_seeds) != PAIRRANK_SEEDS:
        raise RuntimeError("loaded PairRank seed order changed")
    if set(rows) != set(FEATURE_VIEWS):
        raise RuntimeError("shared feature state returned an unexpected view inventory")
    normalized_rows: dict[str, np.ndarray] = {}
    for view in FEATURE_VIEWS:
        row = np.asarray(rows[view], dtype=np.float32)
        expected_width = feature_runtime.VIEW_FEATURE_COUNTS[view]
        if row.shape != (expected_width,) or not np.isfinite(row).all():
            raise RuntimeError(f"shared feature row changed for {view}")
        normalized_rows[view] = row

    probabilities: list[float] = []
    for component in resources.lgb_components:
        probabilities.append(
            _one_prediction(
                component.booster,
                normalized_rows[component.view],
                raw_score=False,
            )
        )
    pair_sum = 0.0
    pair_row = normalized_rows["v4_seq18"]
    for seed in resources.pairrank_seeds:
        raw = _one_prediction(seed.booster, pair_row, raw_score=True)
        pair_sum += seed.calibration.apply(raw)
    probabilities.append(pair_sum / float(len(PAIRRANK_SEEDS)))
    probabilities.append(float(resources.xgboost_model.predict_one(pair_row)))
    probabilities.append(float(resources.extratrees_model.predict_one(pair_row)))
    if len(probabilities) != len(BASE_ORDER):
        raise RuntimeError("six-base prediction inventory changed")
    for base, probability in zip(BASE_ORDER, probabilities, strict=True):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"{base} produced an invalid probability")
    return tuple(probabilities)


def train_impl(
    datasets: Any,
    model_directory_path: str,
    *,
    _check_runtime: bool = True,
) -> None:
    """Validation-only train; consumes no data and fits or writes nothing."""

    del datasets
    validate_resources(model_directory_path, check_runtime=_check_runtime)


def infer_impl(
    datasets: Iterable[tuple[Sequence[float], Iterable[float]]],
    model_directory_path: str,
    *,
    _check_runtime: bool = True,
):
    """Yield readiness, then one current/past-only probability per online tick."""

    resources = load_resources(
        model_directory_path, check_runtime=_check_runtime
    )
    # Runs inside the worker before readiness; it never consumes user data.
    feature_runtime._validate_feature_runtime(FEATURE_VIEWS)  # noqa: SLF001
    yield
    for historical, online in datasets:
        state = SinglePassFeatureState(historical, FEATURE_VIEWS)
        for point in online:
            # Exactly one shared update.  All six bases consume views derived
            # from this same current-prefix state before iteration advances.
            rows = state.update(float(point))
            base_probabilities = predict_base_probabilities(resources, rows)
            yield float(resources.meta.predict(base_probabilities))


INFER_PARALLELISM = EXPECTED_INFER_PARALLELISM
