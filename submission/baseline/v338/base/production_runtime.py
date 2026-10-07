"""Manifest-driven, single-pass inference for ADIA pretrained submissions.

The runtime supports the three production shapes under consideration without
changing submission code:

* independent v2 and v3 (or v3 supplement) boosters;
* one v4 + robust12 + sequential18 booster; and
* any global probability/logit blend of those components.

Every required streaming state consumes each point exactly once.  Derived
feature views are assembled from those rows, so a global blend never repeats
the expensive v2/v3/robust/sequential feature calculation.
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


COMPETITION = "ADIA Structural Break Challenge: Real-Time Edition"
MANIFEST_FILENAME = "production_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
REQUIRED_PACKAGES = {"lightgbm": "4.6.0", "numpy": "2.3.3"}
CANONICAL_REQUIREMENTS_SHA256 = (
    "b1b54bb65f8928926774e357a55b16c2a95349ae16850eb20b5c985c73e65862"
)
EXPECTED_INFER_PARALLELISM = 4
PREDICTION_CLIP = 1e-7

# Crunch uses requirements.txt while building the environment but does not
# guarantee that file is mounted under /context/code for train/infer.  Only
# uploaded Python code belongs in the runtime source-integrity contract.
MOUNTED_CODE_SOURCE_PATHS = frozenset(
    {
        "main.py",
        "production_runtime.py",
        "experiments/__init__.py",
        "experiments/v2_candidate.py",
        "experiments/v3_candidate.py",
        "experiments/v4_hybrid.py",
        "experiments/v5_robust_supplement.py",
        "experiments/v5_sequential_evidence.py",
    }
)

VIEW_FEATURE_COUNTS: dict[str, int] = {
    "v2": 229,
    "v3_full": 269,
    "v3_supplement": 193,
    "v4": 422,
    "v4_robust12": 434,
    "v4_seq18": 440,
    "v4_robust12_seq18": 452,
}

ROBUST12_LOCAL_INDICES = (37, 38, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55)
ROBUST12_NAMES = (
    "robust_w256_abs_calibrated_rank_iqr",
    "robust_w256_abs_calibrated_absdev_rank_median",
    "robust_w256_abs_calibrated_absdiff_rank_energy",
    "robust_w256_abs_calibrated_abs_second_diff_rank_energy",
    "rank_cusum_signed",
    "rank_cusum_abs",
    "rank_cusum_range",
    "rank_page_excursion",
    "absdev_rank_cusum_signed",
    "absdev_rank_cusum_abs",
    "absdev_rank_cusum_range",
    "absdev_rank_page_excursion",
)
SEQUENTIAL18_LOCAL_INDICES = tuple(range(6, 12)) + tuple(range(18, 30))
SEQUENTIAL18_FAMILIES = (
    "ar10_rank_location",
    "ar10_pit_likelihood",
    "ar10_lag_product",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def _resolved_child(root: Path, value: Any, *, label: str) -> Path:
    relative = _safe_relative_path(value, label=label)
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"{label} escapes its resource root")
    return candidate


def _read_manifest(model_directory_path: str | Path) -> tuple[Path, dict[str, Any]]:
    resource_root = Path(model_directory_path).expanduser().resolve()
    manifest_path = resource_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing pretrained manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("production manifest must be a JSON object")
    return resource_root, manifest


def _validate_runtime_versions() -> None:
    # Metadata lookup deliberately does not import LightGBM.  ``train()`` runs
    # in Crunch's parent process before four inference workers are forked; any
    # pre-fork LightGBM/OpenMP initialization can leave libomp barriers in an
    # inherited state and deadlock the workers.
    installed = {
        name: importlib_metadata.version(name) for name in REQUIRED_PACKAGES
    }
    mismatches = {
        name: (installed[name], required)
        for name, required in REQUIRED_PACKAGES.items()
        if installed[name] != required
    }
    if mismatches:
        detail = ", ".join(
            f"{name} {actual} != {required}"
            for name, (actual, required) in sorted(mismatches.items())
        )
        raise RuntimeError(f"cloud runtime pin mismatch: {detail}")


def _validate_top_level_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported production manifest schema")
    if manifest.get("competition") != COMPETITION:
        raise ValueError("resource is not certified for the ADIA real-time competition")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("production manifest lacks runtime pins")
    if dict(runtime.get("packages", {})) != REQUIRED_PACKAGES:
        raise ValueError("production manifest runtime package pins changed")
    if runtime.get("requirements_sha256") != CANONICAL_REQUIREMENTS_SHA256:
        raise ValueError("production manifest requirements SHA-256 changed")
    if runtime.get("infer_parallelism") != EXPECTED_INFER_PARALLELISM:
        raise ValueError("production manifest must lock INFER_PARALLELISM=4")
    blend = manifest.get("blend")
    if not isinstance(blend, Mapping):
        raise ValueError("production manifest lacks a blend specification")
    if blend.get("space") not in {"logit", "probability"}:
        raise ValueError("blend space must be 'logit' or 'probability'")
    clip = float(blend.get("prediction_clip", -1.0))
    if clip != PREDICTION_CLIP:
        raise ValueError(f"prediction clip must equal {PREDICTION_CLIP}")


def _required_mounted_code_source_paths(views: Sequence[str]) -> frozenset[str]:
    names = {"main.py", "production_runtime.py", "experiments/__init__.py"}
    states = _required_state_names(views)
    if "v2" in states:
        names.add("experiments/v2_candidate.py")
    if "v3" in states:
        names.add("experiments/v3_candidate.py")
    if any(view.startswith("v4") or view == "v3_supplement" for view in views):
        names.add("experiments/v4_hybrid.py")
        names.update(
            ("experiments/v2_candidate.py", "experiments/v3_candidate.py")
        )
    if "robust" in states:
        names.add("experiments/v5_robust_supplement.py")
    if "sequential" in states:
        names.add("experiments/v5_sequential_evidence.py")
    result = frozenset(names)
    if not result <= MOUNTED_CODE_SOURCE_PATHS:
        raise RuntimeError("runtime mounted-code source inventory is incomplete")
    return result


def _validate_source_files(manifest: Mapping[str, Any]) -> None:
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("production manifest lacks source SHA-256 values")
    raw_views = manifest.get("single_pass_feature_views")
    if (
        not isinstance(raw_views, list)
        or not raw_views
        or any(not isinstance(view, str) for view in raw_views)
        or raw_views != sorted(set(raw_views))
    ):
        raise ValueError("production manifest feature-view inventory changed")
    source_names = set(source_hashes)
    expected_names = _required_mounted_code_source_paths(tuple(raw_views))
    if source_names != expected_names:
        missing = sorted(expected_names - source_names)
        unexpected = sorted(str(name) for name in source_names - expected_names)
        raise ValueError(
            "runtime source SHA-256 inventory mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    code_root = Path(__file__).resolve().parent
    for relative_name, expected in source_hashes.items():
        path = _resolved_child(code_root, relative_name, label="source path")
        if not path.is_file():
            raise FileNotFoundError(f"missing packaged source: {relative_name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"packaged source SHA-256 mismatch: {relative_name}")


def _validate_provenance(
    resource_root: Path,
    component: Mapping[str, Any],
    *,
    model_sha256: str,
    feature_count: int,
) -> None:
    relative = component.get("artifact_manifest_file")
    if relative is None:
        return
    provenance_path = _resolved_child(
        resource_root, relative, label="artifact manifest path"
    )
    if not provenance_path.is_file():
        raise FileNotFoundError(f"missing artifact manifest: {provenance_path}")
    expected_provenance_sha = component.get("artifact_manifest_sha256")
    if sha256_file(provenance_path) != expected_provenance_sha:
        raise RuntimeError("artifact manifest SHA-256 mismatch")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, Mapping):
        raise ValueError("artifact manifest must be a JSON object")
    provenance_model = provenance.get("model")
    if not isinstance(provenance_model, Mapping):
        raise ValueError("artifact manifest lacks model provenance")
    if provenance_model.get("sha256") != model_sha256:
        raise ValueError("artifact manifest/model SHA-256 mismatch")
    if int(provenance_model.get("num_features", -1)) != feature_count:
        raise ValueError("artifact manifest/model feature-count mismatch")
    provenance_runtime = provenance.get("runtime")
    if not isinstance(provenance_runtime, Mapping):
        raise ValueError("artifact manifest lacks pinned training runtime")
    if dict(provenance_runtime.get("required_packages", {})) != REQUIRED_PACKAGES:
        raise ValueError("artifact manifest training package pins changed")
    if dict(provenance_runtime.get("installed_packages", {})) != REQUIRED_PACKAGES:
        raise ValueError("artifact was not fitted with exact cloud package versions")
    if provenance_runtime.get("is_exact_match") is not True:
        raise ValueError("artifact manifest does not certify an exact pinned fit")
    if not isinstance(provenance.get("source_sha256"), Mapping) or not provenance[
        "source_sha256"
    ]:
        raise ValueError("artifact manifest lacks fitting-source provenance")


def _import_experiment(module_name: str):
    return importlib.import_module(f"experiments.{module_name}")


def _required_state_names(views: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for view in views:
        if view not in VIEW_FEATURE_COUNTS:
            raise ValueError(f"unknown feature view: {view}")
        if view == "v2":
            names.add("v2")
        elif view in {"v3_full", "v3_supplement"}:
            names.add("v3")
        else:
            names.update(("v2", "v3"))
        if "robust12" in view:
            names.add("robust")
        if "seq18" in view:
            names.add("sequential")
    return names


class SinglePassFeatureState:
    """Update each requested source state exactly once per online point."""

    def __init__(self, historical: Sequence[float] | np.ndarray, views: Sequence[str]):
        self.views = tuple(dict.fromkeys(views))
        required = _required_state_names(self.views)
        self.v2_module = None
        self.v3_module = None
        self.v4_module = None
        self.robust_module = None
        self.sequential_module = None
        self.v2_state = None
        self.v3_state = None
        self.robust_state = None
        self.sequential_state = None

        history = np.array(historical, dtype=np.float64, copy=True).reshape(-1)
        history.setflags(write=False)
        if "v2" in required:
            self.v2_module = _import_experiment("v2_candidate")
            self.v2_state = self.v2_module.StreamingFeaturesV2(history)
        if "v3" in required:
            self.v3_module = _import_experiment("v3_candidate")
            self.v3_state = self.v3_module.StreamingFeaturesV3(history)
        if any(view.startswith("v4") or view == "v3_supplement" for view in self.views):
            self.v4_module = _import_experiment("v4_hybrid")
        if "robust" in required:
            self.robust_module = _import_experiment("v5_robust_supplement")
            expected_indices = tuple(
                self.robust_module.FEATURE_NAMES.index(name) for name in ROBUST12_NAMES
            )
            if expected_indices != ROBUST12_LOCAL_INDICES:
                raise RuntimeError("robust12 source schema changed")
            self.robust_state = self.robust_module.StreamingRobustSupplement(history)
        if "sequential" in required:
            self.sequential_module = _import_experiment("v5_sequential_evidence")
            families = tuple(self.sequential_module.FAMILY_NAMES)
            width = int(self.sequential_module.SUMMARY_COUNT)
            indices: list[int] = []
            for family in SEQUENTIAL18_FAMILIES:
                family_index = families.index(family)
                indices.extend(range(family_index * width, (family_index + 1) * width))
            if tuple(indices) != SEQUENTIAL18_LOCAL_INDICES:
                raise RuntimeError("sequential18 source schema changed")
            self.sequential_state = self.sequential_module.StreamingSequentialEvidence(
                history
            )

    def update(self, raw_value: float) -> dict[str, np.ndarray]:
        rows: dict[str, np.ndarray] = {}
        v2_row = self.v2_state.update(raw_value) if self.v2_state is not None else None
        v3_row = self.v3_state.update(raw_value) if self.v3_state is not None else None
        robust_row = (
            self.robust_state.update(raw_value)
            if self.robust_state is not None
            else None
        )
        sequential_row = (
            self.sequential_state.update(raw_value)
            if self.sequential_state is not None
            else None
        )

        if "v2" in self.views:
            rows["v2"] = np.asarray(v2_row, dtype=np.float32)
        if "v3_full" in self.views:
            rows["v3_full"] = np.asarray(v3_row, dtype=np.float32)

        v3_supplement = None
        if "v3_supplement" in self.views or any(
            view.startswith("v4") for view in self.views
        ):
            if self.v4_module is None or v3_row is None:
                raise RuntimeError("v4 schema is unavailable")
            v3_supplement = np.take(
                v3_row,
                self.v4_module.V3_SUPPLEMENT_SOURCE_INDEX_ARRAY,
            ).astype(np.float32, copy=False)
        if "v3_supplement" in self.views:
            rows["v3_supplement"] = v3_supplement

        v4_row = None
        if any(view.startswith("v4") for view in self.views):
            if v2_row is None or v3_supplement is None:
                raise RuntimeError("v4 source states are unavailable")
            v4_row = np.concatenate((v2_row, v3_supplement)).astype(
                np.float32, copy=False
            )
        if "v4" in self.views:
            rows["v4"] = v4_row
        if "v4_robust12" in self.views:
            rows["v4_robust12"] = np.concatenate(
                (v4_row, np.take(robust_row, ROBUST12_LOCAL_INDICES))
            ).astype(np.float32, copy=False)
        if "v4_seq18" in self.views:
            rows["v4_seq18"] = np.concatenate(
                (v4_row, np.take(sequential_row, SEQUENTIAL18_LOCAL_INDICES))
            ).astype(np.float32, copy=False)
        if "v4_robust12_seq18" in self.views:
            rows["v4_robust12_seq18"] = np.concatenate(
                (
                    v4_row,
                    np.take(robust_row, ROBUST12_LOCAL_INDICES),
                    np.take(sequential_row, SEQUENTIAL18_LOCAL_INDICES),
                )
            ).astype(np.float32, copy=False)

        for view, row in rows.items():
            expected = VIEW_FEATURE_COUNTS[view]
            if row.shape != (expected,):
                raise RuntimeError(
                    f"feature width changed for {view}: {row.shape} != ({expected},)"
                )
            if not np.isfinite(row).all():
                raise RuntimeError(f"non-finite feature generated for {view}")
        return rows


def _validate_feature_runtime(views: Sequence[str]) -> None:
    # Fail before the readiness yield if a packaged module/schema has drifted.
    history = np.linspace(-1.0, 1.0, 300, dtype=np.float64)
    state = SinglePassFeatureState(history, views)
    produced = state.update(0.125)
    if set(produced) != set(views):
        raise RuntimeError("feature runtime did not produce every requested view")


@dataclass(frozen=True)
class LoadedComponent:
    name: str
    view: str
    weight: float
    booster: Any


@dataclass(frozen=True)
class LoadedResources:
    components: tuple[LoadedComponent, ...]
    views: tuple[str, ...]
    blend_space: str
    prediction_clip: float
    candidate_id: str


@dataclass(frozen=True)
class ValidatedComponent:
    """Pure-Python component metadata safe to construct before worker fork."""

    name: str
    view: str
    weight: float
    model_path: Path
    feature_count: int
    num_trees: int


@dataclass(frozen=True)
class ValidatedResources:
    components: tuple[ValidatedComponent, ...]
    views: tuple[str, ...]
    blend_space: str
    prediction_clip: float
    candidate_id: str


@dataclass(frozen=True)
class TextModelMetadata:
    feature_count: int
    num_trees: int


def _inspect_lightgbm_text_model(path: Path) -> TextModelMetadata:
    """Validate binary LightGBM text structure without importing LightGBM.

    Model SHA-256 remains authoritative.  This parser additionally proves the
    feature width and tree count recorded in the manifest while avoiding every
    LightGBM C-API/OpenMP call in the pre-fork parent process.
    """

    version: str | None = None
    num_class: int | None = None
    trees_per_iteration: int | None = None
    max_feature_index: int | None = None
    objective: str | None = None
    declared_tree_sizes: int | None = None
    declared_feature_names: int | None = None
    tree_count = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
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
                tree_index = int(line.split("=", 1)[1])
                if tree_index != tree_count:
                    raise ValueError(
                        f"non-contiguous LightGBM tree index {tree_index} "
                        f"at expected index {tree_count}: {path}"
                    )
                tree_count += 1
    if version != "v4":
        raise ValueError(f"unsupported LightGBM text-model version {version!r}: {path}")
    if num_class != 1 or trees_per_iteration != 1:
        raise ValueError(f"production model must be single-output binary: {path}")
    if objective is None or not objective.startswith("binary "):
        raise ValueError(f"production model objective is not binary: {path}")
    if max_feature_index is None or max_feature_index < 0:
        raise ValueError(f"LightGBM model lacks a valid max_feature_idx: {path}")
    feature_count = max_feature_index + 1
    if declared_feature_names != feature_count:
        raise ValueError(f"LightGBM feature-name width changed: {path}")
    if tree_count <= 0 or declared_tree_sizes != tree_count:
        raise ValueError(f"LightGBM tree-count structure changed: {path}")
    return TextModelMetadata(feature_count=feature_count, num_trees=tree_count)


def validate_resources(
    model_directory_path: str | Path,
    *,
    check_runtime: bool = True,
) -> ValidatedResources:
    """Validate immutable resources without importing or calling LightGBM."""

    resource_root, manifest = _read_manifest(model_directory_path)
    _validate_top_level_manifest(manifest)
    if check_runtime:
        _validate_runtime_versions()
    _validate_source_files(manifest)

    raw_components = manifest.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("production manifest must contain at least one component")
    validated: list[ValidatedComponent] = []
    seen_names: set[str] = set()
    weights: list[float] = []
    views: list[str] = []
    for raw in raw_components:
        if not isinstance(raw, Mapping):
            raise ValueError("component must be a JSON object")
        name = raw.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("component names must be non-empty and unique")
        seen_names.add(name)
        view = raw.get("feature_view")
        if view not in VIEW_FEATURE_COUNTS:
            raise ValueError(f"unknown component feature view: {view}")
        expected_features = VIEW_FEATURE_COUNTS[view]
        if int(raw.get("feature_count", -1)) != expected_features:
            raise ValueError(f"manifest feature count mismatch for {name}")
        expected_trees = int(raw.get("num_trees", -1))
        if expected_trees <= 0:
            raise ValueError(f"manifest tree count is invalid for {name}")
        weight = float(raw.get("weight", math.nan))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"component weight must be finite and positive: {name}")
        model_path = _resolved_child(
            resource_root, raw.get("model_file"), label="model file"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"missing pretrained model: {model_path}")
        actual_sha = sha256_file(model_path)
        if actual_sha != raw.get("sha256"):
            raise RuntimeError(f"pretrained model SHA-256 mismatch: {name}")
        text_metadata = _inspect_lightgbm_text_model(model_path)
        if text_metadata.feature_count != expected_features:
            raise RuntimeError(
                f"pretrained model width {text_metadata.feature_count} != "
                f"{expected_features} for {name}"
            )
        if text_metadata.num_trees != expected_trees:
            raise RuntimeError(
                f"pretrained model tree count {text_metadata.num_trees} != "
                f"{expected_trees} for {name}"
            )
        _validate_provenance(
            resource_root,
            raw,
            model_sha256=actual_sha,
            feature_count=expected_features,
        )
        validated.append(
            ValidatedComponent(
                name=name,
                view=str(view),
                weight=weight,
                model_path=model_path,
                feature_count=expected_features,
                num_trees=expected_trees,
            )
        )
        weights.append(weight)
        views.append(str(view))

    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("component weights must sum to exactly one within 1e-12")
    blend = manifest["blend"]
    return ValidatedResources(
        components=tuple(validated),
        views=tuple(dict.fromkeys(views)),
        blend_space=str(blend["space"]),
        prediction_clip=float(blend["prediction_clip"]),
        candidate_id=str(manifest.get("candidate_id", "")),
    )


def load_resources(
    model_directory_path: str | Path,
    *,
    check_runtime: bool = True,
) -> LoadedResources:
    validated = validate_resources(
        model_directory_path, check_runtime=check_runtime
    )
    # This is the sole LightGBM import path in production.  ``infer_impl`` is
    # entered inside each forked worker, so every worker initializes its own
    # Booster/OpenMP state rather than inheriting it from the parent.
    lgb = importlib.import_module("lightgbm")
    loaded: list[LoadedComponent] = []
    for component in validated.components:
        booster = lgb.Booster(model_file=str(component.model_path))
        if booster.num_feature() != component.feature_count:
            raise RuntimeError(
                f"pretrained model width {booster.num_feature()} != "
                f"{component.feature_count} for {component.name}"
            )
        if booster.num_trees() != component.num_trees:
            raise RuntimeError(
                f"pretrained model tree count {booster.num_trees()} != "
                f"{component.num_trees} for {component.name}"
            )
        loaded.append(
            LoadedComponent(
                component.name,
                component.view,
                component.weight,
                booster,
            )
        )

    _validate_feature_runtime(validated.views)
    return LoadedResources(
        components=tuple(loaded),
        views=validated.views,
        blend_space=validated.blend_space,
        prediction_clip=validated.prediction_clip,
        candidate_id=validated.candidate_id,
    )


def _blend_predictions(
    predictions: Sequence[float],
    weights: Sequence[float],
    *,
    space: str,
    clip: float,
) -> float:
    values = np.clip(np.asarray(predictions, dtype=np.float64), clip, 1.0 - clip)
    weight_array = np.asarray(weights, dtype=np.float64)
    if space == "probability":
        result = float(weight_array @ values)
    elif space == "logit":
        logits = np.log(values) - np.log1p(-values)
        blended_logit = float(weight_array @ logits)
        if blended_logit >= 0.0:
            result = 1.0 / (1.0 + math.exp(-blended_logit))
        else:
            exponential = math.exp(blended_logit)
            result = exponential / (1.0 + exponential)
    else:
        raise ValueError(f"unknown blend space: {space}")
    return min(max(result, clip), 1.0 - clip)


def train_impl(datasets, model_directory_path: str) -> None:
    """Pre-fork no-op train: validate resources without importing LightGBM."""

    del datasets
    validate_resources(model_directory_path, check_runtime=True)


def infer_impl(
    datasets: Iterable[tuple[Sequence[float], Iterable[float]]],
    model_directory_path: str,
    *,
    _check_runtime: bool = True,
):
    resources = load_resources(
        model_directory_path, check_runtime=_check_runtime
    )
    weights = tuple(component.weight for component in resources.components)
    yield
    for historical, online in datasets:
        state = SinglePassFeatureState(historical, resources.views)
        for point in online:
            rows = state.update(point)
            predictions = [
                float(
                    component.booster.predict(
                        rows[component.view].reshape(1, -1), num_threads=1
                    )[0]
                )
                for component in resources.components
            ]
            if not np.isfinite(predictions).all():
                raise RuntimeError("pretrained component produced a non-finite score")
            yield _blend_predictions(
                predictions,
                weights,
                space=resources.blend_space,
                clip=resources.prediction_clip,
            )
