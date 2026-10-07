"""Fail-closed NumPy runtime for the ADIA v13 standalone ExtraTrees model.

``train_impl`` performs validation only.  In particular, it never constructs a
feature state and therefore never imports the experiment modules (whose module
initialisation imports LightGBM).  The flat forest and causal ``v4_seq18``
feature state are loaded inside an inference worker before its readiness yield.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np

try:  # Package import in the repository.
    from . import production_runtime as feature_runtime
except ImportError:  # Top-level import after the files are copied into a bundle.
    import production_runtime as feature_runtime  # type: ignore[no-redef]


COMPETITION = "ADIA Structural Break Challenge: Real-Time Edition"
CANDIDATE_ID = "v13_extratrees_standalone_fullfit"
MANIFEST_FILENAME = "production_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
FORMAT_NAME = "v13_flat_extratrees_npz"
FORMAT_VERSION = 1
FEATURE_VIEW = "v4_seq18"
FEATURE_COUNT = 440
EXPECTED_TREE_COUNT = 400
PREDICTION_CLIP = 1e-7
EXPECTED_INFER_PARALLELISM = 4
TREE_ACCUMULATION = "fixed_tree_order_float64_left_to_right_mean"
TRAIN_MODE = "validation-only-no-op"
REQUIRED_PACKAGES = {"lightgbm": "4.6.0", "numpy": "2.3.3"}
CANONICAL_REQUIREMENTS_SHA256 = (
    "b1b54bb65f8928926774e357a55b16c2a95349ae16850eb20b5c985c73e65862"
)
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "tree_roots": np.dtype("<i8"),
    "children_left": np.dtype("<i8"),
    "children_right": np.dtype("<i8"),
    "feature": np.dtype("<i4"),
    "threshold": np.dtype("<f8"),
    "positive_probability": np.dtype("<f8"),
    "classes": np.dtype("<i8"),
}
SOURCE_PATHS = frozenset(
    {
        "main.py",
        "extratrees_runtime.py",
        "production_runtime.py",
        "experiments/__init__.py",
        "experiments/v2_candidate.py",
        "experiments/v3_candidate.py",
        "experiments/v4_hybrid.py",
        "experiments/v5_sequential_evidence.py",
    }
)

_MAX_LEAVES_PER_TREE = 500_000 // 200
MAX_NODES_PER_TREE = 2 * _MAX_LEAVES_PER_TREE - 1
MAX_TOTAL_NODES = EXPECTED_TREE_COUNT * MAX_NODES_PER_TREE
MAX_NPZ_BYTES = 256 * 1024 * 1024
_SHA256_LENGTH = 64

# This is an alias, not a second feature implementation.
SinglePassFeatureState = feature_runtime.SinglePassFeatureState


@dataclass(frozen=True)
class FlatForest:
    tree_roots: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    positive_probability: np.ndarray
    classes: np.ndarray
    max_traversal_steps: int

    @property
    def tree_count(self) -> int:
        return int(self.tree_roots.size)

    @property
    def node_count(self) -> int:
        return int(self.children_left.size)

    def predict_one(self, features: Sequence[float] | np.ndarray) -> float:
        """Traverse all trees together, then average in fixed tree order."""

        row = np.asarray(features)
        if row.shape != (FEATURE_COUNT,):
            raise ValueError(f"features must have shape ({FEATURE_COUNT},)")
        if row.dtype != np.dtype("float32"):
            row = np.asarray(row, dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError("flat-forest features must be finite")

        nodes = self.tree_roots.copy()
        for _ in range(self.max_traversal_steps + 1):
            left = self.children_left[nodes]
            active = left >= 0
            if not bool(np.any(active)):
                break
            positions = np.flatnonzero(active)
            current = nodes[positions]
            columns = self.feature[current]
            go_left = row[columns] <= self.threshold[current]
            nodes[positions] = np.where(
                go_left,
                self.children_left[current],
                self.children_right[current],
            )
        else:
            raise RuntimeError("flat-forest traversal did not terminate")

        if np.any(self.children_left[nodes] != -1):
            raise RuntimeError("flat-forest traversal ended on an internal node")
        leaves = self.positive_probability[nodes]
        if not np.isfinite(leaves).all():
            raise RuntimeError("flat-forest produced a non-finite leaf probability")
        total = 0.0
        for tree_index in range(self.tree_count):
            total += float(leaves[tree_index])
        result = total / float(self.tree_count)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise RuntimeError("flat-forest produced an invalid probability")
        return result


@dataclass(frozen=True)
class ValidatedResources:
    candidate_id: str
    model_path: Path
    provenance_path: Path
    model_sha256: str
    model_size_bytes: int
    array_records: Mapping[str, Any]

    @property
    def tree_count(self) -> int:
        return int(self.array_records["tree_roots"]["shape"][0])

    @property
    def feature_count(self) -> int:
        return FEATURE_COUNT


@dataclass(frozen=True)
class LoadedResources:
    candidate_id: str
    forest: FlatForest
    prediction_clip: float


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def _resolved_child(root: Path, value: Any, *, label: str) -> Path:
    relative = _safe_relative_path(value, label=label)
    resolved_root = root.resolve()
    path = (resolved_root / Path(*relative.parts)).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"{label} escapes its resource root")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _source_path(relative_name: str) -> Path:
    code_root = Path(__file__).resolve().parent
    candidates = (code_root / relative_name, code_root.parent / relative_name)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"missing packaged source: {relative_name}")


def _validate_source_files(manifest: Mapping[str, Any]) -> None:
    records = manifest.get("source_sha256")
    if not isinstance(records, Mapping) or set(records) != set(SOURCE_PATHS):
        raise ValueError("runtime source SHA-256 inventory changed")
    for relative_name in sorted(SOURCE_PATHS):
        expected = records.get(relative_name)
        if not _is_sha256(expected):
            raise ValueError(f"invalid source SHA-256: {relative_name}")
        if sha256_file(_source_path(relative_name)) != expected:
            raise RuntimeError(f"packaged source SHA-256 mismatch: {relative_name}")


def _validate_runtime_versions() -> None:
    # Package metadata lookup does not import LightGBM or any feature module.
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


def _array_record(record: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest array record is missing: {name}")
    dtype = record.get("dtype")
    shape = record.get("shape")
    digest = record.get("sha256")
    if dtype != ARRAY_DTYPES[name].str:
        raise ValueError(f"manifest array dtype changed: {name}")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] <= 0
    ):
        raise ValueError(f"manifest array shape is invalid: {name}")
    if not _is_sha256(digest):
        raise ValueError(f"manifest array SHA-256 is invalid: {name}")
    return {"dtype": dtype, "shape": list(shape), "sha256": digest}


def _validate_array_records(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping) or set(raw) != set(ARRAY_DTYPES):
        raise ValueError("manifest flat-forest array inventory changed")
    records = {
        name: _array_record(raw[name], name=name) for name in ARRAY_DTYPES
    }
    if records["tree_roots"]["shape"] != [EXPECTED_TREE_COUNT]:
        raise ValueError("manifest tree-root count changed")
    if records["classes"]["shape"] != [2]:
        raise ValueError("manifest class shape changed")
    node_count = records["children_left"]["shape"][0]
    if not 1 <= node_count <= MAX_TOTAL_NODES:
        raise ValueError("manifest node count is outside the frozen bound")
    for name in (
        "children_right",
        "feature",
        "threshold",
        "positive_probability",
    ):
        if records[name]["shape"] != [node_count]:
            raise ValueError(f"manifest node-array length changed: {name}")
    return records


def _inspect_npz(path: Path) -> None:
    if path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("flat-forest NPZ exceeds the size bound")
    expected = {f"{name}.npy" for name in ARRAY_DTYPES}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("flat-forest NPZ member inventory changed")
            if any(
                member.is_dir()
                or member.flag_bits & 0x1
                or member.compress_type != zipfile.ZIP_STORED
                for member in members
            ):
                raise ValueError("flat-forest NPZ zip structure is unsupported")
            if sum(member.file_size for member in members) > MAX_NPZ_BYTES:
                raise ValueError("flat-forest NPZ expands beyond the size bound")
            if archive.testzip() is not None:
                raise ValueError("flat-forest NPZ failed its CRC check")
    except zipfile.BadZipFile as error:
        raise ValueError("flat-forest NPZ is not a valid zip archive") from error


def _validate_forest_structure(arrays: Mapping[str, np.ndarray]) -> int:
    roots = arrays["tree_roots"]
    left = arrays["children_left"]
    right = arrays["children_right"]
    feature = arrays["feature"]
    threshold = arrays["threshold"]
    probability = arrays["positive_probability"]
    classes = arrays["classes"]

    if not np.array_equal(classes, np.asarray([0, 1], dtype="<i8")):
        raise ValueError("flat-forest classes must be exactly [0, 1]")
    node_count = int(left.size)
    if (
        roots.size != EXPECTED_TREE_COUNT
        or int(roots[0]) != 0
        or np.any(np.diff(roots) <= 0)
        or int(roots[-1]) >= node_count
    ):
        raise ValueError("flat-forest tree roots are invalid")
    if not np.isfinite(threshold).all():
        raise ValueError("flat-forest threshold contains a non-finite value")
    leaves = left == -1
    if not np.array_equal(leaves, right == -1):
        raise ValueError("flat-forest leaf child sentinels differ")
    if np.any(feature[leaves] != -2):
        raise ValueError("flat-forest leaf feature sentinel changed")
    internal = ~leaves
    if np.any(feature[internal] < 0) or np.any(feature[internal] >= FEATURE_COUNT):
        raise ValueError("flat-forest internal feature index is invalid")
    if not np.isfinite(probability[leaves]).all():
        raise ValueError("flat-forest leaf probability contains a non-finite value")
    if np.any((probability[leaves] < 0.0) | (probability[leaves] > 1.0)):
        raise ValueError("flat-forest leaf probability is outside [0, 1]")
    if not np.isnan(probability[internal]).all():
        raise ValueError("flat-forest internal probability must be NaN")

    maximum_nodes = 0
    for tree_index in range(EXPECTED_TREE_COUNT):
        first = int(roots[tree_index])
        last = (
            int(roots[tree_index + 1])
            if tree_index + 1 < EXPECTED_TREE_COUNT
            else node_count
        )
        tree_nodes = last - first
        if not 1 <= tree_nodes <= MAX_NODES_PER_TREE:
            raise ValueError("flat-forest per-tree node count is invalid")
        maximum_nodes = max(maximum_nodes, tree_nodes)
        tree_left = left[first:last]
        tree_right = right[first:last]
        tree_internal = tree_left >= 0
        if not np.any(~tree_internal):
            raise ValueError("flat-forest tree contains no leaf")
        parent_indices = np.arange(first, last, dtype=np.int64)[tree_internal]
        for children in (tree_left[tree_internal], tree_right[tree_internal]):
            if (
                np.any(children < first)
                or np.any(children >= last)
                or np.any(children <= parent_indices)
            ):
                raise ValueError("flat-forest child index is invalid")
        child_indices = np.concatenate(
            (tree_left[tree_internal], tree_right[tree_internal])
        )
        parent_count = np.bincount(child_indices - first, minlength=tree_nodes)
        if parent_count[0] != 0 or np.any(parent_count[1:] != 1):
            raise ValueError("flat-forest nodes are not one rooted tree")
    return maximum_nodes


def _load_flat_forest(
    path: Path, records: Mapping[str, Mapping[str, Any]]
) -> FlatForest:
    _inspect_npz(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_DTYPES):
                raise ValueError("flat-forest NPZ array inventory changed")
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in ARRAY_DTYPES
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("flat-forest"):
            raise
        raise ValueError("flat-forest NPZ array load failed") from error

    for name, expected_dtype in ARRAY_DTYPES.items():
        value = arrays[name]
        record = records[name]
        if value.dtype != expected_dtype:
            raise ValueError(f"flat-forest dtype changed: {name}")
        if list(value.shape) != record["shape"]:
            raise ValueError(f"flat-forest shape changed: {name}")
        if array_sha256(value) != record["sha256"]:
            raise RuntimeError(f"flat-forest array SHA-256 mismatch: {name}")
        value.setflags(write=False)
    maximum_nodes = _validate_forest_structure(arrays)
    return FlatForest(
        tree_roots=arrays["tree_roots"],
        children_left=arrays["children_left"],
        children_right=arrays["children_right"],
        feature=arrays["feature"],
        threshold=arrays["threshold"],
        positive_probability=arrays["positive_probability"],
        classes=arrays["classes"],
        max_traversal_steps=maximum_nodes,
    )


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    manifest_path: Path,
    model_path: Path,
    model_sha256: str,
    model_size_bytes: int,
    array_records: Mapping[str, Any],
) -> None:
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("authorized", True),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
        ("stage", "standalone_all_series_fullfit_flat_export"),
        ("feature_view", FEATURE_VIEW),
        ("feature_count", FEATURE_COUNT),
        ("tree_count", EXPECTED_TREE_COUNT),
        ("classes", [0, 1]),
        ("npz_file", model_path.name),
        ("npz_sha256", model_sha256),
    ):
        _require_exact(provenance.get(key), expected, f"artifact {key}")
    _require_exact(provenance.get("arrays"), dict(array_records), "artifact arrays")

    flat = provenance.get("flat_forest")
    if not isinstance(flat, Mapping):
        raise ValueError("artifact manifest lacks flat-forest provenance")
    for key, expected in (
        ("format", FORMAT_NAME),
        ("format_version", FORMAT_VERSION),
        ("sklearn_runtime_dependency", False),
        ("filename", model_path.name),
        ("file_sha256", model_sha256),
        ("file_size_bytes", model_size_bytes),
        ("arrays", dict(array_records)),
        ("global_child_indices", True),
        ("threshold_dtype", "float64"),
        ("leaf_probability_dtype", "float64"),
        ("tree_order", "sklearn estimator order, ascending estimator index"),
        ("branch_semantics", "float32 feature <= float64 threshold goes left"),
    ):
        _require_exact(flat.get(key), expected, f"artifact flat_forest.{key}")

    feature_contract = provenance.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise ValueError("artifact manifest lacks feature contract")
    _require_exact(
        feature_contract.get("causal_single_pass_required_at_inference"),
        True,
        "artifact causal feature contract",
    )
    _require_exact(
        feature_contract.get("future_ticks_used"),
        False,
        "artifact future-tick contract",
    )
    parity = provenance.get("export_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError("artifact export parity is not certified")
    difference = parity.get("max_abs_difference")
    if (
        isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or not math.isfinite(float(difference))
        or float(difference) > 1e-12
    ):
        raise ValueError("artifact export parity exceeds 1e-12")
    environment = provenance.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("artifact manifest lacks fit environment")
    _require_exact(environment.get("numpy"), "2.3.3", "artifact NumPy version")
    _require_exact(
        environment.get("scikit_learn"), "1.9.0", "artifact sklearn version"
    )
    outputs = provenance.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("artifact manifest lacks outputs")
    _require_exact(outputs.get("forest_npz"), model_path.name, "artifact forest output")
    _require_exact(
        outputs.get("manifest_json"), manifest_path.name, "artifact manifest output"
    )


def _validate_bundle(
    model_directory_path: str | Path,
    *,
    check_runtime: bool,
) -> tuple[ValidatedResources, FlatForest]:
    resource_root = Path(model_directory_path).expanduser().resolve()
    manifest = _read_json(resource_root / MANIFEST_FILENAME, "production manifest")
    for key, expected in (
        ("schema_version", MANIFEST_SCHEMA_VERSION),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("candidate_id", CANDIDATE_ID),
    ):
        _require_exact(manifest.get(key), expected, f"production {key}")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("production manifest lacks runtime contract")
    for key, expected in (
        ("packages", REQUIRED_PACKAGES),
        ("requirements_sha256", CANONICAL_REQUIREMENTS_SHA256),
        ("infer_parallelism", EXPECTED_INFER_PARALLELISM),
        ("train_mode", TRAIN_MODE),
        ("prediction_clip", PREDICTION_CLIP),
        ("tree_accumulation", TREE_ACCUMULATION),
    ):
        _require_exact(runtime.get(key), expected, f"runtime {key}")
    if check_runtime:
        _validate_runtime_versions()
    _validate_source_files(manifest)

    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("production manifest lacks model contract")
    for key, expected in (
        ("format", FORMAT_NAME),
        ("format_version", FORMAT_VERSION),
        ("feature_view", FEATURE_VIEW),
        ("feature_count", FEATURE_COUNT),
        ("tree_count", EXPECTED_TREE_COUNT),
        ("classes", [0, 1]),
    ):
        _require_exact(model.get(key), expected, f"model {key}")
    records = _validate_array_records(model.get("arrays"))
    model_path = _resolved_child(resource_root, model.get("file"), label="model file")
    if not model_path.is_file():
        raise FileNotFoundError(f"missing flat forest: {model_path}")
    model_sha = sha256_file(model_path)
    if not _is_sha256(model.get("sha256")) or model_sha != model.get("sha256"):
        raise RuntimeError("flat-forest NPZ SHA-256 mismatch")
    model_size = int(model_path.stat().st_size)
    if type(model.get("size_bytes")) is not int or model.get("size_bytes") != model_size:
        raise ValueError("flat-forest NPZ size changed")

    provenance_path = _resolved_child(
        resource_root,
        model.get("artifact_manifest_file"),
        label="artifact manifest file",
    )
    provenance_sha = model.get("artifact_manifest_sha256")
    if not _is_sha256(provenance_sha):
        raise ValueError("artifact manifest SHA-256 is invalid")
    if sha256_file(provenance_path) != provenance_sha:
        raise RuntimeError("artifact manifest SHA-256 mismatch")
    provenance = _read_json(provenance_path, "artifact manifest")
    _validate_provenance(
        provenance,
        manifest_path=provenance_path,
        model_path=model_path,
        model_sha256=model_sha,
        model_size_bytes=model_size,
        array_records=records,
    )
    forest = _load_flat_forest(model_path, records)
    validated = ValidatedResources(
        candidate_id=CANDIDATE_ID,
        model_path=model_path,
        provenance_path=provenance_path,
        model_sha256=model_sha,
        model_size_bytes=model_size,
        array_records=records,
    )
    return validated, forest


def validate_resources(
    model_directory_path: str | Path,
    *,
    check_runtime: bool = True,
) -> ValidatedResources:
    """Pure pre-fork validation; imports no model or feature library."""

    validated, _ = _validate_bundle(
        model_directory_path, check_runtime=check_runtime
    )
    return validated


def load_resources(
    model_directory_path: str | Path,
    *,
    check_runtime: bool = True,
) -> LoadedResources:
    """Load and authenticate the numeric forest inside an inference worker."""

    validated, forest = _validate_bundle(
        model_directory_path, check_runtime=check_runtime
    )
    return LoadedResources(
        candidate_id=validated.candidate_id,
        forest=forest,
        prediction_clip=PREDICTION_CLIP,
    )


def train_impl(
    datasets: Any,
    model_directory_path: str,
    *,
    _check_runtime: bool = True,
) -> None:
    """Validate immutable resources without consuming data or importing features."""

    del datasets
    validate_resources(model_directory_path, check_runtime=_check_runtime)


def infer_impl(
    datasets: Iterable[tuple[Sequence[float], Iterable[float]]],
    model_directory_path: str,
    *,
    _check_runtime: bool = True,
):
    """Yield readiness, then one current-prefix probability for every tick."""

    resources = load_resources(
        model_directory_path, check_runtime=_check_runtime
    )
    # This deliberately runs in the inference worker and before readiness.
    feature_runtime._validate_feature_runtime((FEATURE_VIEW,))  # noqa: SLF001
    yield
    for historical, online in datasets:
        state = SinglePassFeatureState(historical, (FEATURE_VIEW,))
        for point in online:
            row = state.update(point)[FEATURE_VIEW]
            prediction = resources.forest.predict_one(row)
            yield min(
                max(prediction, resources.prediction_clip),
                1.0 - resources.prediction_clip,
            )


INFER_PARALLELISM = EXPECTED_INFER_PARALLELISM
