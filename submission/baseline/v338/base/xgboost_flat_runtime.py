"""NumPy-only runtime for a flattened XGBoost 3.0.4 binary ``gbtree``.

The export format deliberately contains only primitive numeric arrays.  The
runtime has no XGBoost dependency and implements the frozen prediction path:
float32 feature comparison, exact missing/default-left routing, fixed JSON
tree order, float64 left-to-right margin accumulation, and the binary-logistic
link applied to the persisted base margin plus leaf contributions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


COMPETITION = "ADIA Structural Break Challenge: Real-Time Edition"
FORMAT_NAME = "v13_flat_xgboost_binary_npz"
FORMAT_VERSION = 1
SOURCE_XGBOOST_VERSION = "3.0.4"
OBJECTIVE = "binary:logistic"
OUTPUT_LINK = "logistic"
MISSING_VALUE = "NaN"
BRANCH_SEMANTICS = (
    "missing follows default_left; otherwise float32 feature < float32 "
    "split_condition goes left"
)
TREE_ORDER = "XGBoost model JSON trees order, ascending tree index"
TREE_ACCUMULATION = "base_margin then fixed-tree-order float64 left-to-right sum"
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "tree_roots": np.dtype("<i8"),
    "iteration_indptr": np.dtype("<i8"),
    "children_left": np.dtype("<i8"),
    "children_right": np.dtype("<i8"),
    "split_index": np.dtype("<i4"),
    "split_condition": np.dtype("<f4"),
    "default_left": np.dtype("u1"),
    "leaf_weight": np.dtype("<f4"),
    "base_margin": np.dtype("<f8"),
}
MAX_TREES = 10_000
MAX_TOTAL_NODES = 10_000_000
MAX_NPZ_BYTES = 512 * 1024 * 1024
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class FlatXGBoostBinary:
    tree_roots: np.ndarray
    iteration_indptr: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    split_index: np.ndarray
    split_condition: np.ndarray
    default_left: np.ndarray
    leaf_weight: np.ndarray
    base_margin: float
    feature_count: int
    max_depth: int

    @property
    def tree_count(self) -> int:
        return int(self.tree_roots.size)

    @property
    def node_count(self) -> int:
        return int(self.children_left.size)

    def predict_margin_one(self, features: Sequence[float] | np.ndarray) -> float:
        """Return one raw margin using deterministic float64 accumulation."""

        row = np.asarray(features)
        if row.shape != (self.feature_count,):
            raise ValueError(f"features must have shape ({self.feature_count},)")
        if row.dtype != np.dtype("float32"):
            row = np.asarray(row, dtype=np.float32)
        if np.isinf(row).any():
            raise ValueError("flat XGBoost features contain infinity")

        nodes = self.tree_roots.copy()
        for _ in range(self.max_depth + 1):
            left = self.children_left[nodes]
            active = left >= 0
            if not bool(np.any(active)):
                break
            positions = np.flatnonzero(active)
            current = nodes[positions]
            columns = self.split_index[current]
            values = row[columns]
            missing = np.isnan(values)
            go_left = np.where(
                missing,
                self.default_left[current] == 1,
                values < self.split_condition[current],
            )
            nodes[positions] = np.where(
                go_left,
                self.children_left[current],
                self.children_right[current],
            )
        else:
            raise RuntimeError("flat XGBoost traversal did not terminate")

        if np.any(self.children_left[nodes] != -1):
            raise RuntimeError("flat XGBoost traversal ended on an internal node")
        leaves = self.leaf_weight[nodes]
        if not np.isfinite(leaves).all():
            raise RuntimeError("flat XGBoost produced a non-finite leaf weight")
        margin = float(self.base_margin)
        for tree_index in range(self.tree_count):
            margin += float(leaves[tree_index])
        if not math.isfinite(margin):
            raise RuntimeError("flat XGBoost produced a non-finite margin")
        return margin

    def predict_one(self, features: Sequence[float] | np.ndarray) -> float:
        """Return one binary-logistic probability."""

        margin = self.predict_margin_one(features)
        if margin >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-margin))
        else:
            exponential = math.exp(margin)
            probability = exponential / (1.0 + exponential)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("flat XGBoost produced an invalid probability")
        return probability

    def predict_margin(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise ValueError(
                f"features must have shape (rows, {self.feature_count})"
            )
        output = np.empty(matrix.shape[0], dtype=np.float64)
        for row_index in range(matrix.shape[0]):
            output[row_index] = self.predict_margin_one(matrix[row_index])
        return output

    def predict(self, features: np.ndarray) -> np.ndarray:
        margins = self.predict_margin(features)
        output = np.empty_like(margins)
        nonnegative = margins >= 0.0
        output[nonnegative] = 1.0 / (1.0 + np.exp(-margins[nonnegative]))
        exponential = np.exp(margins[~nonnegative])
        output[~nonnegative] = exponential / (1.0 + exponential)
        return output


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact(actual: Any, expected: Any, description: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{description} changed: expected {expected!r}, got {actual!r}"
        )


def array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError("flat XGBoost array inventory changed")
    return {
        name: {
            "dtype": np.asarray(arrays[name]).dtype.str,
            "shape": list(np.asarray(arrays[name]).shape),
            "sha256": array_sha256(np.asarray(arrays[name])),
        }
        for name in ARRAY_DTYPES
    }


def _validate_array_record(
    value: Any, *, name: str, expected_shape: Sequence[int]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"flat XGBoost array record is missing: {name}")
    _require_exact(value.get("dtype"), ARRAY_DTYPES[name].str, f"{name} dtype")
    _require_exact(value.get("shape"), list(expected_shape), f"{name} shape")
    digest = value.get("sha256")
    if not _is_sha256(digest):
        raise ValueError(f"flat XGBoost array hash is invalid: {name}")
    return {
        "dtype": ARRAY_DTYPES[name].str,
        "shape": list(expected_shape),
        "sha256": digest,
    }


def validate_flat_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_feature_count: int | None = None,
    expected_tree_count: int | None = None,
) -> dict[str, int]:
    """Fail closed on the complete flat-tree structure and return its bounds."""

    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError("flat XGBoost array inventory changed")
    normalized: dict[str, np.ndarray] = {}
    for name, expected_dtype in ARRAY_DTYPES.items():
        value = np.asarray(arrays[name])
        if value.dtype != expected_dtype:
            raise ValueError(
                f"flat XGBoost {name} dtype {value.dtype.str} != "
                f"{expected_dtype.str}"
            )
        if value.ndim != 1:
            raise ValueError(f"flat XGBoost {name} must be one-dimensional")
        normalized[name] = value

    roots = normalized["tree_roots"]
    indptr = normalized["iteration_indptr"]
    left = normalized["children_left"]
    right = normalized["children_right"]
    split_index = normalized["split_index"]
    split_condition = normalized["split_condition"]
    default_left = normalized["default_left"]
    leaf_weight = normalized["leaf_weight"]
    base_margin = normalized["base_margin"]
    tree_count = int(roots.size)
    node_count = int(left.size)
    if not 1 <= tree_count <= MAX_TREES:
        raise ValueError("flat XGBoost tree count is outside the safety bound")
    if expected_tree_count is not None and tree_count != expected_tree_count:
        raise ValueError("flat XGBoost tree count changed")
    if not tree_count <= node_count <= MAX_TOTAL_NODES:
        raise ValueError("flat XGBoost node count is outside the safety bound")
    if roots.shape != (tree_count,) or roots[0] != 0:
        raise ValueError("flat XGBoost roots are invalid")
    if tree_count > 1 and np.any(roots[1:] <= roots[:-1]):
        raise ValueError("flat XGBoost roots are not strictly increasing")
    if np.any(roots < 0) or np.any(roots >= node_count):
        raise ValueError("flat XGBoost root is outside the node arrays")
    if indptr.shape != (tree_count + 1,):
        raise ValueError("flat XGBoost iteration_indptr shape changed")
    if indptr[0] != 0 or indptr[-1] != tree_count:
        raise ValueError("flat XGBoost iteration_indptr endpoints changed")
    if np.any(np.diff(indptr) != 1):
        raise ValueError("flat XGBoost requires exactly one tree per iteration")
    if base_margin.shape != (1,) or not np.isfinite(base_margin).all():
        raise ValueError("flat XGBoost base margin is invalid")
    for name in (
        "children_right",
        "split_index",
        "split_condition",
        "default_left",
        "leaf_weight",
    ):
        if normalized[name].shape != (node_count,):
            raise ValueError(f"flat XGBoost node-array length changed: {name}")

    leaf = left == -1
    internal = ~leaf
    if np.any(right[leaf] != -1) or np.any(right[internal] < 0):
        raise ValueError("flat XGBoost child sentinels differ")
    if not np.any(leaf):
        raise ValueError("flat XGBoost contains no leaf")
    if np.any(split_index[leaf] != -1):
        raise ValueError("flat XGBoost leaf feature sentinel changed")
    if np.any(~np.isnan(split_condition[leaf])):
        raise ValueError("flat XGBoost leaf threshold must be NaN")
    if np.any(~np.isnan(leaf_weight[internal])):
        raise ValueError("flat XGBoost internal leaf weight must be NaN")
    if np.any(~np.isfinite(leaf_weight[leaf])):
        raise ValueError("flat XGBoost leaf weight is non-finite")
    if np.any(~np.isfinite(split_condition[internal])):
        raise ValueError("flat XGBoost split threshold is non-finite")
    if np.any((default_left != 0) & (default_left != 1)):
        raise ValueError("flat XGBoost default_left is not binary")
    if np.any(default_left[leaf] != 0):
        raise ValueError("flat XGBoost leaf default_left sentinel changed")
    if expected_feature_count is not None:
        if expected_feature_count <= 0:
            raise ValueError("expected feature count must be positive")
        if np.any(
            (split_index[internal] < 0)
            | (split_index[internal] >= expected_feature_count)
        ):
            raise ValueError("flat XGBoost internal feature index is invalid")
        feature_count = int(expected_feature_count)
    else:
        if np.any(split_index[internal] < 0):
            raise ValueError("flat XGBoost internal feature index is negative")
        feature_count = int(split_index[internal].max(initial=-1)) + 1
        if feature_count <= 0:
            raise ValueError("flat XGBoost feature count cannot be inferred")

    maximum_depth = 0
    for tree_index in range(tree_count):
        first = int(roots[tree_index])
        last = int(roots[tree_index + 1]) if tree_index + 1 < tree_count else node_count
        if not first < last:
            raise ValueError("flat XGBoost tree boundary is empty")
        tree_nodes = last - first
        parent_count = np.zeros(tree_nodes, dtype=np.int32)
        stack: list[tuple[int, int]] = [(first, 0)]
        visited: set[int] = set()
        while stack:
            node, depth = stack.pop()
            if node in visited:
                raise ValueError("flat XGBoost tree contains a cycle")
            visited.add(node)
            maximum_depth = max(maximum_depth, depth)
            if left[node] == -1:
                continue
            for child in (int(left[node]), int(right[node])):
                if child <= node or child < first or child >= last:
                    raise ValueError("flat XGBoost child crosses a tree boundary")
                parent_count[child - first] += 1
                stack.append((child, depth + 1))
        if len(visited) != tree_nodes:
            raise ValueError("flat XGBoost tree has unreachable nodes")
        if parent_count[0] != 0 or np.any(parent_count[1:] != 1):
            raise ValueError("flat XGBoost nodes are not one rooted tree")

    return {
        "tree_count": tree_count,
        "node_count": node_count,
        "feature_count": feature_count,
        "max_depth": maximum_depth,
    }


def _inspect_npz(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("flat XGBoost NPZ exceeds the size bound")
    expected = {f"{name}.npy" for name in ARRAY_DTYPES}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("flat XGBoost NPZ member inventory changed")
            if any(
                member.is_dir()
                or member.flag_bits & 0x1
                or member.compress_type != zipfile.ZIP_STORED
                for member in members
            ):
                raise ValueError("flat XGBoost NPZ zip structure is unsupported")
            if sum(member.file_size for member in members) > MAX_NPZ_BYTES:
                raise ValueError("flat XGBoost NPZ expands beyond the size bound")
            if archive.testzip() is not None:
                raise ValueError("flat XGBoost NPZ failed its CRC check")
    except zipfile.BadZipFile as error:
        raise ValueError("flat XGBoost NPZ is not a valid zip archive") from error


def load_flat_npz(
    path: Path,
    *,
    expected_records: Mapping[str, Any] | None = None,
    expected_feature_count: int | None = None,
    expected_tree_count: int | None = None,
) -> tuple[FlatXGBoostBinary, dict[str, int]]:
    resolved = path.expanduser().resolve()
    _inspect_npz(resolved)
    before = sha256_file(resolved)
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_DTYPES):
                raise ValueError("flat XGBoost NPZ array inventory changed")
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in ARRAY_DTYPES
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and str(error).startswith("flat XGBoost"):
            raise
        raise ValueError("flat XGBoost NPZ array load failed") from error
    if sha256_file(resolved) != before:
        raise RuntimeError("flat XGBoost NPZ changed while being read")
    summary = validate_flat_arrays(
        arrays,
        expected_feature_count=expected_feature_count,
        expected_tree_count=expected_tree_count,
    )
    if expected_records is not None:
        if not isinstance(expected_records, Mapping) or set(expected_records) != set(
            ARRAY_DTYPES
        ):
            raise ValueError("flat XGBoost manifest array inventory changed")
        shapes = {name: list(arrays[name].shape) for name in ARRAY_DTYPES}
        for name in ARRAY_DTYPES:
            record = _validate_array_record(
                expected_records[name], name=name, expected_shape=shapes[name]
            )
            if array_sha256(arrays[name]) != record["sha256"]:
                raise RuntimeError(f"flat XGBoost array hash mismatch: {name}")
    model = FlatXGBoostBinary(
        tree_roots=arrays["tree_roots"],
        iteration_indptr=arrays["iteration_indptr"],
        children_left=arrays["children_left"],
        children_right=arrays["children_right"],
        split_index=arrays["split_index"],
        split_condition=arrays["split_condition"],
        default_left=arrays["default_left"],
        leaf_weight=arrays["leaf_weight"],
        base_margin=float(arrays["base_margin"][0]),
        feature_count=summary["feature_count"],
        max_depth=summary["max_depth"],
    )
    return model, summary


def _read_stable_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = sha256_file(path)
    def reject_constant(value: str) -> None:
        raise ValueError(f"flat XGBoost manifest contains {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("flat XGBoost manifest changed while being read")
    if not isinstance(value, dict):
        raise ValueError("flat XGBoost manifest must be an object")
    return value, before


def load_export(npz_path: Path, manifest_path: Path) -> FlatXGBoostBinary:
    """Authenticate an exporter pair and load it without importing XGBoost."""

    npz = npz_path.expanduser().resolve()
    manifest_file = manifest_path.expanduser().resolve()
    manifest, _ = _read_stable_json(manifest_file)
    if set(manifest) != {
        "schema_version",
        "status",
        "competition",
        "candidate_id",
        "format",
        "format_version",
        "source",
        "model",
        "output",
        "execution_scope",
    }:
        raise ValueError("flat XGBoost manifest field inventory changed")
    for key, expected in (
        ("schema_version", 1),
        ("status", "complete"),
        ("competition", COMPETITION),
        ("format", FORMAT_NAME),
        ("format_version", FORMAT_VERSION),
    ):
        _require_exact(manifest.get(key), expected, f"manifest {key}")
    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("flat XGBoost candidate_id is invalid")
    source = manifest.get("source")
    model_contract = manifest.get("model")
    output = manifest.get("output")
    execution = manifest.get("execution_scope")
    if not all(
        isinstance(value, Mapping)
        for value in (source, model_contract, output, execution)
    ):
        raise ValueError("flat XGBoost manifest sections are missing")
    if set(source) != {
        "xgboost_version",
        "model_json_sha256",
        "model_json_size_bytes",
        "config_sha256",
        "config_size_bytes",
        "exporter_sha256",
        "runtime_sha256",
        "source_model_file",
    }:
        raise ValueError("flat XGBoost source provenance inventory changed")
    if set(model_contract) != {
        "source_model_version",
        "objective",
        "output_link",
        "booster",
        "num_parallel_tree",
        "feature_count",
        "feature_names",
        "feature_types",
        "tree_count",
        "node_count",
        "max_depth",
        "base_score_probability",
        "base_margin",
        "tree_json_sha256",
        "branch_semantics",
        "missing_value",
        "tree_order",
        "tree_accumulation",
        "xgboost_runtime_dependency",
    }:
        raise ValueError("flat XGBoost model provenance inventory changed")
    if set(output) != {"npz_file", "npz_sha256", "npz_size_bytes", "arrays"}:
        raise ValueError("flat XGBoost output provenance inventory changed")
    _require_exact(
        execution,
        {
            "training_performed": False,
            "metrics_computed": False,
            "predictions_computed": False,
            "only_model_serialization_and_flattening": True,
        },
        "flat XGBoost execution scope",
    )
    for key, expected in (
        ("xgboost_version", SOURCE_XGBOOST_VERSION),
        ("objective", OBJECTIVE),
        ("output_link", OUTPUT_LINK),
        ("booster", "gbtree"),
        ("num_parallel_tree", 1),
    ):
        section = source if key == "xgboost_version" else model_contract
        _require_exact(section.get(key), expected, f"manifest {key}")
    for key in (
        "model_json_sha256",
        "config_sha256",
        "exporter_sha256",
        "runtime_sha256",
    ):
        if not _is_sha256(source.get(key)):
            raise ValueError(f"flat XGBoost source hash is invalid: {key}")
    for key in ("model_json_size_bytes", "config_size_bytes"):
        value = source.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"flat XGBoost source size is invalid: {key}")
    source_model = source.get("source_model_file")
    if source_model is not None:
        if not isinstance(source_model, Mapping):
            raise ValueError("flat XGBoost source model record is invalid")
        filename = source_model.get("filename")
        digest = source_model.get("sha256")
        size = source_model.get("size_bytes")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not _is_sha256(digest)
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError("flat XGBoost source model provenance is invalid")
    feature_count = model_contract.get("feature_count")
    tree_count = model_contract.get("tree_count")
    if type(feature_count) is not int or feature_count <= 0:
        raise ValueError("flat XGBoost feature_count is invalid")
    if type(tree_count) is not int or tree_count <= 0:
        raise ValueError("flat XGBoost tree_count is invalid")
    _require_exact(
        model_contract.get("source_model_version"),
        [3, 0, 4],
        "manifest source model version",
    )
    feature_names = model_contract.get("feature_names")
    feature_types = model_contract.get("feature_types")
    for description, values in (
        ("feature_names", feature_names),
        ("feature_types", feature_types),
    ):
        if (
            not isinstance(values, list)
            or len(values) not in (0, feature_count)
            or any(not isinstance(item, str) for item in values)
        ):
            raise ValueError(f"flat XGBoost {description} provenance is invalid")
    if feature_names and len(set(feature_names)) != len(feature_names):
        raise ValueError("flat XGBoost feature names are not unique")
    tree_hashes = model_contract.get("tree_json_sha256")
    if (
        not isinstance(tree_hashes, list)
        or len(tree_hashes) != tree_count
        or any(not _is_sha256(value) for value in tree_hashes)
    ):
        raise ValueError("flat XGBoost tree provenance hashes are invalid")
    base_score = model_contract.get("base_score_probability")
    base_margin = model_contract.get("base_margin")
    if (
        isinstance(base_score, bool)
        or not isinstance(base_score, (int, float))
        or not 0.0 < float(base_score) < 1.0
        or isinstance(base_margin, bool)
        or not isinstance(base_margin, (int, float))
        or not math.isfinite(float(base_margin))
    ):
        raise ValueError("flat XGBoost base score/margin is invalid")
    expected_margin = math.log(float(base_score)) - math.log1p(-float(base_score))
    if float(base_margin) != expected_margin:
        raise ValueError("flat XGBoost base score/link transformation changed")
    for key, expected in (
        ("branch_semantics", BRANCH_SEMANTICS),
        ("missing_value", MISSING_VALUE),
        ("tree_order", TREE_ORDER),
        ("tree_accumulation", TREE_ACCUMULATION),
        ("xgboost_runtime_dependency", False),
    ):
        _require_exact(model_contract.get(key), expected, f"manifest model {key}")
    _require_exact(output.get("npz_file"), npz.name, "manifest NPZ filename")
    npz_hash = output.get("npz_sha256")
    if not _is_sha256(npz_hash) or sha256_file(npz) != npz_hash:
        raise RuntimeError("flat XGBoost NPZ hash mismatch")
    _require_exact(
        output.get("npz_size_bytes"), int(npz.stat().st_size), "manifest NPZ size"
    )
    records = output.get("arrays")
    loaded, summary = load_flat_npz(
        npz,
        expected_records=records,
        expected_feature_count=feature_count,
        expected_tree_count=tree_count,
    )
    if float(loaded.base_margin) != float(base_margin):
        raise RuntimeError("flat XGBoost numeric base margin changed")
    _require_exact(
        model_contract.get("node_count"), summary["node_count"], "model node_count"
    )
    _require_exact(
        model_contract.get("max_depth"), summary["max_depth"], "model max_depth"
    )
    return loaded
