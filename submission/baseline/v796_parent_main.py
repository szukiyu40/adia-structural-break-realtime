"""ADIA V796: audited V793 plus selective native ExtraTrees transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import v793_parent_main  # noqa: E402
from v705_native_tree_runtime import install_lightweight_parent_validation  # noqa: E402
from v795_native_extratrees_fast_runtime import load as load_native_extratrees  # noqa: E402


_INSTALLED_ROOT: Path | None = None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate(model_directory_path: str) -> Path:
    root = Path(model_directory_path).expanduser().resolve()
    manifest = json.loads((root / "v796_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_pending_parent_parity_audit"
        or manifest.get("candidate_id") != "v796_v780_native_extratrees_transport"
        or manifest.get("formula_changed") is not False
        or manifest.get("v795_audit_status") != "pass"
    ):
        raise RuntimeError("V796 manifest contract changed")
    for relative, digest in manifest.get("source_files", {}).items():
        path = (CODE_ROOT / relative).resolve()
        if CODE_ROOT not in path.parents or not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V796 source changed: {relative}")
    for relative, digest in manifest.get("evidence_files", {}).items():
        path = root / relative
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V796 evidence changed: {relative}")
    return root


def _install(model_directory_path: str) -> None:
    global _INSTALLED_ROOT
    root = _validate(model_directory_path)
    if _INSTALLED_ROOT is not None:
        if _INSTALLED_ROOT != root:
            raise RuntimeError("V796 installed for a different resource root")
        return

    parent = v793_parent_main.v787_parent_main
    base_runtime = parent.v338_main.base_runtime
    install_lightweight_parent_validation(base_runtime.original)
    native_extratrees = load_native_extratrees(root)

    # Install V793's exact merged-tree transport after lightweight validators
    # are in place, then replace only the authenticated ExtraTrees evaluator.
    v793_parent_main.install(parent, root)
    transported_load = base_runtime.load_resources

    def load_base_resources(directory, *, check_runtime: bool = True):
        resources = transported_load(directory, check_runtime=check_runtime)
        return base_runtime.LoadedResources(
            pairrank_seeds=resources.pairrank_seeds,
            extratrees_model=native_extratrees,
            meta=resources.meta,
        )

    base_runtime.load_resources = load_base_resources
    _INSTALLED_ROOT = root


def train(datasets, model_directory_path: str):
    v793_parent_main.train(datasets, model_directory_path)
    _validate(model_directory_path)


def infer(datasets, model_directory_path: str):
    _install(model_directory_path)
    yield from v793_parent_main.infer(datasets, model_directory_path)


# @crunch/keep:on
INFER_PARALLELISM = 4

