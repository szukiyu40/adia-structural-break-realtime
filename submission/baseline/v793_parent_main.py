"""ADIA V793: exact V780 state and multi-output tree transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import v787_parent_main  # noqa: E402
from v790_exact_state_transport import (  # noqa: E402
    ExactFastCurvatureState,
    ExactFastPositionalState,
)
from v791_tree_transport_runtime import install, validate_resources  # noqa: E402


v787_parent_main.LeanPositionalState = ExactFastPositionalState
v787_parent_main.LeanCurvatureState = ExactFastCurvatureState


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate(model_directory_path: str) -> None:
    root = Path(model_directory_path).expanduser().resolve()
    manifest = json.loads((root / "v793_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_pending_runtime_gates"
        or manifest.get("candidate_id") != "v793_v780_exact_multiclass_transport"
        or manifest.get("formula_changed") is not False
        or manifest.get("v790_component_audit_status") != "pass"
        or manifest.get("v792_independent_audit_status") != "pass"
    ):
        raise RuntimeError("V793 manifest contract changed")
    for relative, digest in manifest.get("source_files", {}).items():
        path = (CODE_ROOT / relative).resolve()
        if CODE_ROOT not in path.parents or not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V793 source changed: {relative}")
    for relative, digest in manifest.get("evidence_files", {}).items():
        path = root / relative
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V793 evidence changed: {relative}")
    validate_resources(root)


def train(datasets, model_directory_path: str):
    v787_parent_main.train(datasets, model_directory_path)
    _validate(model_directory_path)


def infer(datasets, model_directory_path: str):
    _validate(model_directory_path)
    install(v787_parent_main, model_directory_path)
    yield from v787_parent_main.infer(datasets, model_directory_path)


# @crunch/keep:on
INFER_PARALLELISM = 4

