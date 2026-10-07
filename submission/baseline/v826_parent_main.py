"""ADIA V826: V818 plus independently audited V824 same-tick mega trees."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import v818_parent_main  # noqa: E402
from v824_v4_mega_tree_runtime import install as mega_install  # noqa: E402


_v811 = v818_parent_main.v811_parent_main
_v805 = _v811.v805_parent_main
_v800 = _v805.v800_parent_main
_v796 = _v800.v796_parent_main
_v793 = _v796.v793_parent_main
_v793.install = mega_install


class _UnusedExtraTrees:
    """V824 owns the authenticated ExtraTrees output for this exact row."""


_v796.load_native_extratrees = lambda _root: _UnusedExtraTrees()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate(model_directory_path: str) -> None:
    root = Path(model_directory_path).expanduser().resolve()
    manifest = json.loads((root / "v826_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_pending_parent_parity_audit"
        or manifest.get("candidate_id") != "v826_v780_mega_fast8_transport"
        or manifest.get("formula_changed") is not False
        or manifest.get("v825_audit_status") != "pass"
        or manifest.get("infer_parallelism") != 8
    ):
        raise RuntimeError("V826 manifest contract changed")
    for relative, digest in manifest.get("source_files", {}).items():
        path = (CODE_ROOT / relative).resolve()
        if CODE_ROOT not in path.parents or not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V826 source changed: {relative}")
    for relative, digest in manifest.get("evidence_files", {}).items():
        path = root / relative
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V826 evidence changed: {relative}")


def train(datasets, model_directory_path: str):
    v818_parent_main.train(datasets, model_directory_path)
    _validate(model_directory_path)


def infer(datasets, model_directory_path: str):
    _validate(model_directory_path)
    yield from v818_parent_main.infer(datasets, model_directory_path)


# @crunch/keep:on
INFER_PARALLELISM = 8

