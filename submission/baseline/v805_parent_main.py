"""ADIA V805: V780-equivalent compact/native/fast-LightGBM, eight workers."""

from __future__ import annotations

import os


# Each Crunch worker owns one series at a time.  Keep every numerical kernel
# single-threaded so eight outer workers cannot oversubscribe the CPU.
for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import v800_parent_main  # noqa: E402
from v803_fast_merged_lightgbm_runtime import install as fast_install  # noqa: E402


# V796 calls this symbol lazily, before loading any inference resources.  Swap
# only the already-audited transport, leaving the frozen V800 formula intact.
_v796 = v800_parent_main.v796_parent_main
_v793 = _v796.v793_parent_main
_v793.install = fast_install


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate(model_directory_path: str) -> None:
    root = Path(model_directory_path).expanduser().resolve()
    manifest = json.loads((root / "v805_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_pending_parent_parity_audit"
        or manifest.get("candidate_id") != "v805_v780_fast8_transport"
        or manifest.get("formula_changed") is not False
        or manifest.get("v803_audit_status") != "pass"
        or manifest.get("infer_parallelism") != 8
    ):
        raise RuntimeError("V805 manifest contract changed")
    for relative, digest in manifest.get("source_files", {}).items():
        path = (CODE_ROOT / relative).resolve()
        if CODE_ROOT not in path.parents or not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V805 source changed: {relative}")
    for relative, digest in manifest.get("evidence_files", {}).items():
        path = root / relative
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V805 evidence changed: {relative}")


def train(datasets, model_directory_path: str):
    v800_parent_main.train(datasets, model_directory_path)
    _validate(model_directory_path)


def infer(datasets, model_directory_path: str):
    _validate(model_directory_path)
    yield from v800_parent_main.infer(datasets, model_directory_path)


# @crunch/keep:on
INFER_PARALLELISM = 8

