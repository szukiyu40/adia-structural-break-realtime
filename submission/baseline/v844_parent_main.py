"""ADIA V844: exact V826 formula with ten outer workers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import v826_parent_main  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate(model_directory_path: str) -> None:
    root = Path(model_directory_path).expanduser().resolve()
    manifest = json.loads((root / "v844_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_pending_parity_and_runtime_audits"
        or manifest.get("candidate_id") != "v844_v826_parallel10_transport"
        or manifest.get("parent") != "v826_v780_mega_fast8_transport"
        or manifest.get("formula_changed") is not False
        or manifest.get("infer_parallelism") != 10
    ):
        raise RuntimeError("V844 manifest contract changed")
    for relative, digest in manifest.get("source_files", {}).items():
        path = (CODE_ROOT / relative).resolve()
        if CODE_ROOT not in path.parents or not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V844 source changed: {relative}")


def train(datasets, model_directory_path: str):
    v826_parent_main.train(datasets, model_directory_path)
    _validate(model_directory_path)


def infer(datasets, model_directory_path: str):
    _validate(model_directory_path)
    yield from v826_parent_main.infer(datasets, model_directory_path)


# @crunch/keep:on
INFER_PARALLELISM = 10

