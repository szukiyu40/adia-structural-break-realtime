"""ADIA V811: V805 with the bit-exact V810 buffered feature state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import v805_parent_main  # noqa: E402
from v810_exact_feature_buffer_transport import (  # noqa: E402
    ExactBufferedV4Seq18State,
)


def _base_runtime():
    v800 = v805_parent_main.v800_parent_main
    return (
        v800.v796_parent_main.v793_parent_main.v787_parent_main.v338_main.base_runtime
    )


class _BoundBufferedFeatureState(ExactBufferedV4Seq18State):
    def __init__(self, historical, views) -> None:
        super().__init__(historical, views, _base_runtime().feature_runtime)


_base_runtime().SinglePassFeatureState = _BoundBufferedFeatureState


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate(model_directory_path: str) -> None:
    root = Path(model_directory_path).expanduser().resolve()
    manifest = json.loads((root / "v811_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_pending_parent_parity_audit"
        or manifest.get("candidate_id") != "v811_v780_buffered_fast8_transport"
        or manifest.get("formula_changed") is not False
        or manifest.get("v810_audit_status") != "pass"
        or manifest.get("infer_parallelism") != 8
    ):
        raise RuntimeError("V811 manifest contract changed")
    for relative, digest in manifest.get("source_files", {}).items():
        path = (CODE_ROOT / relative).resolve()
        if CODE_ROOT not in path.parents or not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V811 source changed: {relative}")
    for relative, digest in manifest.get("evidence_files", {}).items():
        path = root / relative
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"V811 evidence changed: {relative}")


def train(datasets, model_directory_path: str):
    v805_parent_main.train(datasets, model_directory_path)
    _validate(model_directory_path)


def infer(datasets, model_directory_path: str):
    _validate(model_directory_path)
    yield from v805_parent_main.infer(datasets, model_directory_path)


# @crunch/keep:on
INFER_PARALLELISM = 8

