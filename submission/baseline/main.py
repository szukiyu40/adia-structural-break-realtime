"""ADIA V861: frozen E1e-b checkpoint correction over the exact V844 parent."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Iterator


for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


class _SeriesSlot:
    """Handshake between the online proxy and the parent score generator."""

    def __init__(self) -> None:
        self.state: Any | None = None
        self.pending_score = False


def _causal_datasets(
    datasets: Iterable[tuple[Any, Iterable[float]]],
    resources: Any,
    state_factory: Callable[[Any, Any], Any],
    slot: _SeriesSlot,
) -> Iterator[tuple[Any, Iterator[float]]]:
    for historical, online in datasets:
        if slot.pending_score:
            raise RuntimeError("V861 parent omitted an online score")
        state = state_factory(historical, resources)
        slot.state = state

        def causal_online(
            source: Iterable[float] = online,
            current: Any = state,
        ) -> Iterator[float]:
            for point in source:
                if slot.pending_score:
                    raise RuntimeError(
                        "V861 parent read a future point before yielding the current score"
                    )
                current.update(point)
                slot.pending_score = True
                yield point
            if slot.pending_score:
                raise RuntimeError("V861 parent omitted the final online score")

        yield historical, causal_online()


def train(datasets, model_directory_path: str):
    """Run the unchanged parent train hook, then authenticate V861 resources."""

    parent = importlib.import_module("v844_parent_main")
    parent.train(datasets, model_directory_path)
    runtime = importlib.import_module("v861_e1eb_runtime")
    runtime.load_resources(model_directory_path)


def infer(datasets, model_directory_path: str):
    """Preserve parent readiness and apply exactly one causal score per tick."""

    runtime = importlib.import_module("v861_e1eb_runtime")
    resources = runtime.load_resources(model_directory_path)
    parent = importlib.import_module("v844_parent_main")
    slot = _SeriesSlot()
    proxied = _causal_datasets(
        datasets,
        resources,
        runtime.CausalE1EBState,
        slot,
    )
    parent_outputs = iter(parent.infer(proxied, model_directory_path))
    try:
        ready = next(parent_outputs)
    except StopIteration as error:
        raise RuntimeError("V861 parent omitted the readiness yield") from error
    if slot.state is not None or slot.pending_score:
        raise RuntimeError("V861 parent consumed online data before readiness")
    yield ready

    for parent_probability in parent_outputs:
        if slot.state is None or not slot.pending_score:
            raise RuntimeError("V861 parent yielded a score without an online point")
        result = slot.state.apply(parent_probability)
        slot.pending_score = False
        yield result
    if slot.pending_score:
        raise RuntimeError("V861 parent stopped before the final online score")


# @crunch/keep:on
INFER_PARALLELISM = 10
