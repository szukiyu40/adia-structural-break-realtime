"""Authenticated ADIA v13 heterogeneous-stack Crunch entry points."""

from __future__ import annotations

from v13_heterogeneous_stack_runtime import infer_impl, train_impl


def train(datasets, model_directory_path: str):
    """Validate immutable pretrained resources; perform no fitting."""

    return train_impl(datasets, model_directory_path)


def infer(datasets, model_directory_path: str):
    """Yield readiness followed by one causal probability per online point."""

    yield from infer_impl(datasets, model_directory_path)


# @crunch/keep:on
INFER_PARALLELISM = 4
