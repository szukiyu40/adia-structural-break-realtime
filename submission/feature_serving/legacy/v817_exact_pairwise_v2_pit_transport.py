"""Reproduce NumPy 2.3.3's 16-element pairwise sum without arrays."""

from __future__ import annotations

import math
import sys

import numpy as np

from .v810_exact_feature_buffer_transport import ExactBufferedV4Seq18State


def _install(module) -> None:
    def pit_statistics(counts: np.ndarray, n: int):
        expected = n / module.N_BINS
        denominator = max(expected, module.EPS)
        accumulators = [0.0] * 8
        cumulative = 0
        largest = 0.0
        for index in range(module.N_BINS):
            count = int(counts[index])
            difference = float(count) - expected
            square = difference * difference
            term = square / denominator
            if index < 8:
                accumulators[index] = term
            else:
                accumulators[index - 8] += term
            cumulative += count
            discrepancy = abs(
                cumulative / n - (index + 1) / module.N_BINS
            )
            largest = max(largest, discrepancy)
        chi_square = (
            (accumulators[0] + accumulators[1])
            + (accumulators[2] + accumulators[3])
        ) + (
            (accumulators[4] + accumulators[5])
            + (accumulators[6] + accumulators[7])
        )
        root_n = math.sqrt(n)
        chi_z = (chi_square - (module.N_BINS - 1)) / math.sqrt(
            2.0 * (module.N_BINS - 1)
        )
        ks = root_n * largest
        edge_rate = float((counts[0] + counts[-1]) / n)
        edge_z = (edge_rate - 2.0 / module.N_BINS) * root_n
        return chi_z, ks, edge_z

    module.RollingComparison._pit_statistics = staticmethod(pit_statistics)
    module.RollingComparison._v810_installed = True


class ExactPairwisePitV4Seq18State(ExactBufferedV4Seq18State):
    def __init__(self, historical, views, feature_runtime) -> None:
        super().__init__(historical, views, feature_runtime)
        module = sys.modules[self.v2_state.source.__class__.__module__]
        _install(module)


__all__ = ["ExactPairwisePitV4Seq18State"]

