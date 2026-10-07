"""Allocation-reduced, bit-exact transport of V799's frozen feature row."""

from __future__ import annotations

import math

import numpy as np

from v799_compact_v4seq18_feature_transport import (
    _RawAR10Rolling,
    _Sequential18State,
    _V3SupplementState,
)


def _install_fast_v2_pit(module) -> None:
    """Replace one pure static helper for every V2 rolling window."""

    if getattr(module.RollingComparison, "_v810_installed", False):
        return
    reference = np.arange(
        1, module.N_BINS + 1, dtype=np.float64
    ) / module.N_BINS

    def pit_statistics(counts: np.ndarray, n: int):
        expected = n / module.N_BINS
        # Integer counts are promoted by subtraction exactly as after the
        # parent's explicit astype(float64), without the first copy.
        differences = counts - expected
        chi_square = float(
            np.sum(differences * differences / max(expected, module.EPS))
        )
        chi_z = (chi_square - (module.N_BINS - 1)) / math.sqrt(
            2.0 * (module.N_BINS - 1)
        )
        cumulative = np.cumsum(counts, dtype=np.float64) / n
        ks = math.sqrt(n) * float(np.max(np.abs(cumulative - reference)))
        edge_rate = float((counts[0] + counts[-1]) / n)
        edge_z = (edge_rate - 2.0 / module.N_BINS) * math.sqrt(n)
        return chi_z, ks, edge_z

    module.RollingComparison._pit_statistics = staticmethod(pit_statistics)
    module.RollingComparison._v810_installed = True


class _FastV2State:
    def __init__(self, historical, module) -> None:
        _install_fast_v2_pit(module)
        self.source = module.StreamingFeaturesV2(historical)

    def update(self, raw_value: float) -> np.ndarray:
        return self.source.update(raw_value)


class _BufferedV3SupplementState(_V3SupplementState):
    def __init__(self, historical, module) -> None:
        super().__init__(historical, module)
        self.log_counts = np.asarray(
            [math.log1p(int(value)) for value in self.source.null_block_counts],
            dtype=np.float64,
        )
        self.effects = np.empty(len(module.WINDOWS) * 16, dtype=np.float64)
        self.output = np.empty(193, dtype=np.float32)

    def update(self, raw_value: float) -> np.ndarray:
        module = self.module
        source = self.source
        value = float(raw_value)
        if not math.isfinite(value):
            value = source.mean
        raw = module._clip_scalar(  # noqa: SLF001
            float((value - source.mean) / source.sd),
            -module.CLIP_VALUE,
            module.CLIP_VALUE,
        )
        residual10 = source._residual(raw, 10)  # noqa: SLF001
        source.previous_raw.append(raw)
        source.n += 1
        point = (
            raw,
            residual10,
            int(np.searchsorted(source.raw_edges, raw, side="right")),
            int(np.searchsorted(source.residual10_edges, residual10, side="right")),
        )
        residual10_start = len(module.ALL_STAT_NAMES) + len(
            module.RESIDUAL5_STAT_INDICES
        )
        cursor = 0
        effect_cursor = 0
        for window_index, (window, rolling) in enumerate(
            zip(module.WINDOWS, self.windows, strict=True)
        ):
            rolling.update(point)
            raw_stats, residual10_stats = rolling.statistics()
            raw_calibrated = (
                np.asarray(raw_stats, dtype=np.float64)
                - source.null_medians[window_index, : len(module.ALL_STAT_NAMES)]
            ) / source.null_scales[window_index, : len(module.ALL_STAT_NAMES)]
            residual10_calibrated = (
                np.asarray(residual10_stats, dtype=np.float64)
                - source.null_medians[
                    window_index,
                    residual10_start : residual10_start + len(module.ALL_STAT_NAMES),
                ]
            ) / source.null_scales[
                window_index,
                residual10_start : residual10_start + len(module.ALL_STAT_NAMES),
            ]
            np.clip(raw_calibrated, -30.0, 30.0, out=raw_calibrated)
            np.clip(residual10_calibrated, -30.0, 30.0, out=residual10_calibrated)
            self.output[cursor] = self.log_counts[window_index]
            cursor += 1
            self.output[cursor : cursor + 15] = raw_calibrated[1:]
            cursor += 15
            self.output[cursor : cursor + 16] = residual10_calibrated
            cursor += 16
            raw_effects = module._analytic_effects_scalar(  # noqa: SLF001
                raw_stats, source.references[0], window
            )
            residual_effects = module._analytic_effects_scalar(  # noqa: SLF001
                residual10_stats, source.references[2], window
            )
            self.effects[effect_cursor : effect_cursor + 8] = raw_effects
            self.effects[effect_cursor + 8 : effect_cursor + 16] = residual_effects
            effect_cursor += 16

        ranks = np.sum(
            source.analytic_null_matrix <= self.effects[:, None], axis=1
        )
        probabilities = (ranks + 0.5) / (source.analytic_null_counts + 1.0)
        largest = np.partition(probabilities, -8)[-8:]
        source.last_analytic_top8 = float(largest.mean())
        self.output[192] = source.last_analytic_top8
        return self.output


class _BufferedSequential18State(_Sequential18State):
    def __init__(self, historical, module) -> None:
        super().__init__(historical, module)
        self.output = np.empty((3, module.SUMMARY_COUNT), dtype=np.float64)

    def update(self, raw_value: float) -> np.ndarray:
        module = self.module
        base = self.source._base_scores(raw_value)[self.SELECTED]  # noqa: SLF001
        scores = np.clip(
            (base - self.family_centers) / self.family_scales,
            -module.SCORE_CLIP,
            module.SCORE_CLIP,
        )
        expanded = scores[:, None]
        np.maximum(
            0.0,
            self.page_positive + expanded - module.KAPPA_ARRAY[None, :],
            out=self.page_positive,
        )
        np.maximum(
            0.0,
            self.page_negative - expanded - module.KAPPA_ARRAY[None, :],
            out=self.page_negative,
        )
        np.maximum(
            self.page_positive,
            module.PAGE_DECAY * self.decayed_positive,
            out=self.decayed_positive,
        )
        np.maximum(
            self.page_negative,
            module.PAGE_DECAY * self.decayed_negative,
            out=self.decayed_negative,
        )
        normalized_positive = np.clip(
            self.page_positive / self.positive_null_scales,
            0.0,
            module.FEATURE_CLIP,
        )
        normalized_negative = np.clip(
            self.page_negative / self.negative_null_scales,
            0.0,
            module.FEATURE_CLIP,
        )
        normalized_decayed_positive = np.clip(
            self.decayed_positive / self.positive_null_scales,
            0.0,
            module.FEATURE_CLIP,
        )
        normalized_decayed_negative = np.clip(
            self.decayed_negative / self.negative_null_scales,
            0.0,
            module.FEATURE_CLIP,
        )
        for family in range(3):
            current_both = np.concatenate(
                [normalized_positive[family], normalized_negative[family]]
            )
            decayed_both = np.concatenate(
                [
                    normalized_decayed_positive[family],
                    normalized_decayed_negative[family],
                ]
            )
            self.output[family] = (
                module._logmeanexp(normalized_positive[family]),  # noqa: SLF001
                module._logmeanexp(normalized_negative[family]),  # noqa: SLF001
                float(current_both.max()),
                module._logmeanexp(current_both),  # noqa: SLF001
                float(decayed_both.max()),
                module._logmeanexp(decayed_both),  # noqa: SLF001
            )
        return np.asarray(self.output.ravel(), dtype=np.float32)


class ExactBufferedV4Seq18State:
    """Return the frozen 440 columns from reusable exact buffers."""

    def __init__(self, historical, views, feature_runtime) -> None:
        if tuple(views) != ("v4_seq18",):
            raise RuntimeError("V810 requires the frozen V4-seq18-only view")
        v2_module = feature_runtime._import_experiment("v2_candidate")  # noqa: SLF001
        v3_module = feature_runtime._import_experiment("v3_candidate")  # noqa: SLF001
        sequential_module = feature_runtime._import_experiment(  # noqa: SLF001
            "v5_sequential_evidence"
        )
        history = np.array(historical, dtype=np.float64, copy=True).reshape(-1)
        history.setflags(write=False)
        self.v2_state = _FastV2State(history, v2_module)
        self.v3_state = _BufferedV3SupplementState(history, v3_module)
        self.sequential_state = _BufferedSequential18State(
            history, sequential_module
        )
        self.row = np.empty(440, dtype=np.float32)

    def update(self, raw_value: float) -> dict[str, np.ndarray]:
        self.row[:229] = self.v2_state.update(raw_value)
        self.row[229:422] = self.v3_state.update(raw_value)
        self.row[422:] = self.sequential_state.update(raw_value)
        return {"v4_seq18": self.row}


__all__ = ["ExactBufferedV4Seq18State"]
