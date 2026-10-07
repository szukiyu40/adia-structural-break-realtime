"""Generate exactly the frozen V4-seq18 columns without discarded channels."""

from __future__ import annotations

from collections import deque
import math

import numpy as np


class _RawAR10Rolling:
    """Two-channel projection of V3's initialized three-channel state."""

    def __init__(self, source, module) -> None:
        self.module = module
        self.size = int(source.size)
        self.values = deque(
            (value[0], value[2], value[3], value[4]) for value in source.values
        )
        self.sum1 = [source.sum1[0], source.sum1[2]]
        self.sum2 = [source.sum2[0], source.sum2[2]]
        self.sum3 = [source.sum3[0], source.sum3[2]]
        self.sum4 = [source.sum4[0], source.sum4[2]]
        self.sum_abs = [source.sum_abs[0], source.sum_abs[2]]
        self.sum_abs2 = [source.sum_abs2[0], source.sum_abs2[2]]
        self.sum_diff = [source.sum_diff[0], source.sum_diff[2]]
        self.sum_diff2 = [source.sum_diff2[0], source.sum_diff2[2]]
        self.cross = [source.cross[0].copy(), source.cross[2].copy()]
        self.abs_cross = [
            source.abs_cross[0].copy(),
            source.abs_cross[2].copy(),
        ]
        self.pit_counts = [
            source.pit_counts[0].copy(),
            source.pit_counts[1].copy(),
        ]

    def update(self, point: tuple[float, float, int, int]) -> None:
        queue = self.values
        if len(queue) == self.size:
            old = queue[0]
            old_values = old[:2]
            if len(queue) > 1:
                second = queue[1]
                for channel in range(2):
                    difference = second[channel] - old_values[channel]
                    self.sum_diff[channel] -= difference
                    self.sum_diff2[channel] -= difference * difference
            for lag_index, lag in enumerate(self.module.LAGS):
                if len(queue) > lag:
                    other = queue[lag]
                    for channel in range(2):
                        self.cross[channel][lag_index] -= (
                            old_values[channel] * other[channel]
                        )
                        self.abs_cross[channel][lag_index] -= (
                            abs(old_values[channel]) * abs(other[channel])
                        )
            queue.popleft()
            for channel in range(2):
                value = old_values[channel]
                square = value * value
                self.sum1[channel] -= value
                self.sum2[channel] -= square
                self.sum3[channel] -= square * value
                self.sum4[channel] -= square * square
                self.sum_abs[channel] -= abs(value)
                self.sum_abs2[channel] -= abs(value) ** 2
            self.pit_counts[0][old[2]] -= 1
            self.pit_counts[1][old[3]] -= 1

        if queue:
            previous = queue[-1]
            for channel in range(2):
                difference = point[channel] - previous[channel]
                self.sum_diff[channel] += difference
                self.sum_diff2[channel] += difference * difference
        for lag_index, lag in enumerate(self.module.LAGS):
            if len(queue) >= lag:
                previous = queue[-lag]
                for channel in range(2):
                    self.cross[channel][lag_index] += previous[channel] * point[channel]
                    self.abs_cross[channel][lag_index] += (
                        abs(previous[channel]) * abs(point[channel])
                    )
        queue.append(point)
        for channel in range(2):
            value = point[channel]
            square = value * value
            self.sum1[channel] += value
            self.sum2[channel] += square
            self.sum3[channel] += square * value
            self.sum4[channel] += square * square
            self.sum_abs[channel] += abs(value)
            self.sum_abs2[channel] += abs(value) ** 2
        self.pit_counts[0][point[2]] += 1
        self.pit_counts[1][point[3]] += 1

    def statistics(self) -> tuple[list[float], list[float]]:
        method = self.module._RollingChannels._channel_statistics  # noqa: SLF001
        return method(self, 0, 0), method(self, 1, 1)


class _V3SupplementState:
    def __init__(self, historical, module) -> None:
        self.module = module
        self.source = module.StreamingFeaturesV3(historical)
        self.windows = [
            _RawAR10Rolling(window, module) for window in self.source.windows
        ]

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
            int(
                np.searchsorted(
                    source.residual10_edges, residual10, side="right"
                )
            ),
        )

        features: list[float] = []
        analytic_probabilities: list[float] = []
        residual10_start = len(module.ALL_STAT_NAMES) + len(
            module.RESIDUAL5_STAT_INDICES
        )
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
            np.clip(
                residual10_calibrated,
                -30.0,
                30.0,
                out=residual10_calibrated,
            )
            features.append(math.log1p(int(source.null_block_counts[window_index])))
            features.extend(raw_calibrated[1:].tolist())
            features.extend(residual10_calibrated.tolist())
            analytic_probabilities.extend(
                module._analytic_effects_scalar(  # noqa: SLF001
                    raw_stats, source.references[0], window
                )
            )
            analytic_probabilities.extend(
                module._analytic_effects_scalar(  # noqa: SLF001
                    residual10_stats, source.references[2], window
                )
            )

        analytic_effect_array = np.asarray(
            analytic_probabilities, dtype=np.float64
        )
        ranks = np.sum(
            source.analytic_null_matrix <= analytic_effect_array[:, None], axis=1
        )
        probabilities = (ranks + 0.5) / (source.analytic_null_counts + 1.0)
        largest = np.partition(probabilities, -8)[-8:]
        source.last_analytic_top8 = float(largest.mean())
        features.append(source.last_analytic_top8)
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (193,):
            raise RuntimeError("V799 V3 supplement width changed")
        return result


class _Sequential18State:
    SELECTED = np.asarray((1, 3, 4), dtype=np.intp)

    def __init__(self, historical, module) -> None:
        self.module = module
        self.source = module.StreamingSequentialEvidence(historical)
        selected = self.SELECTED
        self.family_centers = self.source.family_centers[selected].copy()
        self.family_scales = self.source.family_scales[selected].copy()
        self.positive_null_scales = self.source.positive_null_scales[selected].copy()
        self.negative_null_scales = self.source.negative_null_scales[selected].copy()
        shape = (selected.size, module.KAPPA_ARRAY.size)
        self.page_positive = np.zeros(shape, dtype=np.float64)
        self.page_negative = np.zeros(shape, dtype=np.float64)
        self.decayed_positive = np.zeros(shape, dtype=np.float64)
        self.decayed_negative = np.zeros(shape, dtype=np.float64)

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
        output = np.empty((3, module.SUMMARY_COUNT), dtype=np.float64)
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
            output[family] = (
                module._logmeanexp(normalized_positive[family]),  # noqa: SLF001
                module._logmeanexp(normalized_negative[family]),  # noqa: SLF001
                float(current_both.max()),
                module._logmeanexp(current_both),  # noqa: SLF001
                float(decayed_both.max()),
                module._logmeanexp(decayed_both),  # noqa: SLF001
            )
        result = np.asarray(output.ravel(), dtype=np.float32)
        return np.nan_to_num(
            result,
            copy=False,
            nan=0.0,
            posinf=module.FEATURE_CLIP,
            neginf=0.0,
        )


class ExactCompactV4Seq18State:
    """Drop only channels discarded by the frozen V4-seq18 assembly."""

    def __init__(self, historical, views, feature_runtime) -> None:
        if tuple(views) != ("v4_seq18",):
            raise RuntimeError("V799 requires the frozen V4-seq18-only view")
        self.v2_module = feature_runtime._import_experiment("v2_candidate")  # noqa: SLF001
        self.v3_module = feature_runtime._import_experiment("v3_candidate")  # noqa: SLF001
        self.sequential_module = feature_runtime._import_experiment(  # noqa: SLF001
            "v5_sequential_evidence"
        )
        history = np.array(historical, dtype=np.float64, copy=True).reshape(-1)
        history.setflags(write=False)
        self.v2_state = self.v2_module.StreamingFeaturesV2(history)
        self.v3_state = _V3SupplementState(history, self.v3_module)
        self.sequential_state = _Sequential18State(
            history, self.sequential_module
        )

    def update(self, raw_value: float) -> dict[str, np.ndarray]:
        v2 = self.v2_state.update(raw_value)
        v3 = self.v3_state.update(raw_value)
        sequential = self.sequential_state.update(raw_value)
        row = np.concatenate((v2, v3, sequential)).astype(
            np.float32, copy=False
        )
        if row.shape != (440,) or not np.isfinite(row).all():
            raise RuntimeError("V799 compact V4-seq18 row changed")
        return {"v4_seq18": row}


__all__ = ["ExactCompactV4Seq18State"]

