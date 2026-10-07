"""Prediction-exact low-allocation transport for V780 positional/curvature states."""

from __future__ import annotations

import math

import numpy as np

import v784_v780_component_runtime as frozen


class ExactFastPositionalState(frozen.LeanPositionalState):
    """Write the frozen 94 features directly into their final row."""

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.center
        z = float(np.clip((clean - self.center) / self.scale, -12.0, 12.0))
        raw_difference = z - self.previous_z
        difference = float(
            np.clip(
                (raw_difference - self.difference_center) / self.difference_scale,
                -12.0,
                12.0,
            )
        )
        self.previous_z = z
        current = (
            z,
            abs(z) - self.history_abs_mean,
            z * z - self.history_square_mean,
            difference,
        )
        self.count += 1
        for stream, item in enumerate(current):
            self.cumulative[stream].append(self.cumulative[stream][-1] + item)
            self.cumulative_square[stream].append(
                self.cumulative_square[stream][-1] + item * item
            )

        output = np.empty(frozen.POSITIONAL_WIDTH, dtype=np.float64)
        variance_offsets = {(0, 4): 48, (0, 8): 52, (3, 4): 60, (3, 8): 64}
        means8_offsets: list[int] = []
        feature_offset = 0
        for stream in range(4):
            cumulative = self.cumulative[stream]
            square = self.cumulative_square[stream]
            for bins in (4, 8):
                means = output[feature_offset : feature_offset + bins]
                means.fill(0.0)
                variance_offset = variance_offsets.get((stream, bins))
                variances = (
                    output[variance_offset : variance_offset + bins]
                    if variance_offset is not None
                    else None
                )
                if variances is not None:
                    variances.fill(0.0)
                for index in range(bins):
                    left = (self.count * index) // bins
                    right = (self.count * (index + 1)) // bins
                    width = right - left
                    if width:
                        total = cumulative[right] - cumulative[left]
                        mean = total / width
                        means[index] = mean
                        if variances is not None:
                            variances[index] = max(
                                (square[right] - square[left]) / width
                                - mean * mean,
                                0.0,
                            )
                if bins == 8:
                    means8_offsets.append(feature_offset)
                feature_offset += bins

        output[48:72] = np.log(output[48:72] + 1e-8)
        for stream, means_offset in enumerate(means8_offsets):
            output[72 + 5 * stream : 77 + 5 * stream] = self._aggregate(
                output[means_offset : means_offset + 8]
            )
        output[92:94] = (math.log1p(self.count), math.log1p(self.history_size))
        np.nan_to_num(output, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(output, -20.0, 20.0, out=output)
        if output.shape != (frozen.POSITIONAL_WIDTH,) or not np.isfinite(output).all():
            raise RuntimeError("V790 positional row changed")
        return output.astype(np.float32)


class ExactFastCurvatureState(frozen.LeanCurvatureState):
    """Avoid the frozen one-row matrix/column-stack curvature temporaries."""

    def __init__(self, historical):
        super().__init__(historical)
        initial = np.asarray(self.window, dtype=np.float64)
        self._fast_window = np.empty(frozen.CURVATURE_WINDOW, dtype=np.float64)
        self._fast_window[: initial.size] = initial
        self._fast_window_size = int(initial.size)
        del self.window

    def update(self, value: float) -> np.ndarray:
        clean = float(value) if math.isfinite(float(value)) else self.replacement
        z = float(np.clip((clean - self.center) / self.scale, -12.0, 12.0))
        if self._fast_window_size < frozen.CURVATURE_WINDOW:
            self._fast_window[self._fast_window_size] = z
            self._fast_window_size += 1
        else:
            self._fast_window[:-1] = self._fast_window[1:]
            self._fast_window[-1] = z

        second = np.diff(self._fast_window[: self._fast_window_size], n=2)
        absolute_second = np.abs(second)
        positive = np.maximum(second, 0.0)
        negative = np.minimum(second, 0.0)
        raw = np.asarray(
            (
                np.mean(absolute_second),
                np.max(absolute_second),
                np.log(
                    (np.sum(positive * positive) + 1e-6)
                    / (np.sum(negative * negative) + 1e-6)
                ),
            ),
            dtype=np.float64,
        )
        np.nan_to_num(
            raw,
            copy=False,
            nan=0.0,
            posinf=frozen.CURVATURE_CLIP,
            neginf=-frozen.CURVATURE_CLIP,
        )
        np.clip(
            raw,
            -frozen.CURVATURE_CLIP,
            frozen.CURVATURE_CLIP,
            out=raw,
        )
        calibrated = np.clip(
            (raw - self.reference_center) / self.reference_scale,
            -frozen.CURVATURE_CLIP,
            frozen.CURVATURE_CLIP,
        ).astype(np.float32).astype(np.float64)
        signed = float(calibrated[0])
        absolute = np.abs(calibrated[[0, 2, 1]])
        self.count += 1
        denominator = float(self.count)

        self.signed_sum += signed
        self.signed_fast = 0.875 * self.signed_fast + 0.125 * signed
        self.signed_slow = (63.0 / 64.0) * self.signed_slow + signed / 64.0
        self.signed_positive_sum += max(signed, 0.0)
        self.signed_negative_sum += max(-signed, 0.0)
        self.signed_positive_count += signed > 0.0
        self.signed_minimum = min(self.signed_minimum, signed)
        self.signed_maximum = max(self.signed_maximum, signed)
        signed_output = np.asarray(
            (
                self.signed_sum / denominator,
                self.signed_fast,
                self.signed_slow,
                self.signed_fast - self.signed_slow,
                self.signed_positive_sum / denominator,
                self.signed_negative_sum / denominator,
                self.signed_positive_count / denominator,
                self.signed_maximum - self.signed_minimum,
            ),
            dtype=np.float64,
        )

        self.absolute_sum += absolute
        self.absolute_fast = 0.875 * self.absolute_fast + 0.125 * absolute
        self.absolute_slow = (63.0 / 64.0) * self.absolute_slow + absolute / 64.0
        self.absolute_above_count += absolute > 1.0
        self.absolute_excess_sum += np.maximum(absolute - 1.0, 0.0)
        new_maximum = absolute >= self.absolute_maximum
        self.absolute_maximum = np.maximum(self.absolute_maximum, absolute)
        self.absolute_age = np.where(new_maximum, 0.0, self.absolute_age + 1.0)
        absolute_output = np.column_stack(
            (
                self.absolute_sum / denominator,
                self.absolute_fast,
                self.absolute_slow,
                self.absolute_fast - self.absolute_slow,
                self.absolute_above_count / denominator,
                self.absolute_excess_sum / denominator,
                self.absolute_maximum - absolute,
                np.log1p(self.absolute_age),
            )
        ).reshape(-1)
        output = np.concatenate((signed_output, absolute_output))
        np.nan_to_num(output, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
        np.clip(output, -20.0, 20.0, out=output)
        if output.shape != (32,) or not np.isfinite(output).all():
            raise RuntimeError("V790 curvature row changed")
        return output.astype(np.float32)


__all__ = ["ExactFastCurvatureState", "ExactFastPositionalState"]

