"""Frozen Chronos encoder differences from a series' own normal history.

Embeddings only: no forecast residuals, online fitting, labels or other series.
All network computations use CPU float32 and a fixed checkpoint.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import torch
from chronos import ChronosBoltPipeline

CHECKPOINT = Path(__file__).resolve().parent / 'chronos_probe_v85/checkpoint'
WIDTHS = (128, 512)
STEP = 16
EMBED_DIM = 768
DIM = 2 * (EMBED_DIM + 8 + 2) + 2

class Encoder:
    def __init__(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.pipeline = ChronosBoltPipeline.from_pretrained(CHECKPOINT, device_map='cpu',
            dtype=torch.float32, local_files_only=True, trust_remote_code=False)
        self.pipeline.model.eval()
        for parameter in self.pipeline.model.parameters():
            parameter.requires_grad_(False)

    def windows(self, windows):
        x = np.asarray(windows, np.float32)
        assert x.ndim == 2 and x.shape[1] in WIDTHS and np.isfinite(x).all()
        result = []
        for lo in range(0, len(x), 32):
            chunk = x[lo:lo + 32]
            # Fixed separate signs make global sign reversal swap only two terms.
            pair = []
            for sign in (1., -1.):
                with torch.inference_mode():
                    embeddings, _ = self.pipeline.embed(torch.from_numpy(chunk * sign))
                a = embeddings.numpy()
                pair.append(np.concatenate((a[:, -1], a[:, :-1].mean(axis=1)), axis=1))
            result.append((pair[0].astype(np.float64) + pair[1].astype(np.float64)) * .5)
        return np.concatenate(result)

class Reference:
    def __init__(self, encoder, history):
        h = np.asarray(history, np.float64)
        assert h.ndim == 1 and len(h) >= 512 and np.isfinite(h).all()
        self.encoder, self.history = encoder, h.copy()
        self.mean, self.scale = float(h.mean()), max(float(h.std()), 1e-6)
        self.statistics = {}
        for width in WIDTHS:
            endpoints = np.unique(np.linspace(width, len(h), 24, dtype=np.int64))
            windows = np.stack([h[end-width:end] for end in endpoints])
            e = encoder.windows(windows)
            center = e.mean(axis=0)
            variance = e.var(axis=0)
            # Per-pool floors prevent nearly constant encoder axes becoming IDs.
            floor = np.repeat(.2 * np.sqrt(variance.reshape(2, 384).mean(axis=1)) + 1e-3, 384)
            scale = np.maximum(np.sqrt(variance), floor)
            self.statistics[width] = dict(center=center, scale=scale, endpoints=endpoints)

    def at(self, online, ticks):
        online = np.asarray(online, np.float64)
        ticks = np.asarray(ticks, np.int64)
        assert np.all((ticks >= 0) & (ticks < len(online)))
        sequence = np.r_[self.history[-512:], online]
        values = []
        for width in WIDTHS:
            ends = 512 + ticks + 1
            windows = np.stack([sequence[end-width:end] for end in ends])
            e = self.encoder.windows(windows)
            stat = self.statistics[width]
            z = np.clip((e - stat['center']) / stat['scale'], -12., 12.)
            pools = z.reshape(len(ticks), 2, 384)
            summaries = np.concatenate((np.mean(pools**2, axis=2), np.mean(np.abs(pools), axis=2),
                np.quantile(np.abs(pools), .9, axis=2), np.max(np.abs(pools), axis=2)), axis=1)
            # Magnitude is sign-invariant. Level changes are represented by absolute size.
            location = np.abs(windows.mean(axis=1) - self.mean) / self.scale
            log_scale = np.log(np.maximum(windows.std(axis=1), 1e-6) / self.scale)
            values.append(np.c_[z, summaries, np.clip(location, 0, 12), np.clip(log_scale, -8, 8)])
        age = ticks + 1
        values.append(np.c_[np.log1p(age)/np.log(1001), age/(age+128)])
        result = np.concatenate(values, axis=1).astype(np.float32)
        assert result.shape == (len(ticks), DIM) and np.isfinite(result).all()
        return result

class Stream:
    def __init__(self, encoder, history):
        self.reference = Reference(encoder, history)
        self.online = []
        self.last = None
        self.calls = 0

    def consume(self, point):
        self.online.append(float(point))
        tick = len(self.online) - 1
        if tick % STEP == 0:
            self.last = self.reference.at(self.online, [tick])[0]
            self.calls += 1
        result = self.last.copy()
        age = tick + 1
        result[-2:] = np.log1p(age)/np.log(1001), age/(age+128)
        return result

def expand_updates(updates, length):
    result = np.asarray(updates)[np.arange(length)//STEP].copy()
    age = np.arange(length) + 1
    result[:, -2] = np.log1p(age)/np.log(1001)
    result[:, -1] = age/(age+128)
    return result
