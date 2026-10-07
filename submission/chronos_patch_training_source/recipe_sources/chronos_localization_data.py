"""Causal observed-prefix inputs; targets enter the training objective only."""
# Publication change: private paths use configurable ADIA_* locations.
import os
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
WORK = ROOT/'work/chronos_localization'
DATA = WORK/'data'
BASE_MODEL = ROOT/'outputs/research_baseline_gpu_group_0920/resources/chronos_adapter/base'
OLD = Path(os.environ.get('ADIA_CHRONOS_WORK', str(Path.cwd() / 'external-artifacts/chronos2')))
BLEND_ROOT = ROOT.parent/'https-hub-crunchdao-com-competitions-structural-3/outputs'
INCUMBENT = BLEND_ROOT/'blend_crossfit/fixed_contrast_0p1_probability.npy'
OLD_BLEND = ROOT.parent/'https-hub-crunchdao-com-competitions-structural-2/work/predictive_distribution/baseline.npy'


class Data:
    def __init__(self):
        self.bank = np.load(DATA/'causal_bank.npy', mmap_mode='r')
        self.labels = np.load(DATA/'labels.npy')
        self.lengths = np.load(DATA/'lengths.npy')
        self.tau = np.load(DATA/'tau.npy')
        assert self.bank.shape == (10000, 2024)

    def history(self, ids):
        return np.array(self.bank[ids, :1024], dtype='f4', copy=True)

    def current(self, ids, ticks):
        ids, ticks = np.asarray(ids), np.asarray(ticks)
        assert ids.shape == ticks.shape and (ticks >= 0).all()
        # The largest read column is 1024+t, the currently observed value.
        return np.array(self.bank[ids[:, None], ticks[:, None]+1+np.arange(1024)[None]], dtype='f4', copy=True)

    def targets(self, ids, ticks):
        endpoints = ticks[:, None]-1008+16*np.arange(64)[None]
        onset = self.tau[ids, None]
        target = ((onset >= 0) & (endpoints >= onset)).astype('f4')
        valid = endpoints >= 0
        assert np.array_equal(target[:, -1], self.labels[ids, ticks])
        return target, valid


class Sampler:
    def __init__(self, data, fit, seed):
        self.positive = [fit[data.labels[fit, t] == 1] for t in range(1000)]
        self.negative = [fit[data.labels[fit, t] == 0] for t in range(1000)]
        weights = np.array([len(a)*len(b) for a, b in zip(self.positive, self.negative)], dtype='f8')
        self.weights = weights/weights.sum()
        self.rng = np.random.default_rng(seed)

    def batch(self):
        tick = int(self.rng.choice(1000, p=self.weights))
        p, n = self.positive[tick], self.negative[tick]
        ids = np.r_[self.rng.choice(p, 8), self.rng.choice(n, 8)]
        return ids, np.full(16, tick, dtype='i8'), len(p)/(len(p)+len(n))


def feature_contrast(fold):
    directory = BLEND_ROOT/f'feature_integration/fold{fold}'
    ids = np.load(directory/'held_ids.npy')
    assert np.array_equal(ids, np.flatnonzero(np.arange(10000) % 5 == fold))
    values = [np.load(directory/f'{name}_probability.npy').astype('f8') for name in ('augmented', 'control')]
    ticks = np.arange(1000)//4*4
    a, c = [v[:, ticks] for v in values]
    return logit(np.clip(a, 1e-7, 1-1e-7))-logit(np.clip(c, 1e-7, 1-1e-7))


def integrate(fold, probability):
    held = np.flatnonzero(np.arange(10000) % 5 == fold)
    parent = np.load(DATA/'parent.npy', mmap_mode='r')[held].astype('f8')
    base = np.clip(.875*parent+.125*probability, 1e-7, 1-1e-7)
    delta = feature_contrast(fold)
    return np.where(delta == 0, base, expit(logit(base)+.1*delta))
