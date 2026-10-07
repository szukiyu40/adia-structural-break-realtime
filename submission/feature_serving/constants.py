"""Frozen predictor/feature constants; no training-workspace dependencies."""
import numpy as np

QUANTILES = np.array([.01, .05, .1, .25, .5, .75, .9, .95, .99], dtype='f4')
ORDER = 12
INFER_STRIDE = 4
