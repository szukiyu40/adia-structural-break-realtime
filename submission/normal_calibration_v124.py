"""Causal recalibration of an existing V120 probability using its normal q90.

The inverse-logistic round trip is part of the serving definition. The original
0.25 coefficient is an exact identity, including floating-point values.
"""
import numpy as np
from scipy.special import expit, logit

COEFFICIENTS = (0., .25, .5, 1.)

def recalibrate(probability, normal_q90, coefficient):
    assert coefficient in COEFFICIENTS
    probability = np.asarray(probability, np.float64)
    normal_q90 = np.asarray(normal_q90, np.float64)
    assert np.isfinite(probability).all() and np.isfinite(normal_q90).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    if coefficient == .25:
        return probability.copy()
    return expit(logit(probability) - (coefficient - .25) * normal_q90)
