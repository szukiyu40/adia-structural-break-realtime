"""Reuse the retained experiment without changing its source or artifacts."""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent/'chronos_localization'
sys.path.insert(0, str(ORIGINAL))
from data import Data, Sampler, ROOT, DATA, BASE_MODEL, INCUMBENT, integrate
from network import Encoder, PatchHead, raw_features, objective
from util import write, sha, state_sha, metric

STEPS = 1000
SEED = 920901
STRIDE = 16
CONDITIONS = ('head_only', 'tail_and_head')

