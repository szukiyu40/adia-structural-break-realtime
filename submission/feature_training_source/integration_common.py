"""Frozen, isolated 422 versus 566 feature comparison."""
# Publication change: private paths use configurable ADIA_* locations.
import os
import sys
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
WORK = ROOT / 'work/feature_integration'
OLD = Path(os.environ.get('ADIA_LEGACY_ROOT', str(Path.cwd() / 'external-artifacts/legacy')))
PREDICTIVE = Path(os.environ.get('ADIA_PREDICTIVE_ROOT', str(Path.cwd() / 'external-artifacts/predictive')))
ARRAYS = PREDICTIVE / 'work/predictive_distribution'
PRED_SOURCE = PREDICTIVE / 'outputs/predictive_distribution'
MANIFEST = OLD / 'outputs/models/full_parent_detector_v180/features/manifest.json'
RECIPE = Path(os.environ.get('ADIA_FEATURE_RECIPE', str(Path.cwd() / 'external-artifacts/run_v255_equal_series_logtick_v4.py')))
PYTHON = OLD / 'work/eda-env/bin/python'
SEEDS = (0, 1, 2)
INFER_STRIDE = 4
THREADS = 4

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def log_steps(length, count=40):
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.expm1(np.linspace(0., np.log1p(length-1), count))).astype(np.int64))

def parameters(seed):
    tree = ast.parse(RECIPE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'MODEL_PARAMETERS' for t in n.targets))
    p = ast.literal_eval(node.value)
    for name in ('random_state', 'bagging_seed', 'feature_fraction_seed', 'data_random_seed'):
        p[name] = seed
    p['n_jobs'] = THREADS
    assert p['n_estimators'] == 500 and p['num_leaves'] == 63
    return p

def arrays():
    return {name: np.load(ARRAYS / f'{name}.npy', mmap_mode='r') for name in
            ('values', 'innovations', 'hist_lengths', 'lengths', 'static', 'labels', 'baseline')}

def predictor_path(fold, inner):
    return OUT / 'predictors' / f'fold{fold}' / f'predictor{inner}' / 'latest.pt'

def predictor_modules():
    sys.path.insert(0, str(OUT / 'predictor_source'))
    import common
    common.CACHE = ARRAYS
    import network, features
    return network, features

def metric(y, p):
    assert y.shape == p.shape and np.isfinite(p[y >= 0]).all()
    wins = 0.; pairs = 0
    for t in range(y.shape[1]):
        mask = y[:, t] >= 0; yy = y[mask, t]; pp = p[mask, t]
        n = int((yy == 1).sum()); m = int((yy == 0).sum())
        if n * m:
            wins += rankdata(pp)[yy == 1].sum() - n * (n + 1) / 2
            pairs += n * m
    return float(wins / pairs) if pairs else .5

def ensure_protocol():
    p = read(OUT / 'protocol.json')
    for name, expected in p['experiment_sources'].items():
        assert sha(OUT / name) == expected, f'Experiment source changed: {name}'
    return p
