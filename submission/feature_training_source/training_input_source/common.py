"""Paths and scoring for an isolated, resumable local experiment."""
# Publication change: private paths use configurable ADIA_* locations.
import os
import sys
from pathlib import Path
import hashlib, json, time
import numpy as np
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT / 'work/predictive_distribution'
OLD = Path(os.environ.get('ADIA_PRIOR_ROOT', str(Path.cwd() / 'external-artifacts/prior')))
DATA = Path(os.environ.get('ADIA_DATA_DIR', str(Path.cwd() / 'data')))
PYTHON = os.environ.get('ADIA_PYTHON', sys.executable)
QUANTILES = np.array([.01,.05,.1,.25,.5,.75,.9,.95,.99], dtype='f4')
ORDER = 12
SEED = 91943

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def metric(y, p):
    assert y.shape == p.shape and np.isfinite(p[y>=0]).all()
    wins = 0.; pairs = 0
    for t in range(y.shape[1]):
        valid = y[:,t]>=0; yy=y[valid,t]; pp=p[valid,t]
        n=int((yy==1).sum()); m=int((yy==0).sum())
        if n*m:
            wins += rankdata(pp)[yy==1].sum()-n*(n+1)/2
            pairs += n*m
    return float(wins/pairs) if pairs else .5

def stream(path):
    import pyarrow.parquet as pq
    tail=None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=262144):
        a={k:batch.column(k).to_numpy() for k in batch.schema.names}
        if tail is not None:
            a={k:np.concatenate([tail[k],v]) for k,v in a.items()}
        cuts=np.r_[0,np.flatnonzero(np.diff(a['id']))+1,len(a['id'])]
        for lo,hi in zip(cuts[:-2],cuts[1:-1]):
            yield {k:v[lo:hi] for k,v in a.items()}
        tail={k:v[cuts[-2]:] for k,v in a.items()}
    if tail is not None: yield tail

def loads():
    return {n:np.load(CACHE/f'{n}.npy',mmap_mode='r') for n in
            ['values','innovations','hist_lengths','lengths','static','labels','baseline']}
