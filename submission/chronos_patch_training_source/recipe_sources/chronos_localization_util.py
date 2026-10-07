import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def metric(y, probability, independent=False):
    wins, pairs = 0., 0
    for t in range(y.shape[1]):
        valid = y[:, t] >= 0
        yy, pp = y[valid, t], probability[valid, t]
        n, m = int((yy == 1).sum()), int((yy == 0).sum())
        if n*m:
            assert np.isfinite(pp).all()
            wins += roc_auc_score(yy, pp)*n*m if independent else rankdata(pp)[yy == 1].sum()-n*(n+1)/2
            pairs += n*m
    return float(wins/pairs)


def state_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
