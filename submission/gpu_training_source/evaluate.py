# Publication change: private paths use configurable ADIA_* locations.
import os
import sys
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];OLD=Path(os.environ.get('ADIA_LEGACY_ROOT', str(Path.cwd() / 'external-artifacts/legacy')));sys.path.insert(0,str(OLD/'work'))
import numpy as np,pandas as pd
from scipy.special import expit,logit
from fit_time_rcd_online_v197 import metric

def logits(p):return logit(np.clip(p,1e-7,1-1e-7))
L=np.load(OLD/'outputs/models/full_parent_scope_v179/v35/online_lengths.npy');off=np.r_[0,np.cumsum(L)];tau=pd.read_parquet(OLD/'work/structural-break-real-time-dominant-wolf/data/y_train_index.parquet').loc[np.arange(10000),'tau_index'].to_numpy(np.int64)
y=np.where(np.arange(1000)[None,:]<L[:,None],((tau[:,None]>=0)&(np.arange(1000)[None,:]>=tau[:,None])).astype('i1'),-1)
f=np.load(OLD/'outputs/models/bayes_strength_v204/primary_probability.npy');base=np.zeros((10000,1000))
for sid,n in enumerate(L):base[sid,:n]=f[off[sid]:off[sid+1]]
adaptive=base.copy()
for fold in range(5):
 with np.load(ROOT/f'outputs/adaptive_variance_experiment/fold{fold}/sampled_corrections.npz') as z:ids=z['id'];ticks=z['tick'];d=z['difference']
 cursor=0
 for sid in np.unique(ids):
  n=L[sid];count=(n+3)//4;assert np.array_equal(ids[cursor:cursor+count],np.full(count,sid));adaptive[sid,:n]=expit(logits(base[sid,:n])+.5*np.repeat(d[cursor:cursor+count],4)[:n]);cursor+=count

def evaluate(name):
 folder=ROOT/'outputs'/name
 if not all((folder/f'fold{f}/corrections.npz').exists() for f in range(5)):return
 corr={k:np.zeros((10000,250)) for k in ['augmented','difference']}
 for fold in range(5):
  with np.load(folder/f'fold{fold}/corrections.npz') as z:
   assert np.all(z['id']%5==fold)
   for k in corr:corr[k][z['id']]=np.clip(z[k],-1,1)
 results=[]
 for kind in corr:
  for parent_name,parent in [('public_best',base),('previous_adaptive',adaptive)]:
   for alpha in [.1,.25,.5]:
    p=parent.copy()
    for sid,n in enumerate(L):p[sid,:n]=expit(logits(parent[sid,:n])+alpha*np.repeat(corr[kind][sid],4)[:n])
    scores=[]
    for fold in range(5):
     ix=np.arange(10000)%5==fold;scores.append(metric(y[ix],p[ix])-metric(y[ix],parent[ix]))
    results.append(dict(kind=kind,parent=parent_name,alpha=alpha,score=metric(y,p),delta_vs_parent=metric(y,p)-metric(y,parent),fold_deltas=scores,positive_folds=sum(d>0 for d in scores)))
 print(name,json.dumps(results),flush=True);(folder/'expanded_evaluation.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('name');args=ap.parse_args();evaluate(args.name)
