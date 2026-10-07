import os
os.environ['NUMBA_NUM_THREADS']='4'
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from evaluate import ROOT,OLD,L,y,base,adaptive,metric,logits
NAMES=['gpu_tcn_grouped_fixed','gpu_tcn_seed2','gpu_tcn_seed3']
OUT=ROOT/'outputs/gpu_temporal_ensemble';OUT.mkdir(exist_ok=True)
residual=[];sources=[]
for name in NAMES:
 d=np.zeros((10000,250),np.float64)
 for fold in range(5):
  path=ROOT/f'outputs/{name}/fold{fold}/corrections.npz'
  with np.load(path) as z:ids=z['id'];v=z['augmented']
  assert np.all(ids%5==fold);d[ids]=v
  with path.open('rb') as f:sources.append({'path':str(path),'sha256':hashlib.file_digest(f,'sha256').hexdigest()})
 residual.append(d)
delta=np.clip(np.mean(residual,axis=0),-1,1);np.save(OUT/'oof_residual.npy',delta)
results=[];predictions={}
for pname,parent in [('public_best',base),('previous_adaptive',adaptive)]:
 for alpha in [.1,.25,.5]:
  p=parent.copy()
  for sid,n in enumerate(L):p[sid,:n]=expit(logits(parent[sid,:n])+alpha*np.repeat(delta[sid],4)[:n])
  folds=[]
  for fold in range(5):
   ix=np.arange(10000)%5==fold;folds.append({'fold':fold,'parent':metric(y[ix],parent[ix]),'candidate':metric(y[ix],p[ix])})
  row=dict(parent=pname,alpha=alpha,score=metric(y,p),delta=metric(y,p)-metric(y,parent),folds=folds,positive_folds=sum(f['candidate']>f['parent'] for f in folds));results.append(row);print(row,flush=True)
  if alpha==.25:predictions[pname]=p
(OUT/'evaluation.json').write_text(json.dumps({'baseline':metric(y,base),'previous_adaptive':metric(y,adaptive),'results':results,'source_predictions':sources,'primary_alpha':.25,'development_reused':True,'official_score':False},indent=2))
# Independently check the same metric against sklearn at selected times.
from sklearn.metrics import roc_auc_score
for t in [1,4,16,64,128,256,512,999]:
 keep=y[:,t]>=0
 if len(np.unique(y[keep,t]))==2:assert abs(metric(y[keep,t:t+1],predictions['previous_adaptive'][keep,t:t+1])-roc_auc_score(y[keep,t],predictions['previous_adaptive'][keep,t]))<1e-12
print('metric_sklearn_checks_passed',flush=True)
