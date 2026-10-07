import os
os.environ['NUMBA_NUM_THREADS']='4'
from pathlib import Path
import json,sys
import numpy as np
from scipy.special import expit
from numba import njit,prange
from evaluate import ROOT,OLD,L,y,base,adaptive,metric,logits
OUT=ROOT/'outputs/gpu_temporal_ensemble'
def ordered(pred):
 ids=[];labels=[];new=[];starts=[0]
 for t in range(1000):
  ix=np.flatnonzero(y[:,t]>=0);ix=ix[np.argsort(pred[ix,t],kind='stable')];v=pred[ix,t]
  ids.append(ix.astype('i2'));labels.append(y[ix,t]);new.append(np.r_[True,v[1:]!=v[:-1]]);starts.append(starts[-1]+len(ix))
 return np.concatenate(ids),np.concatenate(labels),np.concatenate(new),np.array(starts)
@njit
def weighted_auc(w,ids,labels,new,starts):
 U=0.;D=0.
 for t in range(len(starts)-1):
  neg=0.;pos=0.;gp=0.;gn=0.;u=0.
  for k in range(starts[t],starts[t+1]):
   if new[k]:
    u+=gp*(neg+.5*gn);neg+=gn;pos+=gp;gp=0.;gn=0.
   if labels[k]:gp+=w[ids[k]]
   else:gn+=w[ids[k]]
  u+=gp*(neg+.5*gn);neg+=gn;pos+=gp;U+=u;D+=neg*pos
 return U/D
@njit(parallel=True)
def all_scores(weights,ids,labels,new,starts):
 out=np.empty(len(weights))
 for i in prange(len(weights)):out[i]=weighted_auc(weights[i],ids,labels,new,starts)
 return out

if __name__=='__main__':
 delta=np.load(OUT/'oof_residual.npy');p=adaptive.copy()
 for sid,n in enumerate(L):p[sid,:n]=expit(logits(adaptive[sid,:n])+.25*np.repeat(delta[sid],4)[:n])
 rng=np.random.default_rng(91624);weights=np.array([np.bincount(rng.integers(10000,size=10000),minlength=10000) for _ in range(399)],dtype='i2')
 scores={};points={}
 for name,pred in [('public_best',base),('previous_adaptive',adaptive),('gpu_candidate',p)]:
  args=ordered(pred);single=weighted_auc(np.ones(10000),*args);assert abs(single-metric(y,pred))<1e-12
  scores[name]=all_scores(weights,*args);points[name]=single;print(name,single,flush=True)
 records={}
 for name in ['public_best','previous_adaptive']:
  difference=scores['gpu_candidate']-scores[name]
  records[name]={'point_delta':points['gpu_candidate']-points[name],'paired_series_95_interval':np.quantile(difference,[.025,.975]).tolist(),'fraction_positive':float(np.mean(difference>0))}
 result={'points':points,'bootstrap_replicates':399,'comparisons':records,'caveat':'Fixed OOF predictions, resample whole series; no refitting. Reused development data and exploratory selection. Not a fresh test or official score.'}
 (OUT/'bootstrap.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
