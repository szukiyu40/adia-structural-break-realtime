"""Fit fixed GPU temporal recipe on all training series with OOF core offsets."""
from pathlib import Path
import json,time,sys
import numpy as np,pandas as pd,torch
from scipy.special import logit
from train_replicate import ROOT,OLD,fit,sha,savejson

if __name__=='__main__':
 out=ROOT/'outputs/gpu_temporal_ensemble/fullfit';out.mkdir(parents=True,exist_ok=False)
 L=np.load(OLD/'outputs/models/full_parent_scope_v179/v35/online_lengths.npy');off=np.r_[0,np.cumsum((L+3)//4)]
 F=np.load(OLD/'outputs/models/discord_bayes_v198/features.npy',mmap_mode='r');A=np.load(ROOT/'work/adaptive_variance.npy',mmap_mode='r')
 scores=np.zeros((10000,250),np.float32);seen=np.zeros(10000,bool);scopes=[]
 for fold in range(5):
  scope=OLD/f'outputs/models/full_parent_scope_v179/core/fold{fold}';info=json.loads((scope/'scope.json').read_text());path=scope/'evaluation.npz'
  assert sha(path)==info['files']['evaluation.npz'] and info['common_outer_excluded_from_both_nested_bases']
  with np.load(path) as z:
   keep=z['tick']%4==0;ids=z['id'][keep];ticks=z['tick'][keep];p=z['core'][keep]
  assert np.all(ids%5==fold);seen[ids]=True;scores[ids,ticks//4]=logit(np.clip(p,1e-7,1-1e-7));scopes.append({'fold':fold,'source':str(path),'sha256':sha(path)})
 assert seen.all()
 x=np.zeros((10000,250,130),np.float32);x[:,:,0]=scores
 for sid,n in enumerate(L):
  count=(n+3)//4;x[sid,:count,1:58]=F[off[sid]:off[sid+1]];x[sid,:count,58:]=A[off[sid]:off[sid+1]]
 for k in range(0,72,12):x[:,:,58+k:58+k+6]=abs(x[:,:,58+k:58+k+6])
 tau=pd.read_parquet(OLD/'work/structural-break-real-time-dominant-wolf/data/y_train_index.parquet').loc[np.arange(10000),'tau_index'].to_numpy(np.int64)
 device='cuda' if torch.cuda.is_available() else 'mps';assert device=='cuda' or torch.backends.mps.is_available();torch.set_num_threads(2)
 protocol=dict(series=10000,seeds=[91621,91622,91623],steps=1600,device=device,core_sources=scopes,source_sha=sha(__file__),trainer_sha=sha(Path(__file__).with_name('train_replicate.py')),network_sha=sha(Path(__file__).with_name('network.py')),validation_labels_or_full_parent_predictions_used_in_training=False,training_offsets='Each series uses its held-fold core score; rank pairs share the same predicting core model.',caveat='OOF core training-size difference and inherited prior pipeline limitations remain.')
 savejson(out/'protocol.json',protocol);start=time.time();models=[]
 for seed in protocol['seeds']:
  folder=out/f'seed{seed}';folder.mkdir()
  fit(x,scores,tau,L,np.arange(10000),x[:2],folder,'augmented',device,1600,seed)
  b=torch.load(folder/'augmented.pt',map_location='cpu',weights_only=False)
  np.savez_compressed(out/f'model_{seed}.npz',mean=b['mean'],std=b['std'],**{k:v.numpy() for k,v in b['state'].items()})
  models.append({'seed':seed,'file':f'model_{seed}.npz','sha256':sha(out/f'model_{seed}.npz')})
 savejson(out/'metadata.json',dict(features=130,width=32,dilations=[1,4,16],stride=4,buffer_checkpoints=43,alpha=.25,clip=1,model_files=models,formula='Add .25 * clip(mean(three temporal residuals), -1, 1) to the adaptive candidate logit',version='gpu_temporal_ensemble_20260916'))
 savejson(out/'result.json',dict(completed=True,device=device,seconds=time.time()-start,models=models,official_submission=False));print('FULLFIT_COMPLETE',time.time()-start,flush=True)
