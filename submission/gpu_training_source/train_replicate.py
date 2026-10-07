"""GPU same-time ranking; paired control/augmented temporal networks, fixed folds."""
# Publication change: private paths use configurable ADIA_* locations.
import os
import sys
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,time,json,argparse,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=Path(os.environ.get('ADIA_LEGACY_ROOT', str(Path.cwd() / 'external-artifacts/legacy')));sys.path.insert(0,str(OLD/'work'))
import numpy as np,pandas as pd,torch
from scipy.special import expit,logit
from fit_time_rcd_online_v197 import metric
from network import TemporalResidual

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def savejson(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def logits(p):return logit(np.clip(p,1e-7,1-1e-7))

def fit(x,base,tau,lengths,ids,xe,folder,name,device,steps,seed):
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 valid=np.arange(250)[None,:]*4<lengths[:,None]
 values=x[valid];mean=values.mean(0);std=np.maximum(values.std(0),.1)
 x=np.clip((x-mean)/std,-8,8).astype('f4');xe=np.clip((xe-mean)/std,-8,8).astype('f4')
 tx=torch.as_tensor(x,device=device);tb=torch.as_tensor(base,dtype=torch.float32,device=device)
 model=TemporalResidual(x.shape[-1]).to(device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
 pos=[];neg=[];mass=[]
 for group in np.unique(ids%5):
  for t in range(250):
   live=(lengths>4*t)&(ids%5==group);changed=(tau>=0)&(tau<=4*t)
   p=np.flatnonzero(live&changed);n=np.flatnonzero(live&~changed)
   pos.append(p);neg.append(n);mass.append(len(p)*len(n))
 mass=np.asarray(mass,float);mass/=mass.sum();history=[];start=time.time()
 for step in range(1,steps+1):
  group_time=int(rng.choice(len(mass),p=mass));t=group_time%250;selected=np.r_[rng.choice(pos[group_time],32),rng.choice(neg[group_time],32)];assert len(np.unique(ids[selected]%5))==1
  ix=torch.as_tensor(selected,device=device)
  # Receptive field is 43 checkpoints. Prefix truncation is exact after burn-in.
  lo=max(0,t-42);window=tx[ix,lo:t+1]
  window=torch.nn.functional.pad(window,(0,0,0,43-window.shape[1]))
  r=model(window)[:,t-lo]
  z=tb[ix,t]+r
  loss=torch.nn.functional.softplus(-(z[:32,None]-z[None,32:])).mean()+.03*r.square().mean()
  optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  if step%100==0:
   value=float(loss.detach().cpu());history.append({'step':step,'loss':value,'seconds':time.time()-start})
   print(name,folder.name,history[-1],flush=True)
  if step%400==0:
   state={'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'mean':mean,'std':std,'features':x.shape[-1],'width':32,'step':step,'seed':seed}
   torch.save(state,folder/f'{name}_{step}.pt')
 model=model.cpu().eval();torch.save({'state':model.state_dict(),'mean':mean,'std':std,'features':x.shape[-1],'width':32,'step':steps,'seed':seed},folder/f'{name}.pt')
 # Scoring uses CPU float32, as serving will; model selection never reads training loss as AUC.
 with torch.inference_mode():
  prediction=np.concatenate([model(torch.from_numpy(xe[k:k+64])).numpy() for k in range(0,len(xe),64)])
  prefix=model(torch.from_numpy(xe[:2,:17])).numpy();assert np.max(abs(prefix-prediction[:2,:17]))<2e-6
 savejson(folder/f'{name}_learning.json',history)
 del model,tx,tb,optimizer;gc.collect()
 if device=='mps':torch.mps.empty_cache()
 return prediction

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='gpu_tcn_seed2');ap.add_argument('--steps',type=int,default=1600);ap.add_argument('--folds',default='0,1,2,3,4');ap.add_argument('--seed',type=int,default=91622);args=ap.parse_args()
 out=ROOT/'outputs'/args.name;out.mkdir(exist_ok=True)
 device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu';assert device!='cpu';torch.set_num_threads(2)
 L=np.load(OLD/'outputs/models/full_parent_scope_v179/v35/online_lengths.npy');off=np.r_[0,np.cumsum((L+3)//4)];full=np.r_[0,np.cumsum(L)]
 source=OLD/'outputs/models/discord_bayes_v198/features.npy';f=np.load(source,mmap_mode='r');a=np.load(ROOT/'work/adaptive_variance.npy',mmap_mode='r')
 xf=np.zeros((10000,250,57),np.float32);xa=np.zeros((10000,250,72),np.float32)
 for sid,n in enumerate(L):
  c=(n+3)//4;xf[sid,:c]=f[off[sid]:off[sid+1]];xa[sid,:c]=a[off[sid]:off[sid+1]]
 # Mean shift direction is arbitrary: impose global sign invariance at feature level.
 for k in range(0,72,12):xa[:,:,k:k+6]=abs(xa[:,:,k:k+6])
 taupath=OLD/'work/structural-break-real-time-dominant-wolf/data/y_train_index.parquet';tau=pd.read_parquet(taupath).loc[np.arange(10000),'tau_index'].to_numpy(np.int64)
 y=np.where(np.arange(1000)[None,:]<L[:,None],((tau[:,None]>=0)&(np.arange(1000)[None,:]>=tau[:,None])).astype('i1'),-1)
 bf=np.load(OLD/'outputs/models/bayes_strength_v204/primary_probability.npy');base=np.zeros((10000,1000))
 for sid,n in enumerate(L):base[sid,:n]=bf[full[sid]:full[sid+1]]
 protocol=dict(device=device,torch=torch.__version__,steps=args.steps,seed=args.seed,width=32,dilations=[1,4,16],stride=4,primary_alpha=.25,diagnostic_alphas=[.1,.5],loss='same-time, same-inner-model-fold all-pair logistic ranking plus .03 squared residual; group/time sampled by positive-negative pair count',baseline_public_submission=71,baseline_local_score=metric(y,base),training_feature_sha=sha(source),adaptive_feature_sha=sha(ROOT/'work/adaptive_variance.npy'),source_sha=sha(__file__),network_sha=sha(Path(__file__).with_name('network.py')),reused_development=True,outer_labels_used_in_training=False,full_parent_predictions_used_in_training=False,held_series_normalization=False,training_core_meta_has_in_sample_elements=True)
 savejson(out/'protocol.json',protocol);records=[];start=time.time()
 for fold in map(int,args.folds.split(',')):
  folder=out/f'fold{fold}';folder.mkdir(exist_ok=True)
  if (folder/'result.json').exists():continue
  scope=OLD/f'outputs/models/full_parent_scope_v179/core/fold{fold}';sr=json.loads((scope/'scope.json').read_text())
  assert sr['common_outer_excluded_from_both_nested_bases'] and sr['meta_training_ids_exclude_outer']
  scores=np.zeros((10000,250),np.float32)
  for name in ['training','evaluation']:
   path=scope/f'{name}.npz';assert sha(path)==sr['files'][f'{name}.npz']
   with np.load(path) as z:
    mask=z['tick']%4==0;ids=z['id'][mask];ticks=z['tick'][mask];core=z['core'][mask]
   assert np.all((ids%5!=fold) if name=='training' else (ids%5==fold))
   scores[ids,ticks//4]=logits(core)
  training=np.flatnonzero(np.arange(10000)%5!=fold);held=np.flatnonzero(np.arange(10000)%5==fold)
  x=np.concatenate([scores[:,:,None],xf],axis=-1)
  p=fit(x[training],scores[training],tau[training],L[training],training,x[held],folder,'control',device,args.steps,args.seed)
  x=np.concatenate([x,xa],axis=-1)
  q=fit(x[training],scores[training],tau[training],L[training],training,x[held],folder,'augmented',device,args.steps,args.seed)
  diff=np.clip(q-p,-1,1);np.savez_compressed(folder/'corrections.npz',id=held,difference=diff,control=p,augmented=q)
  rr={'fold':fold,'baseline':metric(y[held],base[held]),'scores':{},'direct_diagnostic':{},'seconds':time.time()-start}
  for alpha in [.1,.25,.5]:
   candidate=base[held].copy();direct=base[held].copy()
   for k,sid in enumerate(held):
    n=L[sid];candidate[k,:n]=expit(logits(base[sid,:n])+alpha*np.repeat(diff[k],4)[:n]);direct[k,:n]=expit(logits(base[sid,:n])+alpha*np.repeat(np.clip(q[k],-1,1),4)[:n])
   rr['scores'][alpha]=metric(y[held],candidate);rr['direct_diagnostic'][alpha]=metric(y[held],direct)
  savejson(folder/'result.json',rr);print('FOLD_RESULT',json.dumps(rr),flush=True);records.append(rr)
 if all((out/f'fold{fold}/result.json').exists() for fold in range(5)):
  candidates={alpha:base.copy() for alpha in [.1,.25,.5]}
  for fold in range(5):
   with np.load(out/f'fold{fold}/corrections.npz') as z:ids=z['id'];d=z['difference']
   for k,sid in enumerate(ids):
    n=L[sid]
    for alpha,p in candidates.items():p[sid,:n]=expit(logits(base[sid,:n])+alpha*np.repeat(d[k],4)[:n])
  score={alpha:metric(y,p) for alpha,p in candidates.items()}
  records=[json.loads((out/f'fold{fold}/result.json').read_text()) for fold in range(5)]
  result=dict(baseline=metric(y,base),scores=score,folds=records,device=device,primary_alpha=.25,seconds=time.time()-start,official_score=False)
  savejson(out/'result.json',result);print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
