"""Refit a frozen recipe on selected cache splits without selecting on test scores.

Run only after the independent holdout assessment. The epoch count and learning
rate schedule must be chosen using the train/tune experiment beforehand.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'2')
import argparse,sys,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn

class JoinedFeatures:
    """Gather batches from separate memory maps without materializing the dataset."""
    def __init__(self,arrays):
        self.arrays=arrays
        self.edges=np.r_[0,np.cumsum([len(a) for a in arrays])]
        self.shape=(int(self.edges[-1]),*arrays[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self,key):
        rows,steps=key
        rows=np.arange(self.shape[0])[rows] if isinstance(rows,slice) else np.asarray(rows)
        length=len(np.arange(self.shape[1])[steps])
        out=np.empty((len(rows),length,self.shape[2]),dtype=np.float32)
        for i,a in enumerate(self.arrays):
            mask=(rows>=self.edges[i])&(rows<self.edges[i+1])
            if mask.any():out[mask]=a[rows[mask]-self.edges[i],steps]
        return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--module-dir',type=Path,required=True)
    ap.add_argument('--recipe',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--splits',default='train,tune,holdout')
    ap.add_argument('--epochs',type=int,required=True)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--swa-start',type=int,default=0)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--base-checkpoint',type=Path)
    ap.add_argument('--device',default='auto')
    args=ap.parse_args();sys.path.insert(0,str(args.module_dir.resolve()))
    from common import load_cache
    from network import Detector
    import features
    config=json.loads(args.recipe.read_text())
    recipe=dict(config)
    if args.device=='auto':args.device='mps' if config['kind'] in ('nlinear','mlp') and torch.backends.mps.is_available() else 'cpu'
    config['device']=args.device
    config['resume']=str(args.resume) if args.resume else None
    config.update(seed=args.seed,refit_epochs=args.epochs,refit_splits=args.splits,
                  module_directory=str(args.module_dir),selection='frozen epoch; no evaluation during refit')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    np.random.seed(args.seed);random.seed(args.seed)
    data=[load_cache(args.cache,s) for s in args.splits.split(',')]
    joined={k:np.concatenate([np.asarray(d[k]) for d in data]) for k in ['context','labels','ids','lengths']}
    joined['features']=JoinedFeatures([d['features'] for d in data])
    series_weights=np.concatenate([np.asarray(d.get('series_weights',np.ones(len(d['ids']))),dtype=np.float32) for d in data])
    yall=joined['labels'];pos=((yall==1)*series_weights[:,None]).sum(0);neg=((yall==0)*series_weights[:,None]).sum(0)
    other=torch.tensor(np.stack([pos,neg])/max(len(yall)/2,1),dtype=torch.float32,device=args.device)
    model=Detector(config['kind'],config['hidden'],config['dropout']).to(args.device)
    if hasattr(model,'base') and not args.resume:
        if args.base_checkpoint is None:raise ValueError('A base trained on the same permitted training split is required')
        base=torch.load(args.base_checkpoint,map_location=args.device,weights_only=False)
        model.base.load_state_dict(base['state_dict'])
        config['base_checkpoint']=str(args.base_checkpoint)
    if args.resume:model.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=False)['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,recipe['epochs'],eta_min=config['lr']*.1)
    odd=getattr(features,'AUGMENT_ODD_INDICES',None)
    if odd is None:odd=[b*features.TOTAL_CHANNELS+c for b in range(4) for c in features.ODD_CHANNELS]
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    np.save(args.out/'training_ids.npy',joined['ids'])
    start=time.time();log=[];averaged=None;averaged_count=0
    for epoch in range(1,args.epochs+1):
        model.train();order=np.random.permutation(len(yall));losses=[]
        for st in range(0,len(order),config['batch']):
            ix=order[st:st+config['batch']];length=yall.shape[1] if args.device=='mps' else int(joined['lengths'][ix].max())
            stride=config.get('time_stride',1);phase=np.random.randint(stride);steps=slice(phase,length,stride)
            x=torch.from_numpy(joined['features'][ix,steps]).to(args.device)
            c=torch.from_numpy(joined['context'][ix].copy()).to(args.device)
            sign=torch.where(torch.rand(len(ix),device=args.device)<.5,-1.,1.)
            x[:,:,odd]*=sign[:,None,None];c[:,getattr(features,'CONTEXT_ODD_INDICES',[2,16,20,21])]*=sign[:,None]
            y=torch.from_numpy(yall[ix,steps].copy()).to(args.device);valid=y>=0;target=y.clamp(min=0).float()
            logits,_=model(x,c)
            weight=torch.where(target>0,other[1,steps],other[0,steps])*valid
            weight*=torch.tensor(series_weights[ix],device=args.device)[:,None]
            if config.get('positive_ramp',0):
                ages=np.cumsum(yall[ix,:length]==1,axis=1)[:,steps]
                ramp=torch.tensor(np.minimum(ages/config['positive_ramp'],1),device=args.device,dtype=torch.float32)
                weight*=torch.where(target>0,ramp,1.)
            loss=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).sum()/weight.sum().clamp(min=1)
            if config.get('pair_weight',0):
                if config.get('pair_mode','roll')=='all':
                    pm=(y[:,None,:]==1)&(y[None,:,:]==0);delta=logits[:,None,:]-logits[None,:,:]
                else:
                    roll=torch.randperm(len(ix),device=args.device);pm=(y==1)&(y[roll]==0);delta=logits-logits[roll]
                if pm.any():loss=loss+config['pair_weight']*nn.functional.softplus(-delta[pm]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step();losses.append(float(loss.detach()))
        schedule.step()
        if args.swa_start and epoch>=args.swa_start:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};averaged_count+=1
            if averaged is None:averaged=state
            else:
                for k,v in state.items():
                    if v.is_floating_point():averaged[k]+=(v-averaged[k])/averaged_count
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'seconds':round(time.time()-start,2)}
        log.append(row);print(json.dumps(row),flush=True)
        (args.out/'learning_curve.json').write_text(json.dumps(log,indent=2))
    if averaged is not None:
        model.load_state_dict(averaged);config.update(selected_average=True,swa_start=args.swa_start,averaged_epochs=averaged_count)
    model.eval()
    torch.save({'state_dict':model.cpu().state_dict(),'config':config,'epoch':args.epochs,
                'training_series':len(yall),'selection':'frozen recipe; all-data refit'},args.out/'best.pt')
    (args.out/'result.json').write_text(json.dumps({'training_series':len(yall),'seconds':time.time()-start,'config':config},indent=2))
    print('REFIT_COMPLETE',len(yall),flush=True)

if __name__=='__main__':main()
