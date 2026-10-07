"""Learn a bounded, causal GRU correction to fixed per-series NLinear scores.

Training and validation base models must exclude validation series. The score
folders contain already frozen TTA logits, so no base-model weights are changed.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'2')
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
import torch
from torch import nn
from common import ts_auc


def evidence(z):
    z=np.clip(z,-6,6).astype(np.float32)
    diff=np.diff(z,axis=1,prepend=z[:,:1])
    columns=[z/3,diff,np.abs(diff)]
    for width in [8,32,128]:
        alpha=2/(width+1)
        mean=lfilter([alpha],[1,-(1-alpha)],z,axis=1).astype(np.float32)
        movement=lfilter([alpha],[1,-(1-alpha)],np.maximum(diff,0),axis=1).astype(np.float32)
        columns.extend([mean/3,(z-mean)/3,movement])
    t=np.arange(z.shape[1],dtype=np.float32)[None,:]
    columns.extend([np.broadcast_to(np.log1p(t)/np.log(1001),z.shape),
                    np.broadcast_to(1/np.sqrt(t+1),z.shape)])
    return np.stack(columns,axis=-1).astype(np.float32)


class EvidenceGRU(nn.Module):
    def __init__(self,hidden=12,bound=.5,context_dim=43):
        super().__init__();self.bound=bound
        self.initial=nn.Sequential(nn.Linear(context_dim,hidden),nn.Tanh())
        self.gru=nn.GRU(14,hidden,batch_first=True)
        self.output=nn.Linear(hidden,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,c,state=None):
        if state is None:state=self.initial(c).unsqueeze(0)
        h,state=self.gru(x,state)
        return self.bound*torch.tanh(self.output(h).squeeze(-1)),state


def load(folder,cache):
    ids=np.load(folder/'ids.npy');z=np.load(folder/'logits.npy')
    all_ids=np.concatenate([np.load(cache/s/'ids.npy') for s in ['train','tune','holdout']])
    pos={int(s):i for i,s in enumerate(all_ids)};ix=np.array([pos[int(s)] for s in ids])
    y=np.concatenate([np.load(cache/s/'labels.npy') for s in ['train','tune','holdout']])[ix]
    c=np.concatenate([np.load(cache/s/'context.npy') for s in ['train','tune','holdout']])[ix]
    c[:,[2,16,20,21]]=np.abs(c[:,[2,16,20,21]])
    c=np.clip(c*.2,-4,4).astype(np.float32)
    return {'ids':ids,'z':z.astype(np.float32),'x':evidence(z),'c':c,'y':y}


@torch.inference_mode()
def predict(model,data,batch=64):
    model.eval();out=np.empty_like(data['z'])
    for s in range(0,len(out),batch):
        corr,_=model(torch.from_numpy(data['x'][s:s+batch]),torch.from_numpy(data['c'][s:s+batch]))
        out[s:s+batch]=data['z'][s:s+batch]+corr.numpy()
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-base',type=Path,required=True);ap.add_argument('--valid-base',type=Path)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--hidden',type=int,default=12);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--bound',type=float,default=.5);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=2012);ap.add_argument('--fixed-epoch',action='store_true');ap.add_argument('--refit',action='store_true');args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    train=load(args.train_base,args.cache);valid=None if args.refit else load(args.valid_base,args.cache)
    if valid is not None:assert not set(train['ids']).intersection(valid['ids'])
    args.out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['context_dim']=int(train['c'].shape[1])
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    np.save(args.out/'training_ids.npy',train['ids'])
    if valid is not None:np.save(args.out/'validation_ids.npy',valid['ids'])
    yall=train['y'];pos=(yall==1).sum(0);neg=(yall==0).sum(0)
    other=torch.tensor(np.stack([pos,neg])/max(len(yall)/2,1),dtype=torch.float32)
    model=EvidenceGRU(args.hidden,args.bound,config['context_dim']);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,12,eta_min=args.lr*.1)
    base=None if args.refit else ts_auc(valid['y'],valid['z'])[0];best=base;rows=[];started=time.time()
    print('BASE_VALIDATION',base,flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();order=np.random.permutation(len(yall));losses=[]
        for start in range(0,len(order),args.batch):
            ix=order[start:start+args.batch];length=int((yall[ix]>=0).sum(1).max())
            x=torch.from_numpy(train['x'][ix,:length]);c=torch.from_numpy(train['c'][ix])
            z=torch.from_numpy(train['z'][ix,:length]);y=torch.from_numpy(yall[ix,:length])
            corr,_=model(x,c);logits=z+corr
            phase=np.random.randint(8);zsub=logits[:,phase::8];ysub=y[:,phase::8]
            mask=(ysub[:,None,:]==1)&(ysub[None,:,:]==0)
            delta=zsub[:,None,:]-zsub[None,:,:]
            loss=nn.functional.softplus(-delta[mask]).mean()
            target=y.clamp(min=0).float();weight=torch.where(target>0,other[1,:length],other[0,:length])*(y>=0)
            bce=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).sum()/weight.sum().clamp(min=1)
            loss=loss+.1*bce+.01*(corr.square()*(y>=0)).sum()/(y>=0).sum()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
        scheduler.step()
        if args.refit or (args.fixed_epoch and epoch<args.epochs):
            row={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':round(time.time()-started,1)}
        else:
            pred=predict(model,valid,args.batch);score=ts_auc(valid['y'],pred)[0]
            row={'epoch':epoch,'loss':float(np.mean(losses)),'valid_ts_auc':score,'seconds':round(time.time()-started,1)}
            if score>best or args.fixed_epoch:
                best=score if args.fixed_epoch else max(best,score)
                torch.save({'state_dict':model.state_dict(),'config':config,'epoch':epoch},args.out/'best.pt')
                np.save(args.out/'validation_logits.npy',pred)
        rows.append(row);print(json.dumps(row),flush=True);(args.out/'learning_curve.json').write_text(json.dumps(rows,indent=2))
    if args.refit:
        torch.save({'state_dict':model.state_dict(),'config':config,'epoch':args.epochs,'training_series':len(train['ids'])},args.out/'best.pt')
    result={'base_ts_auc':base,'best_ts_auc':best,'difference':None if args.refit else best-base,'seconds':time.time()-started,'config':config}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
