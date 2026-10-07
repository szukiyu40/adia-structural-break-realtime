"""Train on train split, select epochs only on tune split. Never reads holdout."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from network import Detector
from common import load_cache,ts_auc
from features import AUGMENT_ODD_INDICES,VERSION

def predict(model,data,batch_size=64,device='cpu'):
    result=np.zeros(data['labels'].shape,dtype=np.float32);model.eval()
    with torch.inference_mode():
        for start in range(0,len(result),batch_size):
            end=min(len(result),start+batch_size);length=data['labels'].shape[1] if device=='mps' else int(data['lengths'][start:end].max())
            x=torch.tensor(np.array(data['features'][start:end,:length]),device=device)
            c=torch.tensor(np.array(data['context'][start:end]),device=device)
            logits,_=model(x,c)
            result[start:end,:length]=logits.cpu().numpy()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--kind',choices=['nlinear','nlinear_gru','gru','hybrid','mlp'],required=True)
    ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--hidden',type=int,default=48)
    ap.add_argument('--batch',type=int,default=48);ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--dropout',type=float,default=.1)
    ap.add_argument('--weight-decay',type=float,default=.001);ap.add_argument('--pair-weight',type=float,default=0)
    ap.add_argument('--pair-mode',choices=['roll','all'],default='roll');ap.add_argument('--threads',type=int,default=4);ap.add_argument('--device',default='cpu')
    ap.add_argument('--time-stride',type=int,default=1);ap.add_argument('--positive-ramp',type=int,default=0);ap.add_argument('--swa-start',type=int,default=0);ap.add_argument('--resume',type=Path);args=ap.parse_args()
    if args.time_stride!=1 and args.kind not in ('nlinear','mlp'):raise ValueError('Temporal subsampling requires a pointwise head')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    data=load_cache(args.cache,'train');tune=load_cache(args.cache,'tune')
    yall=np.asarray(data['labels']);series_weights=np.asarray(data.get('series_weights',np.ones(len(yall))),dtype=np.float32)
    pos=((yall==1)*series_weights[:,None]).sum(0).astype(float);neg=((yall==0)*series_weights[:,None]).sum(0).astype(float)
    other=torch.tensor(np.stack([pos,neg])/max(len(yall)/2,1),dtype=torch.float32,device=args.device)
    model=Detector(args.kind,args.hidden,args.dropout).to(args.device)
    if args.resume:model.load_state_dict(torch.load(args.resume,map_location=args.device,weights_only=False)['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=args.lr*.1)
    best=-1;start=time.time();log=[];averaged=None;averaged_count=0
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};config['feature_version']=VERSION;config.update(globals().get('CONFIG_EXTRA',{}))
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    np.save(args.out/'tune_ids.npy',np.array(tune['ids']))
    odd=AUGMENT_ODD_INDICES
    for epoch in range(1,args.epochs+1):
        model.train();order=np.random.permutation(len(yall));losses=[]
        for startidx in range(0,len(order),args.batch):
            ix=order[startidx:startidx+args.batch];length=data['labels'].shape[1] if args.device=='mps' else int(data['lengths'][ix].max())
            phase=np.random.randint(args.time_stride);steps=slice(phase,length,args.time_stride)
            x=torch.tensor(np.array(data['features'][ix,steps]),device=args.device)
            c=torch.tensor(np.array(data['context'][ix]),device=args.device)
            # Sign augmentation acts on standardized odd channels and signed history summaries.
            sign=torch.where(torch.rand(len(ix),device=args.device)<.5,-1.,1.)
            x[:,:,odd]*=sign[:,None,None]
            c[:,[2,16,20,21]]*=sign[:,None]
            y=torch.tensor(np.array(data['labels'][ix,steps]),device=args.device)
            valid=y>=0;target=y.clamp(min=0).float()
            logits,_=model(x,c)
            weight=torch.where(target>0,other[1,steps],other[0,steps])*valid
            weight*=torch.tensor(series_weights[ix],device=args.device)[:,None]
            if args.positive_ramp:
                ages=np.cumsum(yall[ix,:length]==1,axis=1)[:,steps]
                ramp=torch.tensor(np.minimum(ages/args.positive_ramp,1),device=args.device,dtype=torch.float32)
                weight*=torch.where(target>0,ramp,1.)
            loss=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).sum()/weight.sum().clamp(min=1)
            if args.pair_weight:
                # Compare series only at identical online timesteps.
                if args.pair_mode=='all':
                    pm=(y[:,None,:]==1)&(y[None,:,:]==0)
                    delta=logits[:,None,:]-logits[None,:,:]
                else:
                    rolled=torch.randperm(len(ix),device=args.device)
                    pm=(y==1)&(y[rolled]==0);delta=logits-logits[rolled]
                if pm.any():loss=loss+args.pair_weight*nn.functional.softplus(-delta[pm]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step()
        if args.swa_start and epoch>=args.swa_start:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            averaged_count+=1
            if averaged is None:averaged=state
            else:
                for k,v in state.items():
                    if v.is_floating_point():averaged[k]+=(v-averaged[k])/averaged_count
        pred=predict(model,tune,args.batch,args.device)
        score,_,_=ts_auc(np.asarray(tune['labels']),pred)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'tune_ts_auc':score,'seconds':round(time.time()-start,2)}
        log.append(row);print(json.dumps(row),flush=True)
        (args.out/'learning_curve.json').write_text(json.dumps(log,indent=2))
        if score>best:
            best=score
            torch.save({'state_dict':model.cpu().state_dict(),'config':config,'epoch':epoch,'tune_ts_auc':score},args.out/'best.pt')
            model.to(args.device);np.save(args.out/'tune_logits.npy',pred)
    if averaged is not None:
        model.load_state_dict(averaged)
        pred=predict(model,tune,args.batch,args.device);swa_score=ts_auc(np.asarray(tune['labels']),pred)[0]
        swa_config={**config,'selected_average':True,'averaged_epochs':averaged_count}
        torch.save({'state_dict':averaged,'config':swa_config,'epoch':args.epochs,'tune_ts_auc':swa_score},args.out/'swa.pt')
        np.save(args.out/'swa_tune_logits.npy',pred)
        (args.out/'swa_result.json').write_text(json.dumps({'tune_ts_auc':swa_score,'averaged_epochs':averaged_count,'config':swa_config},indent=2))
        print('AVERAGED_SCORE',swa_score,flush=True)
    (args.out/'result.json').write_text(json.dumps({'best_tune_ts_auc':best,'seconds':time.time()-start,'config':config},indent=2))
    print('TRAINING_COMPLETE',best,flush=True)

if __name__=='__main__':main()
