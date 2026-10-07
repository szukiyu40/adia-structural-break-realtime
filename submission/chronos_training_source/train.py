"""Supervised last-two-block adaptation of Chronos-2; independent held series.

Run from the extracted private Colab packet. Parent predictions are evaluation
only. No target, future observation, or other series enters the encoder inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from scipy.stats import rankdata
from chronos import Chronos2Pipeline

MODEL_ID = 'amazon/chronos-2'
REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'
SEED = 91827
STRIDE = 16
BLEND = .125


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def metric(y, p):
    wins = 0.; pairs = 0
    for t in range(y.shape[1]):
        ok = y[:, t] >= 0
        yy, pp = y[ok, t], p[ok, t]
        n, m = int((yy == 1).sum()), int((yy == 0).sum())
        if n*m:
            wins += rankdata(pp)[yy == 1].sum() - n*(n+1)/2
            pairs += n*m
    return float(wins/pairs)


class Detector(nn.Module):
    def __init__(self, device):
        super().__init__()
        pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=REVISION,
            device_map=device, dtype=torch.float32, trust_remote_code=False)
        self.encoder_model = pipeline.model
        for p in self.encoder_model.parameters():
            p.requires_grad_(False)
        for block in self.encoder_model.encoder.block[-2:]:
            for p in block.parameters(): p.requires_grad_(True)
        for p in self.encoder_model.encoder.final_layer_norm.parameters():
            p.requires_grad_(True)
        d = int(self.encoder_model.config.d_model)
        assert d == 768 and self.encoder_model.chronos_config.use_reg_token
        self.head = nn.Sequential(nn.LayerNorm(4*d+6), nn.Linear(4*d+6,64),
            nn.GELU(), nn.Dropout(.1), nn.Linear(64,1)).to(device)
        self.encoder_model.eval()

    def forward(self, history, current, ticks):
        n = len(current)
        context = torch.cat([history, current], 0)
        # Every view is a distinct group: no cross-series or cross-view attention.
        group_ids = torch.arange(2*n, device=context.device)
        enc, _, _, patches = self.encoder_model.encode(context, group_ids=group_ids)
        reg = enc.last_hidden_state[:, patches]
        h, c = reg[:n], reg[n:]
        hc = torch.nan_to_num(history)
        cc = torch.nan_to_num(current)
        hm = hc.sum(1)/torch.isfinite(history).sum(1).clamp_min(1)
        cm = cc.sum(1)/torch.isfinite(current).sum(1).clamp_min(1)
        hs = torch.sqrt(torch.nanmean((history-hm[:,None]).square(),1).clamp_min(1e-6))
        cs = torch.sqrt(torch.nanmean((current-cm[:,None]).square(),1).clamp_min(1e-6))
        scalars = torch.stack([cm-hm, torch.log(cs/hs), torch.abs(cm-hm),
            torch.log1p(ticks.float())/7, (cc[:,-1]-hm)/hs,
            torch.isfinite(current).float().mean(1)],1).clamp(-8,8)
        features = torch.cat([h,c,c-h,torch.abs(c-h),scalars],1)
        return self.head(features).squeeze(1)


def windows(bank, ids, ticks, device, signs=None):
    # A tick t ends at column 1024+t inclusive. No padding after t is read.
    history = np.asarray(bank[ids,:1024]).copy()
    current = bank[ids[:,None], ticks[:,None]+1+np.arange(1024)[None,:]].copy()
    if signs is not None:
        history *= signs[:,None]; current *= signs[:,None]
    return (torch.from_numpy(history).to(device), torch.from_numpy(current).to(device),
            torch.from_numpy(np.asarray(ticks,dtype='i8')).to(device))


def train(args):
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data = Path(args.data)
    manifest = json.loads((data/'data_manifest.json').read_text())
    for name, item in manifest.items():
        with (data/name).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest() == item['sha256'], name
    bank = np.load(data/'causal_bank.npy',mmap_mode='r')
    labels = np.load(data/'labels.npy')
    lengths = np.load(data/'lengths.npy')
    assert bank.shape == (10000,2024) and labels.shape == (10000,1000)
    ids = np.arange(10000)
    held = ids[ids%5 == args.fold] if args.fold >= 0 else np.array([],dtype=int)
    fit = ids[ids%5 != args.fold] if args.fold >= 0 else ids
    assert not np.intersect1d(fit,held).size
    seed = SEED + max(args.fold,0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device = 'cuda'
    assert torch.cuda.is_available(), 'CUDA required for this experiment'
    model = Detector(device)
    trainable_encoder = [p for p in model.encoder_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params':trainable_encoder, 'lr':1e-5}, {'params':model.head.parameters(),'lr':3e-4}],
        weight_decay=.01)
    sampler = np.random.default_rng(seed)
    positives = [fit[labels[fit,t]==1] for t in range(1000)]
    negatives = [fit[labels[fit,t]==0] for t in range(1000)]
    pair_weight = np.array([len(a)*len(b) for a,b in zip(positives,negatives)],dtype=float)
    pair_weight /= pair_weight.sum()
    write(out/'protocol.json', {'model':MODEL_ID,'revision':REVISION,'seed':seed,
        'train_ids':fit.tolist(),'held_ids':held.tolist(),'steps':args.steps,'pairs_per_step':args.pairs,
        'context':1024,'reference':'last 1024 normal historical points','stride':STRIDE,
        'blend_candidate_probability':BLEND,'learned_encoder_blocks':'last two plus final layer norm',
        'loss':'same-tick pairwise logistic + 0.1 balanced BCE',
        'parent_predictions_used_in_training':False,'development_reused':True,
        'official_score':False,'early_selection':'none; fixed final update',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'adoption_gate':{'five_fold_delta_gt':.0005,'positive_folds_at_least':4}})
    start=time.time(); history=[]
    model.head.train(); model.encoder_model.eval()
    for step in range(args.steps):
        tick = int(sampler.choice(1000,p=pair_weight))
        a = sampler.choice(positives[tick],args.pairs)
        b = sampler.choice(negatives[tick],args.pairs)
        batch_ids=np.r_[a,b]; ticks=np.full(len(batch_ids),tick,dtype='i8')
        signs=sampler.choice(np.array([-1.,1.],dtype='f4'),len(batch_ids))
        optimizer.zero_grad(set_to_none=True)
        logits=model(*windows(bank,batch_ids,ticks,device,signs))
        y=torch.cat([torch.ones(args.pairs,device=device),torch.zeros(args.pairs,device=device)])
        loss=F.softplus(-(logits[:args.pairs]-logits[args.pairs:])).mean()+.1*F.binary_cross_entropy_with_logits(logits,y)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.head.parameters())+trainable_encoder,1.)
        optimizer.step()
        history.append(float(loss.detach()))
        if (step+1)%100==0 or step==0 or step+1==args.steps:
            torch.cuda.synchronize()
            record={'step':step+1,'loss_mean_recent':float(np.mean(history[-100:])),
                'elapsed_seconds':time.time()-start,'gpu_peak_gb':torch.cuda.max_memory_allocated()/1e9}
            print(json.dumps(record),flush=True); write(out/'progress.json',record)
    model.eval()
    ck={'trainable_encoder':{name:p.detach().cpu() for name,p in model.encoder_model.named_parameters() if p.requires_grad},
        'head':{name:v.detach().cpu() for name,v in model.head.state_dict().items()},
        'revision':REVISION,'model':MODEL_ID}
    torch.save(ck,out/'adapter.pt')
    # Test view/series independence and save/reload parity on observed prefixes.
    probe_ids=fit[:4]; probe_ticks=np.minimum(np.array([0,15,63,127]),lengths[probe_ids]-1)
    inp=windows(bank,probe_ids,probe_ticks,device)
    with torch.no_grad():
        first=model(*inp).cpu().numpy()
        second=model(*inp).cpu().numpy()
        reverse=model(*(v.flip(0) for v in inp)).flip(0).cpu().numpy()
    assert np.max(np.abs(first-second)) < 1e-6
    assert np.max(np.abs(first-reverse)) < 1e-4
    write(out/'probe.json',{'repeat_max_abs':float(np.max(np.abs(first-second))),
        'reverse_max_abs':float(np.max(np.abs(first-reverse))),
        'training_seconds':time.time()-start,'gpu_peak_gb':torch.cuda.max_memory_allocated()/1e9,
        'note':'GPU research probe; strict 1e-8 deployment determinism is a later separate gate'})
    if args.smoke or args.fold < 0:
        print('SMOKE_OR_FULLFIT_COMPLETE',flush=True); return
    probability=np.full((len(held),1000),.5,dtype='f4')
    endpoints=[(i,int(sid),t) for i,sid in enumerate(held) for t in range(0,int(lengths[sid]),STRIDE)]
    for lo in range(0,len(endpoints),args.eval_batch):
        piece=np.array(endpoints[lo:lo+args.eval_batch])
        with torch.no_grad():
            value=torch.sigmoid(model(*windows(bank,piece[:,1],piece[:,2],device))).cpu().numpy()
        for (i,sid,t),v in zip(piece,value): probability[i,t:min(t+STRIDE,int(lengths[sid]))]=v
        if lo%(args.eval_batch*100)==0:
            print(json.dumps({'eval_done':lo,'eval_total':len(endpoints),'elapsed_seconds':time.time()-start}),flush=True)
    np.save(out/'probability.npy',probability)
    np.save(out/'held_ids.npy',held)
    # Parent probabilities are loaded only after training and prediction finish.
    parent=np.load(data/'parent.npy',mmap_mode='r')[held]
    base=metric(labels[held],parent)
    score=metric(labels[held],(1-BLEND)*parent+BLEND*probability)
    result={'fold':args.fold,'parent':base,'direct':metric(labels[held],probability),'blend':score,
        'delta':score-base,'total_seconds':time.time()-start,'official_score':False}
    write(out/'result.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='/content/sbrt/data')
    parser.add_argument('--out',required=True)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--steps',type=int,default=1600)
    parser.add_argument('--pairs',type=int,default=16)
    parser.add_argument('--eval-batch',type=int,default=64)
    parser.add_argument('--smoke',action='store_true')
    train(parser.parse_args())
