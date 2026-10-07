"""GPU pinball training with complete optimizer and RNG checkpoints."""
import argparse, time
import numpy as np
import torch
from common import *
from network import Predictor, inputs

def train(fold,inner,steps=800,batch=64,out=None):
    out=Path(out) if out else OUT/f'fold{fold}'/f'predictor{inner}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): return
    d=loads(); ids=np.arange(10000)
    fit=ids[(ids%5!=fold)&((ids//5)%2==inner)]
    assert len(fit)==4000
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(2); torch.manual_seed(SEED+10*fold+inner)
    rng=np.random.default_rng(SEED+10*fold+inner)
    model=Predictor().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    checkpoint=out/'latest.pt'; step0=0; losses=[]
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['fit_ids']==fit.tolist() and ck['steps']==steps
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['numpy_rng']; torch.set_rng_state(ck['torch_rng'])
        step0=ck['step']; losses=ck['losses']
    write(out/'protocol.json',{'fold':fold,'inner':inner,'fit_ids':fit.tolist(),'steps':steps,
        'batch_size':batch,'device':device,'parameters':sum(p.numel() for p in model.parameters()),
        'objective':'9 quantile pinball; normal historical second halves only',
        'label_or_baseline_used':False,'torch':torch.__version__})
    start=time.time(); model.train()
    for step in range(step0,steps):
        sid=rng.choice(fit,batch)
        h=d['hist_lengths'][sid]
        begin=h//2+rng.integers(0,2**30,batch)%(h-h//2-193)
        ix=begin[:,None]+np.arange(193)[None,:]
        v=np.asarray(d['values'][sid[:,None],ix]); r=np.asarray(d['innovations'][sid[:,None],ix])
        x=torch.from_numpy(inputs(v[:,:-1],r[:,:-1])).to(device)
        static=torch.from_numpy(np.array(d['static'][sid])).to(device)
        target=torch.from_numpy(np.clip(r[:,1:],-25,25)).to(device)
        opt.zero_grad(set_to_none=True)
        q=model(x,static)
        loss=model.loss(q[:,:,64:],target[:,64:])
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step+1)%100==0 or step+1==steps:
            state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer':opt.state_dict(),'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'step':step+1,'steps':steps,
                'fit_ids':fit.tolist(),'losses':losses}
            tmp=out/'latest.tmp.pt'; torch.save(state,tmp); tmp.replace(checkpoint)
            # Full checkpoints, including optimizer, retained every 200 steps.
            if (step+1)%200==0 or step+1==steps: torch.save(state,out/f'step{step+1}.pt')
            record={'fold':fold,'inner':inner,'step':step+1,'loss_recent':float(np.mean(losses[-100:])),
                'seconds_this_session':time.time()-start}
            write(out/'progress.json',record); print(json.dumps(record),flush=True)
    write(out/'complete.json',{'steps':steps,'seconds_this_session':time.time()-start,
        'loss_first100':float(np.mean(losses[:100])),'loss_last100':float(np.mean(losses[-100:])),
        'checkpoint_sha256':sha(checkpoint),'device':device})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fold',type=int,default=0); p.add_argument('--inner',type=int,default=0)
    p.add_argument('--steps',type=int,default=800); p.add_argument('--out'); a=p.parse_args()
    train(a.fold,a.inner,a.steps,out=a.out)
