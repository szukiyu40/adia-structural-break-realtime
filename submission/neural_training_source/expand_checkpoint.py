"""Warm start an expanded feature bank without changing the initial function."""
import argparse,sys,json
from pathlib import Path
import numpy as np
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--module-dir',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    sys.path.insert(0,str(args.module_dir.resolve()));from network import Detector
    from features import VERSION,CONTEXT_DIM,FEATURE_DIM
    torch.set_num_threads(1)
    bundle=torch.load(args.base,map_location='cpu',weights_only=False);old=bundle['state_dict'];config=bundle['config'];assert config['kind']=='mlp'
    model=Detector('mlp',config['hidden'],config['dropout']);state=model.state_dict();columns={int(v):i for i,v in enumerate(state['indices'])}
    for key in state:
        if key in ['indices','input_encoder.0.weight','context_encoder.0.weight']:continue
        assert state[key].shape==old[key].shape,key
        state[key]=old[key]
    cw=state['context_encoder.0.weight'];cw.zero_();cw[:,:old['context_encoder.0.weight'].shape[1]]=old['context_encoder.0.weight']
    weight=state['input_encoder.0.weight'];weight.zero_()
    for i,feature in enumerate(old['indices']):weight[:,columns[int(feature)]]=old['input_encoder.0.weight'][:,i]
    width=len(state['indices']);oldwidth=len(old['indices']);weight[:,width:]=old['input_encoder.0.weight'][:,oldwidth:]
    model.load_state_dict(state);model.eval();torch.manual_seed(432)
    x=torch.randn(3,17,FEATURE_DIM);c=torch.randn(3,CONTEXT_DIM)
    with torch.inference_mode():
        cc=model.context_encoder(c);xx=torch.cat([x.index_select(-1,old['indices']),cc[:,None].expand(-1,17,-1)],-1)
        enc=torch.nn.functional.silu(torch.nn.functional.linear(xx,old['input_encoder.0.weight'],old['input_encoder.0.bias']))
        expected=model.head(torch.cat([enc,enc],-1)).squeeze(-1);actual=model(x,c)[0]
        error=float((expected-actual).abs().max());assert error<3e-6,error
    config={**config,'feature_version':VERSION,'initialization':'Zero weights on added features; retained trained base function','initial_checkpoint':str(args.base)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':state,'config':config,'parent_epoch':bundle['epoch'],'initialization_max_error':error},args.out)
    print('EXPANDED_CHECKPOINT_PARITY',error,flush=True)

if __name__=='__main__':main()
