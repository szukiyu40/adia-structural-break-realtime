"""Evaluate sign averaging on a frozen model and a specified real split."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(k,'2')
import sys,argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit,logit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--module-dir',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split',default='tune');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--ids-file',type=Path);args=ap.parse_args()
    sys.path.insert(0,str(args.module_dir.resolve()))
    from common import load_cache,ts_auc
    from network import Detector
    import features
    torch.set_num_threads(2)
    if args.ids_file:
        from refit import JoinedFeatures
        from crossvalidate import IndexedFeatures
        maps=[load_cache(args.cache,s) for s in ['train','tune','holdout']]
        full={k:np.concatenate([np.asarray(d[k]) for d in maps]) for k in ['context','labels','ids','lengths']}
        ids=np.load(args.ids_file);positions={sid:i for i,sid in enumerate(full['ids'])};ix=np.array([positions[sid] for sid in ids])
        data={k:v[ix] for k,v in full.items()};data['features']=IndexedFeatures(JoinedFeatures([d['features'] for d in maps]),ix)
    else:data=load_cache(args.cache,args.split)
    bundle=torch.load(args.checkpoint,map_location='cpu',weights_only=False);c=bundle['config']
    model=Detector(c['kind'],c['hidden'],c['dropout']);model.load_state_dict(bundle['state_dict']);model.eval()
    predictions=[np.zeros(data['labels'].shape,np.float32) for _ in range(2)];start=time.time()
    with torch.inference_mode():
        for i in range(0,len(predictions[0]),48):
            j=min(i+48,len(predictions[0]));n=int(data['lengths'][i:j].max())
            x=torch.from_numpy(np.array(data['features'][i:j,:n]));ctx=torch.from_numpy(np.array(data['context'][i:j]))
            predictions[0][i:j,:n]=model(x,ctx)[0].numpy()
            x[:,:,features.AUGMENT_ODD_INDICES]*=-1;ctx[:,getattr(features,'CONTEXT_ODD_INDICES',[2,16,20,21])]*=-1
            predictions[1][i:j,:n]=model(x,ctx)[0].numpy()
    p=logit(np.clip((expit(predictions[0])+expit(predictions[1]))/2,1e-7,1-1e-7))
    args.out.mkdir(parents=True,exist_ok=True)
    np.save(args.out/'logits.npy',p);np.save(args.out/'ids.npy',data['ids'])
    result={'original_ts_auc':ts_auc(data['labels'],predictions[0])[0],'sign_tta_ts_auc':ts_auc(data['labels'],p)[0],
      'seconds':time.time()-start,'checkpoint':str(args.checkpoint),'split':'specified_ids' if args.ids_file else args.split,
      'ids_file':str(args.ids_file) if args.ids_file else None}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
