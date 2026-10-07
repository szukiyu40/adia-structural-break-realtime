"""Cross-validation using fixed epochs; validation labels never select checkpoints."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(k,'2')
import argparse,sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from refit import JoinedFeatures

class IndexedFeatures:
    def __init__(self,base,indices):
        self.base=base;self.indices=np.asarray(indices);self.shape=(len(indices),*base.shape[1:])
    def __len__(self):return len(self.indices)
    def __getitem__(self,key):
        rows,steps=key
        return self.base[self.indices[rows],steps]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--module-dir',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--recipe',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fold',type=int,required=True)
    ap.add_argument('--folds',type=int,default=3);ap.add_argument('--epochs',type=int,required=True)
    ap.add_argument('--swa-start',type=int,default=0);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--device',default='auto');ap.add_argument('--base-models',type=Path)
    ap.add_argument('--resume-models',type=Path);ap.add_argument('--synthetic-cache',type=Path)
    ap.add_argument('--synthetic-weight',type=float,default=.35);ap.add_argument('--prefix-features',action='store_true');args=ap.parse_args()
    sys.path.insert(0,str(args.module_dir.resolve()))
    import common
    maps=[common.load_cache(args.cache,s) for s in ['train','tune','holdout']]
    full={k:np.concatenate([np.asarray(d[k]) for d in maps]) for k in ['context','labels','ids','lengths']}
    full['features']=JoinedFeatures([d['features'] for d in maps])
    manifest=pd.read_csv(Path(__file__).parent/'model_split_manifest.csv').set_index('id')
    folds=np.array([int(manifest.loc[int(sid),'history_sha256'][32:48],16)%args.folds for sid in full['ids']])
    subsets={}
    for name,mask in [('cvtrain',folds!=args.fold),('cvvalidation',folds==args.fold)]:
        ix=np.flatnonzero(mask)
        subsets[name]={k:v[ix] for k,v in full.items() if k!='features'}
        subsets[name]['features']=IndexedFeatures(full['features'],ix)
    assert not np.intersect1d(subsets['cvtrain']['ids'],subsets['cvvalidation']['ids']).size
    real_training_ids=np.array(subsets['cvtrain']['ids'])
    args.out.mkdir(parents=True,exist_ok=True)
    if args.synthetic_cache:
        from augmentation import augment_training
        subsets['cvtrain'],protocol=augment_training(subsets['cvtrain'],args.synthetic_cache,args.synthetic_weight,args.prefix_features)
        source_parents=np.load(args.synthetic_cache/'all'/'parent_ids.npy')
        selected_parents=source_parents[np.isin(source_parents,real_training_ids)]
        assert not np.intersect1d(selected_parents,subsets['cvvalidation']['ids']).size
        (args.out/'augmentation_protocol.json').write_text(json.dumps(protocol,indent=2))
    (args.out/'fold_protocol.json').write_text(json.dumps({'fold':args.fold,'fold_count':args.folds,
        'assignment':'int(history_sha256[32:48],16) % fold_count','fixed_epochs':args.epochs,'swa_start':args.swa_start,
        'training_series':len(subsets['cvtrain']['ids']),'validation_series':len(subsets['cvvalidation']['ids']),
        'selection':'Epochs frozen before fold training. No validation evaluation until fitting is complete.'},indent=2))
    common.load_cache=lambda path,split:subsets[split]
    import refit
    sys.argv=['refit.py','--module-dir',str(args.module_dir),'--recipe',str(args.recipe),'--cache',str(args.cache),
       '--out',str(args.out),'--splits','cvtrain','--epochs',str(args.epochs),'--seed',str(args.seed),
       '--threads',str(args.threads),'--swa-start',str(args.swa_start),'--device',args.device]
    if args.base_models:
        base_folder=args.base_models/f'fold-{args.fold}'
        base_ids=np.load(base_folder/'training_ids.npy')
        assert np.array_equal(np.sort(base_ids),np.sort(real_training_ids)),'Backbone training set must exactly match the permitted real fold training set'
        sys.argv.extend(['--base-checkpoint',str(base_folder/'best.pt')])
    if args.resume_models:
        resume_folder=args.resume_models/f'fold-{args.fold}'
        resume_ids=np.load(resume_folder/'training_ids.npy')
        assert np.array_equal(np.sort(resume_ids),np.sort(real_training_ids)),'Initial checkpoint must be trained on exactly the permitted real fold training set'
        sys.argv.extend(['--resume',str(resume_folder/'best.pt')])
    refit.main()
    import torch
    from network import Detector
    from train import predict
    bundle=torch.load(args.out/'best.pt',map_location='cpu',weights_only=False);c=bundle['config']
    model=Detector(c['kind'],c['hidden'],c['dropout']);model.load_state_dict(bundle['state_dict'])
    val=subsets['cvvalidation'];pred=predict(model,val,batch_size=64)
    score=common.ts_auc(val['labels'],pred)[0]
    np.save(args.out/'validation_logits.npy',pred);np.save(args.out/'validation_ids.npy',val['ids'])
    (args.out/'validation_result.json').write_text(json.dumps({'ts_auc':score,'series':len(pred),'fold':args.fold,'fixed_epochs':args.epochs},indent=2))
    print('FOLD_VALIDATION_SCORE',score,flush=True)

if __name__=='__main__':main()
