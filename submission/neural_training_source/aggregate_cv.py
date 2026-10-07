"""Stitch exactly one held-out prediction per series from fixed-epoch folds."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
from common import load_cache,ts_auc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True);ap.add_argument('--folds',type=int,default=3)
    args=ap.parse_args()
    maps=[load_cache(args.cache,s) for s in ['train','tune','holdout']]
    ids=np.concatenate([np.asarray(d['ids']) for d in maps]);y=np.concatenate([np.asarray(d['labels']) for d in maps])
    lookup={int(s):i for i,s in enumerate(ids)};pred=np.zeros(y.shape,np.float32);seen=np.zeros(len(y),bool);fold_scores=[]
    for f in range(args.folds):
        p=args.models/f'fold-{f}'
        validation_ids=np.load(p/'validation_ids.npy');values=np.load(p/'validation_logits.npy')
        ix=np.array([lookup[int(s)] for s in validation_ids]);assert not seen[ix].any()
        pred[ix]=values;seen[ix]=True;fold_scores.append(ts_auc(y[ix],values)[0])
    assert seen.all(),'Some series have no held-out prediction'
    score=ts_auc(y,pred)[0];np.save(args.models/'oof_logits.npy',pred);np.save(args.models/'oof_ids.npy',ids)
    result={'oof_ts_auc':score,'fold_ts_auc':fold_scores,'series':len(ids),'exactly_once_out_of_fold':True,
            'checkpoint_selection':'Fixed epoch counts; no fold validation checkpoint selection.',
            'note':'Architecture was chosen using an earlier train/tune experiment; this is cross-validation, not an untouched test set.'}
    (args.models/'oof_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
