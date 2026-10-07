import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from features import FeatureStream,FEATURE_DIM,CONTEXT_DIM,VERSION
from common import series_stream

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=Path(__file__).parent/'model_split_manifest.csv')
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--splits',default='train,tune')
    args=ap.parse_args();manifest=pd.read_csv(args.manifest).set_index('id')
    labels=pd.read_parquet(args.data/'y_train_index.parquet');splits=args.splits.split(',')
    stores={};counter={}
    for split in splits:
        p=args.cache/split;p.mkdir(parents=True,exist_ok=True)
        count=int((manifest.model_split==split).sum());counter[split]=0
        stores[split]={
          'features':np.lib.format.open_memmap(p/'features.npy',mode='w+',dtype='float32',shape=(count,1000,FEATURE_DIM)),
          'context':np.lib.format.open_memmap(p/'context.npy',mode='w+',dtype='float32',shape=(count,CONTEXT_DIM)),
          'labels':np.lib.format.open_memmap(p/'labels.npy',mode='w+',dtype='int8',shape=(count,1000)),
          'ids':np.lib.format.open_memmap(p/'ids.npy',mode='w+',dtype='int64',shape=(count,)),
          'lengths':np.lib.format.open_memmap(p/'lengths.npy',mode='w+',dtype='int64',shape=(count,))}
        stores[split]['labels'][:]=-1
    start=time.time();processed=0
    for a in series_stream(args.data/'X_train.parquet'):
        sid=int(a['id'][0]);split=manifest.loc[sid,'model_split']
        if split not in splits:continue
        h=a['value'][a['period']==1];x=a['value'][a['period']==2];tau=int(labels.loc[sid,'tau_index'])
        stream=FeatureStream(h);f=stream.batch(x)
        if processed<3:
            causal=np.stack([stream.consume(v) for v in x]);diff=float(np.max(np.abs(causal-f)))
            assert diff<2e-6, diff
            n=min(100,len(x));prefix=FeatureStream(h).batch(x[:n]);assert np.array_equal(prefix,f[:n])
        pos=counter[split];store=stores[split]
        store['features'][pos,:len(x)]=f;store['context'][pos]=stream.ctx
        store['ids'][pos]=sid;store['lengths'][pos]=len(x)
        store['labels'][pos,:len(x)]=(np.arange(len(x))>=tau) if tau>=0 else 0
        counter[split]+=1;processed+=1
        if processed%1000==0:print(json.dumps({'processed':processed,'seconds':round(time.time()-start,1),'splits':counter}),flush=True)
    for split,store in stores.items():
        assert counter[split]==len(store['ids'])
        for arr in store.values():arr.flush()
        (args.cache/split/'metadata.json').write_text(json.dumps({'version':VERSION,'count':counter[split],'feature_dim':FEATURE_DIM,'context_dim':CONTEXT_DIM,'seconds':time.time()-start},indent=2))
    print('Feature cache complete',counter,flush=True)

if __name__=='__main__':main()
