"""Cache only the mixture supplement; reuse the V8 evidence maps."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,time,multiprocessing
from pathlib import Path
import numpy as np
from common import series_stream
from features import MixtureBank,ResidualStream,EXTRA_DIM,VERSION

def compute(payload):
    sid,h,x,check=payload;s=MixtureBank(ResidualStream(h),h);f=s.batch(x)
    if check:
        expected=np.stack([s.consume(v) for v in x]);delta=float(np.max(np.abs(expected-f)))
        assert delta<2e-6,(sid,delta)
        prefix=MixtureBank(ResidualStream(h),h).batch(x[:100]);assert np.array_equal(prefix,f[:100]),sid
    assert np.isfinite(f).all()
    return sid,f

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-cache',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--splits',default='train,tune,holdout');ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();positions={};stores={}
    for split in args.splits.split(','):
        base=args.base_cache/split;target=args.cache/split;target.mkdir(parents=True,exist_ok=True)
        for key in ['features','context','labels','lengths','ids']:
            dest=target/f'{key}.npy'
            if not dest.exists():dest.symlink_to((base/f'{key}.npy').resolve())
        ids=np.load(base/'ids.npy')
        for pos,sid in enumerate(ids):positions[int(sid)]=(split,pos)
        stores[split]=np.lib.format.open_memmap(target/'extra_features.npy',mode='w+',dtype='float32',shape=(len(ids),1000,EXTRA_DIM))
    def payloads():
        count=0
        for a in series_stream(args.data/'X_train.parquet'):
            sid=int(a['id'][0])
            if sid not in positions:continue
            yield sid,a['value'][a['period']==1],a['value'][a['period']==2],count<3
            count+=1
    start=time.time();count=0
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for sid,f in pool.imap_unordered(compute,payloads(),chunksize=4):
            split,pos=positions[sid];stores[split][pos,:len(f)]=f;count+=1
            if count%1000==0:print(json.dumps({'processed':count,'seconds':round(time.time()-start,1)}),flush=True)
    for split,a in stores.items():
        a.flush();(args.cache/split/'metadata.json').write_text(json.dumps({'version':VERSION,'extra_dim':EXTRA_DIM,'count':len(a),'base_cache':str(args.base_cache.resolve())},indent=2))
    print('MIXTURE_CACHE_COMPLETE',count,flush=True)

if __name__=='__main__':main()
