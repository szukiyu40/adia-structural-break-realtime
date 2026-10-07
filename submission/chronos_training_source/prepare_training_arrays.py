"""Rebuild final-fit inputs directly from organizer training parquet files.

Usage: python prepare_training_arrays.py --data /path/to/data --out /path/to/arrays
Then: python train.py --data /path/to/arrays --out /path/to/full_fit --fold -1 --steps 1600
The baseline predictions are not required for the final fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def series(path):
    tail = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=131072):
        row = {name:batch.column(name).to_numpy() for name in ('id','time','period','value')}
        if tail is not None:
            row = {name:np.concatenate([tail[name],values]) for name,values in row.items()}
        cuts = np.r_[0,np.flatnonzero(np.diff(row['id']))+1,len(row['id'])]
        for lo,hi in zip(cuts[:-2],cuts[1:-1]):
            yield {name:values[lo:hi] for name,values in row.items()}
        tail = {name:values[cuts[-2]:] for name,values in row.items()}
    if tail is not None:
        yield tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    bank = np.lib.format.open_memmap(args.out/'causal_bank.npy',mode='w+',dtype='f4',shape=(10000,2024))
    bank[:] = np.nan
    lengths = np.zeros(10000,dtype='i4')
    seen=[]
    for row in series(args.data/'X_train.parquet'):
        sid=int(row['id'][0]);assert 0<=sid<10000
        assert np.all(np.diff(row['time'])>0)
        history=row['value'][row['period']==1].astype('f4')[-1024:]
        online=row['value'][row['period']==2].astype('f4')
        assert len(history)>0 and 0<len(online)<=1000
        assert np.isfinite(history).all() and np.isfinite(online).all()
        bank[sid,1024-len(history):1024]=history
        bank[sid,1024:1024+len(online)]=online
        lengths[sid]=len(online);seen.append(sid)
    assert seen==list(range(10000))
    bank.flush()
    tau=pd.read_parquet(args.data/'y_train_index.parquet').loc[np.arange(10000),'tau_index'].to_numpy('i4')
    assert np.all((tau==-1)|((tau>=0)&(tau<lengths)))
    ticks=np.arange(1000)[None,:]
    labels=np.where(ticks<lengths[:,None],(tau[:,None]>=0)&(ticks>=tau[:,None]),-1).astype('i1')
    target=pd.read_parquet(args.data/'y_train.parquet')
    for sid,frame in target.groupby(level='id',sort=True):
        assert np.array_equal(frame['target'].to_numpy(),labels[sid,:lengths[sid]])
    for name,value in [('lengths',lengths),('labels',labels),('tau',tau)]:
        np.save(args.out/f'{name}.npy',value)
    manifest={}
    for name in ['causal_bank.npy','lengths.npy','labels.npy','tau.npy']:
        path=args.out/name
        with path.open('rb') as f:
            digest=hashlib.file_digest(f,'sha256').hexdigest()
        manifest[name]={'sha256':digest,'bytes':path.stat().st_size}
    (args.out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'series':len(seen),'online_points':int(lengths.sum()),'manifest':manifest}),flush=True)


if __name__=='__main__':
    main()
