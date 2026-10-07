from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

def series_stream(path):
    tail=None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=262144):
        a={k:batch.column(k).to_numpy() for k in batch.schema.names}
        if tail is not None:a={k:np.concatenate([tail[k],v]) for k,v in a.items()}
        cuts=np.r_[0,np.flatnonzero(np.diff(a['id']))+1,len(a['id'])]
        for lo,hi in zip(cuts[:-2],cuts[1:-1]):yield {k:v[lo:hi] for k,v in a.items()}
        tail={k:v[cuts[-2]:] for k,v in a.items()}
    if tail is not None:yield tail

def ts_auc(y,p):
    a=[];w=[]
    for t in range(y.shape[1]):
        ok=y[:,t]>=0; yy=y[ok,t];pp=p[ok,t]
        n1=int(yy.sum());n0=len(yy)-n1;weight=n1*n0
        w.append(weight)
        if weight:
            r=rankdata(pp).astype(np.float64)
            a.append(float((r[yy==1].sum()-n1*(n1+1)/2)/weight))
        else:a.append(np.nan)
    a=np.asarray(a);w=np.asarray(w,dtype=float)
    return float(np.nansum(a*w)/w.sum()) if w.sum() else .5,a,w

def load_cache(path,split):
    p=Path(path)/split
    return {key:np.load(p/f'{key}.npy',mmap_mode='r') for key in ['features','context','labels','ids','lengths']}
