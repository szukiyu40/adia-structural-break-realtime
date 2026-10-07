"""Read raw local Parquet and fix preprocessing using historical data only."""
import time
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from common import *

def preprocess(x, h):
    cut=h//2
    mu=float(np.mean(x[:cut])); sd=max(float(np.std(x[:cut])),1e-5)
    z=(x-mu)/sd
    design=np.column_stack([z[ORDER-k-1:cut-k-1] for k in range(ORDER)])
    target=z[ORDER:cut]
    gram=design.T@design
    ridge=.01*np.trace(gram)/ORDER+1e-5
    coef=solve(gram+ridge*np.eye(ORDER),design.T@target,assume_a='pos')
    r=lfilter(np.r_[1.,-coef],[1.],z)
    rs=max(float(np.sqrt(np.mean(r[ORDER:cut]**2))),.02)
    r/=rs
    static=np.r_[coef,np.log(rs),np.log(sd),mu,np.log(h)/9].astype('f4')
    return z.astype('f4'),r.astype('f4'),static

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'data_manifest.json').exists():
        print('Already prepared'); return
    start=time.time()
    arrays={n:np.lib.format.open_memmap(CACHE/f'{n}.npy',mode='w+',dtype='f4',shape=(10000,6000)) for n in ['values','innovations']}
    for a in arrays.values(): a[:]=0
    h=np.zeros(10000,dtype='i4'); lengths=h.copy(); static=np.zeros((10000,16),dtype='f4')
    with threadpool_limits(1):
        for s in stream(DATA/'X_train.parquet'):
            sid=int(s['id'][0]); hist=s['value'][s['period']==1]; online=s['value'][s['period']==2]
            assert np.all(np.diff(s['time'])>0)
            x=np.r_[hist,online].astype('f8'); assert np.isfinite(x).all()
            h[sid]=len(hist); lengths[sid]=len(online)
            assert 1000<=len(hist)<=5000 and 10<=len(online)<=1000
            z,r,static[sid]=preprocess(x,len(hist))
            arrays['values'][sid,:len(x)]=z; arrays['innovations'][sid,:len(x)]=r
    for a in arrays.values(): a.flush()
    tau=pd.read_parquet(DATA/'y_train_index.parquet').loc[np.arange(10000),'tau_index'].to_numpy('i4')
    y=np.where(np.arange(1000)[None,:]<lengths[:,None],
        (tau[:,None]>=0)&(np.arange(1000)[None,:]>=tau[:,None]),-1).astype('i1')
    target=pd.read_parquet(DATA/'y_train.parquet')
    for sid,g in target.groupby(level='id',sort=True):
        assert np.array_equal(g['target'].to_numpy(),y[sid,:lengths[sid]])
    parent=np.array(np.load(OLD/'work/raw_gpu/parent.npy'))
    for f in range(5):
        ids=np.load(OLD/f'outputs/colab_chronos2/fold{f}_held_ids.npy')
        pred=np.load(OLD/f'outputs/colab_chronos2/fold{f}_probability.npy')
        assert np.all(ids%5==f)
        parent[ids]=.875*parent[ids]+.125*pred
    score=metric(y,parent); assert abs(score-.6478611402175464)<1e-10
    for n,a in [('hist_lengths',h),('lengths',lengths),('static',static),('labels',y),('baseline',parent)]:
        np.save(CACHE/f'{n}.npy',a)
    write(OUT/'data_manifest.json',{
        'series':10000,'online_points':int(lengths.sum()),'baseline':score,
        'raw_files':{str(p):sha(p) for p in [DATA/'X_train.parquet',DATA/'y_train.parquet',DATA/'y_train_index.parquet']},
        'arrays':{p.name:sha(p) for p in CACHE.glob('*.npy')},
        'baseline_provenance':str(OLD/'outputs/colab_chronos2/decision_local.json'),
        'baseline_has_legacy_meta_in_sample_elements':True,'development_reused':True,
        'seconds':time.time()-start})
    print({'prepared':True,'baseline':score,'seconds':time.time()-start},flush=True)

if __name__=='__main__': main()
