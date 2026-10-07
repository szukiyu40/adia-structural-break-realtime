"""Causal reference-calibrated distribution evidence and AR-only control."""
import numpy as np
from .constants import QUANTILES

WINDOWS=[8,32,128,256,0]

def channels(values,residual,q):
    r=np.clip(residual,-15,15); v=np.clip(values,-15,15)
    control=np.column_stack([r,r*r,abs(r),(abs(r)>2).astype(float),
        r*np.r_[0,r[:-1]],v,v*v])
    # Piecewise linear probability integral transform between ordered quantiles.
    bin=(residual[:,None]>=q).sum(axis=1)
    low=np.maximum(bin-1,0); high=np.minimum(bin,8); row=np.arange(len(r))
    qlo=q[row,low]; qhi=q[row,high]
    u=QUANTILES[low]+(QUANTILES[high]-QUANTILES[low])*np.clip((residual-qlo)/np.maximum(qhi-qlo,1e-5),0,1)
    u=np.where(bin==0,.005,np.where(bin==9,.995,u))
    width=np.maximum(q[:,6]-q[:,2],.1)
    z=np.clip((residual-q[:,4])*2.563103/width,-15,15)
    error=residual[:,None]-q
    pinball=np.mean(np.maximum(QUANTILES*error,(QUANTILES-1)*error),axis=1)
    learned=np.column_stack([u-.5,(u-.5)**2,(u<.1).astype(float),(u>.9).astype(float),
        (abs(u-.5)>.45).astype(float),z,z*z,abs(z),np.log1p(np.maximum(pinball,0)),
        np.clip(q[:,4],-15,15),np.log(width),z*np.r_[0,z[:-1]]])
    return control.astype('f4'),learned.astype('f4')

def aggregate(c,cal,ticks):
    # Fixed normal reference. Online windows contain only points <= current tick.
    hist=c[1:cal].astype('f8'); online=c[cal:].astype('f8')
    mu=hist.mean(axis=0); sd=np.maximum(hist.std(axis=0),.05)
    standardized=np.clip((online-mu)/sd,-30,30)
    cumulative=np.concatenate([np.zeros((1,c.shape[1])),np.cumsum(standardized,axis=0)])
    end=ticks+1; result=[]
    for w in WINDOWS:
        begin=np.maximum(0,end-w) if w else np.zeros_like(end)
        n=end-begin
        delta=(cumulative[end]-cumulative[begin])/n[:,None]
        result.append(delta)
        result.append(delta*np.sqrt(n[:,None]/(1+n[:,None]/len(hist))))
    # The window=0 branch above accumulates prefix evidence. Keep reference scale too.
    result.extend([np.tile(np.log1p(sd),(len(ticks),1)),np.tile(np.clip(mu,-20,20),(len(ticks),1))])
    return np.concatenate(result,axis=1).astype('f4')

def feature_matrix(v,r,q,cal,ticks,static):
    ctrl,learned=channels(v,r,q)
    a=aggregate(ctrl,cal,ticks); b=aggregate(learned,cal,ticks)
    shared=np.concatenate([np.tile(static,(len(ticks),1)),np.log1p(ticks)[:,None]/7],axis=1).astype('f4')
    return np.concatenate([a,shared,b],axis=1),a.shape[1]+shared.shape[1]
