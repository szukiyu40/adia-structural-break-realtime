"""Causal feature stream with history-fitted NLinear predictors.

The NLinear predictors follow last-value subtraction -> linear map -> add back.
Weights are fitted by ridge regression on the first 70% of the supplied normal
history. The remaining history calibrates residuals. No online horizon is used.
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

VERSION='nlinear-stream-v1'
BASE_CHANNELS=12
TOTAL_CHANNELS=16
WINDOWS=(16,64,256)
CONTEXT_DIM=22
FEATURE_DIM=TOTAL_CHANNELS*4+3
ODD_CHANNELS=(0,3,4,12,14)


def acf(x,lag=1):
    a=x[:-lag]-np.mean(x[:-lag]);b=x[lag:]-np.mean(x[lag:])
    return float(np.dot(a,b)/max(np.sqrt(np.dot(a,a)*np.dot(b,b)),1e-12))


def signed_log(x):
    return np.sign(x)*np.log1p(np.abs(x))


def fit_nlinear(h,lookback,ridge=.1):
    view=sliding_window_view(h,lookback+1)
    last=view[:,-2]
    design=view[:,:-2]-last[:,None]
    target=view[:,-1]-last
    xm=design.mean(0);ym=target.mean()
    xc=design-xm;yc=target-ym
    prior=np.full(lookback-1,1/lookback)
    gram=xc.T@xc
    penalty=ridge*len(xc)
    coef=prior+np.linalg.solve(gram+penalty*np.eye(lookback-1),xc.T@(yc-xc@prior))
    bias=ym-xm@coef
    # Equivalent FIR coefficients, preserving the NLinear sum-to-one constraint.
    return np.r_[coef,1-coef.sum()],float(bias)


def context(h):
    q=np.quantile(h,[.01,.1,.25,.5,.75,.9,.99]);h2=h*h
    four=np.mean(np.abs(h)>4);k=np.mean(h**4)
    dx=np.diff(h)
    trend=np.dot(np.linspace(-.5,.5,len(h)),h)/(len(h)/12)
    c=[np.log(len(h)/1000),np.log(max(q[4]-q[2],1e-5)),q[3],
       np.log1p(k),np.log1p(100*four),np.log(max(np.std(dx),1e-5)),
       *[acf(h,l) for l in (1,2,4,8,16,32)],
       *[acf(h2,l) for l in (1,4,16)],acf(np.abs(h)),
       float(trend),float(q[6]-q[0]),float(q[5]-q[1]),
       float(np.mean(np.abs(h))),float(np.mean(np.sign(h))),
       float(np.mean(np.clip(h,-5,5)**3))]
    assert len(c)==CONTEXT_DIM
    return np.clip(np.asarray(c,dtype=np.float32),-20,20)


class FeatureStream:
    def __init__(self,historical):
        raw=np.asarray(historical,dtype=np.float64)
        self.mu=float(raw.mean());self.sd=max(float(raw.std()),1e-8)
        h=(raw-self.mu)/self.sd
        cut=int(len(h)*.7)
        self.predictors=[fit_nlinear(h[:cut],l) for l in (16,64)]
        self.nl_scales=[]
        for lookback,(coef,bias) in zip((16,64),self.predictors):
            win=sliding_window_view(h,lookback+1)
            err=h[lookback:]-(win[:,:-1]@coef+bias)
            rr=err[max(0,cut-lookback):]
            self.nl_scales.append((float(rr.mean()),max(float(rr.std()),.05)))
        self.history_sorted=np.sort(h)
        self.ctx=context(h)
        phi=self._all_channels(h)
        reference=phi[cut:]
        self.ref_mean=reference.mean(0)
        self.ref_sd=np.maximum(reference.std(0),.02)
        self.ewm_scale=[]
        for w in WINDOWS:
            alpha=2/(w+1)
            ev=lfilter([alpha],[1,-(1-alpha)],(reference-self.ref_mean)/self.ref_sd,axis=0)
            scale=np.std(ev[min(w//2,len(ev)//3):],axis=0)
            self.ewm_scale.append(np.maximum(scale,.5/np.sqrt(w)))
        self.ewm_scale=np.stack(self.ewm_scale)
        self.state=np.zeros((3,TOTAL_CHANNELS),dtype=np.float64)
        self.buffer=list(h[-64:]);self.t=0

    def _all_channels(self,h):
        n=len(h);prev=np.r_[h[0],h[:-1]]
        lag4=np.r_[np.repeat(h[0],4),h[:-4]]
        lag16=np.r_[np.repeat(h[0],16),h[:-16]]
        rank=np.searchsorted(self.history_sorted,h,side='right')/len(self.history_sorted)
        cols=[np.clip(h,-8,8),np.minimum(h*h,64),np.minimum(np.abs(h),8),np.sign(h),2*rank-1,
              np.sign(h)*np.sign(prev),np.clip(h*prev,-64,64),np.clip(h*lag4,-64,64),
              np.clip(h*lag16,-64,64),np.minimum(np.abs(h*prev),64),(np.abs(h)>2).astype(float),
              (np.abs(h)>4).astype(float)]
        for l,(coef,bias),(mu,sd) in zip((16,64),self.predictors,self.nl_scales):
            win=sliding_window_view(h,l+1)
            residual=np.zeros(n)
            residual[l:]=(h[l:]-(win[:,:-1]@coef+bias)-mu)/sd
            cols.extend([np.clip(residual,-8,8),np.minimum(residual**2,64)])
        return np.stack(cols,axis=1)

    def _point_channels(self,z):
        prev=self.buffer[-1];lag4=self.buffer[-4];lag16=self.buffer[-16]
        rank=np.searchsorted(self.history_sorted,z,side='right')/len(self.history_sorted)
        cols=[np.clip(z,-8,8),min(z*z,64),min(abs(z),8),np.sign(z),2*rank-1,
              np.sign(z)*np.sign(prev),np.clip(z*prev,-64,64),np.clip(z*lag4,-64,64),
              np.clip(z*lag16,-64,64),min(abs(z*prev),64),float(abs(z)>2),float(abs(z)>4)]
        for l,(coef,bias),(mu,sd) in zip((16,64),self.predictors,self.nl_scales):
            r=(z-(np.dot(coef,self.buffer[-l:])+bias)-mu)/sd
            cols.extend([np.clip(r,-8,8),min(r*r,64)])
        return np.asarray(cols)

    def consume(self,point):
        z=(float(point)-self.mu)/self.sd
        phi=self._point_channels(z)
        c=(phi-self.ref_mean)/self.ref_sd
        for j,w in enumerate(WINDOWS):
            alpha=2/(w+1);self.state[j]=(1-alpha)*self.state[j]+alpha*c
        blocks=np.r_[signed_log(c),signed_log((self.state/self.ewm_scale).ravel())]
        times=np.asarray([np.log1p(self.t)/np.log(1001),self.t/1000,1/np.sqrt(self.t+1)])
        self.buffer.append(z);self.buffer.pop(0);self.t+=1
        return np.asarray(np.r_[np.clip(blocks,-8,8),times],dtype=np.float32)

    def batch(self,online):
        """Vectorized causal equivalent used only for offline training/evaluation."""
        x=np.asarray(online,dtype=np.float64)
        z=(x-self.mu)/self.sd
        h=np.r_[np.asarray(self.buffer),z]
        phi=self._all_channels(h)[64:]
        c=(phi-self.ref_mean)/self.ref_sd
        blocks=[signed_log(c)]
        for j,w in enumerate(WINDOWS):
            alpha=2/(w+1)
            ev=lfilter([alpha],[1,-(1-alpha)],c,axis=0)
            blocks.append(signed_log(ev/self.ewm_scale[j]))
        times=np.arange(len(x),dtype=np.float64)
        return np.asarray(np.c_[np.clip(np.concatenate(blocks,axis=1),-8,8),
                     np.log1p(times)/np.log(1001),times/1000,1/np.sqrt(times+1)],dtype=np.float32)


def feature_indices(kind):
    if kind=='gru': channels=list(range(BASE_CHANNELS))
    elif kind in ('nlinear','nlinear_gru'):channels=list(range(BASE_CHANNELS,TOTAL_CHANNELS))
    elif kind=='hybrid':channels=list(range(TOTAL_CHANNELS))
    else:raise ValueError(kind)
    return [b*TOTAL_CHANNELS+c for b in range(4) for c in channels]+list(range(64,67))
