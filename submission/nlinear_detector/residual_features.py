"""V2: residual-distribution, dependence and persistent evidence features."""
import numpy as np
from scipy.signal import lfilter
from numpy.lib.stride_tricks import sliding_window_view
from .features_base import fit_nlinear,context,acf,signed_log

VERSION='nlinear-stream-v2'
BASE_CHANNELS=12
TOTAL_CHANNELS=33
WINDOWS=(16,64,256)
LOOKBACKS=(16,64,128)
CONTEXT_DIM=43
FEATURE_DIM=TOTAL_CHANNELS*5+3
ODD_CHANNELS=(0,3,4,12,14,15,19,21,22,26,28,29)

class FeatureStream:
    def __init__(self,historical):
        raw=np.asarray(historical,dtype=np.float64)
        self.mu=float(raw.mean());self.sd=max(float(raw.std()),1e-8)
        h=(raw-self.mu)/self.sd;cut=int(len(h)*.7)
        self.predictors=[fit_nlinear(h[:cut],l,ridge=.1) for l in LOOKBACKS]
        self.history_sorted=np.sort(h)
        self.nl_scales=[];self.residual_sorted=[];extra=[]
        for l,(coef,bias) in zip(LOOKBACKS,self.predictors):
            v=sliding_window_view(h,l+1);err=h[l:]-(v[:,:-1]@coef+bias)
            r=err[cut-l:];mu=float(r.mean());sd=max(float(r.std()),.05)
            self.nl_scales.append((mu,sd));self.residual_sorted.append(np.sort((r-mu)/sd))
            rr=(r-mu)/sd
            extra.extend([np.log(sd),np.log1p(np.sum(coef*coef)),acf(rr),acf(rr*rr),
                          acf(np.abs(rr)),np.log1p(np.mean(np.clip(rr,-8,8)**4)),np.mean(np.abs(rr)>2)])
        self.ctx=np.clip(np.r_[context(h),extra],-20,20).astype(np.float32)
        assert len(self.ctx)==CONTEXT_DIM
        phi=self._all_channels(h)
        self.ref_mean=np.zeros(TOTAL_CHANNELS);self.ref_sd=np.ones(TOTAL_CHANNELS)
        self.ewm_scale=np.zeros((len(WINDOWS),TOTAL_CHANNELS))
        for j in range(TOTAL_CHANNELS):
            # Full history for raw statistics; held-out reference for fitted residuals.
            reference=phi[128 if j<BASE_CHANNELS else cut:,j]
            self.ref_mean[j]=reference.mean();self.ref_sd[j]=max(reference.std(),.02)
            c=(reference-self.ref_mean[j])/self.ref_sd[j]
            for i,w in enumerate(WINDOWS):
                alpha=2/(w+1);e=lfilter([alpha],[1,-(1-alpha)],c)
                self.ewm_scale[i,j]=max(float(np.std(e[min(w//2,len(e)//3):])),.5/np.sqrt(w))
        self.state=np.zeros((3,TOTAL_CHANNELS));self.peak=np.zeros(TOTAL_CHANNELS)
        self.buffer=list(h[-129:]);self.t=0

    def _all_channels(self,h):
        n=len(h);prev=np.r_[h[0],h[:-1]]
        lag4=np.r_[np.repeat(h[0],4),h[:-4]];lag16=np.r_[np.repeat(h[0],16),h[:-16]]
        rank=np.searchsorted(self.history_sorted,h,side='right')/len(self.history_sorted)
        cols=[np.clip(h,-8,8),np.minimum(h*h,64),np.minimum(np.abs(h),8),np.sign(h),2*rank-1,
              np.sign(h)*np.sign(prev),np.clip(h*prev,-64,64),np.clip(h*lag4,-64,64),
              np.clip(h*lag16,-64,64),np.minimum(np.abs(h*prev),64),(np.abs(h)>2).astype(float),
              (np.abs(h)>4).astype(float)]
        for l,(coef,bias),(mu,sd),ref in zip(LOOKBACKS,self.predictors,self.nl_scales,self.residual_sorted):
            win=sliding_window_view(h,l+1);r=np.zeros(n)
            r[l:]=(h[l:]-(win[:,:-1]@coef+bias)-mu)/sd
            u=2*(np.searchsorted(ref,r,side='right')+.5)/(len(ref)+1)-1
            prod=r*np.r_[r[0],r[:-1]]
            cols.extend([np.clip(r,-8,8),np.minimum(r*r,64),u,(5*u**3-3*u)/2,
                         (35*u**4-30*u*u+3)/8,np.clip(prod,-64,64),np.minimum(np.abs(prod),64)])
        return np.stack(cols,axis=1)

    def _point_channels(self,z):
        prev=self.buffer[-1];lag4=self.buffer[-4];lag16=self.buffer[-16]
        rank=np.searchsorted(self.history_sorted,z,side='right')/len(self.history_sorted)
        cols=[np.clip(z,-8,8),min(z*z,64),min(abs(z),8),np.sign(z),2*rank-1,
              np.sign(z)*np.sign(prev),np.clip(z*prev,-64,64),np.clip(z*lag4,-64,64),
              np.clip(z*lag16,-64,64),min(abs(z*prev),64),float(abs(z)>2),float(abs(z)>4)]
        for l,(coef,bias),(mu,sd),ref in zip(LOOKBACKS,self.predictors,self.nl_scales,self.residual_sorted):
            r=(z-(np.dot(coef,self.buffer[-l:])+bias)-mu)/sd
            rp=(prev-(np.dot(coef,self.buffer[-l-1:-1])+bias)-mu)/sd
            u=2*(np.searchsorted(ref,r,side='right')+.5)/(len(ref)+1)-1
            prod=r*rp
            cols.extend([np.clip(r,-8,8),min(r*r,64),u,(5*u**3-3*u)/2,
                         (35*u**4-30*u*u+3)/8,np.clip(prod,-64,64),min(abs(prod),64)])
        return np.asarray(cols)

    def consume(self,point):
        z=(float(point)-self.mu)/self.sd
        c=(self._point_channels(z)-self.ref_mean)/self.ref_sd
        for j,w in enumerate(WINDOWS):
            a=2/(w+1);self.state[j]=(1-a)*self.state[j]+a*c
        normalized=self.state/self.ewm_scale
        self.peak=np.maximum(self.peak,np.max(np.abs(normalized),axis=0))
        blocks=np.r_[signed_log(c),signed_log(normalized.ravel()),np.log1p(self.peak)]
        times=np.asarray([np.log1p(self.t)/np.log(1001),self.t/1000,1/np.sqrt(self.t+1)])
        self.buffer.append(z);self.buffer.pop(0);self.t+=1
        return np.r_[np.clip(blocks,-8,8),times].astype(np.float32)

    def batch(self,online):
        z=(np.asarray(online,dtype=np.float64)-self.mu)/self.sd
        phi=self._all_channels(np.r_[self.buffer,z])[129:]
        c=(phi-self.ref_mean)/self.ref_sd
        blocks=[signed_log(c)];states=[]
        for j,w in enumerate(WINDOWS):
            a=2/(w+1);ev=lfilter([a],[1,-(1-a)],c,axis=0)/self.ewm_scale[j]
            states.append(ev);blocks.append(signed_log(ev))
        peaks=np.maximum.accumulate(np.max(np.abs(np.stack(states)),axis=0),axis=0)
        blocks.append(np.log1p(peaks));t=np.arange(len(z),dtype=np.float64)
        return np.c_[np.clip(np.concatenate(blocks,axis=1),-8,8),
                       np.log1p(t)/np.log(1001),t/1000,1/np.sqrt(t+1)].astype(np.float32)

def feature_indices(kind):
    if kind=='gru':channels=list(range(BASE_CHANNELS))
    elif kind in ('nlinear','nlinear_gru'):channels=list(range(BASE_CHANNELS,TOTAL_CHANNELS))
    elif kind in ('hybrid','mlp'):channels=list(range(TOTAL_CHANNELS))
    else:raise ValueError(kind)
    return [b*TOTAL_CHANNELS+c for b in range(5) for c in channels]+list(range(FEATURE_DIM-3,FEATURE_DIM))
