"""NLinear evidence plus the existing causal marginal/dependence feature bank.

legacy_features.py is the feature-only portion of the user's submitted V2
source. No pretrained classifier or its training predictions are used.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from residual_features import FeatureStream as ResidualStream
from residual_features import feature_indices as residual_indices
from residual_features import ODD_CHANNELS,TOTAL_CHANNELS,CONTEXT_DIM
from legacy_features import StreamingFeaturesV2,WINDOWS,LAGS
from features_base import signed_log

VERSION='nlinear-legacy-evidence-v8'
FEATURE_DIM=168+229
AUGMENT_ODD_INDICES=[b*33+c for b in range(4) for c in ODD_CHANNELS]+[168+3]+[
    168+13+w*32+c for w in range(6) for c in (0,4,21)]

def rolling_sum(x,start,end):
    sums=np.r_[0,np.cumsum(x,dtype=float)]
    return sums[end]-sums[start]

def pit_statistics(bins,start,end,n):
    one=np.eye(16,dtype=float)[bins]
    cs=np.vstack([np.zeros((1,16)),np.cumsum(one,axis=0)])
    counts=cs[end]-cs[start];expected=n[:,None]/16
    chi=(np.sum((counts-expected)**2/expected,axis=1)-15)/np.sqrt(30)
    ks=np.sqrt(n)*np.max(np.abs(np.cumsum(counts,axis=1)/n[:,None]-np.arange(1,17)/16),axis=1)
    edge=((counts[:,0]+counts[:,-1])/n-2/16)*np.sqrt(n)
    return chi,ks,edge

def legacy_batch(s,online):
    x=np.asarray(online,dtype=float)
    z=np.clip((x-s.mean)/s.sd,-20,20)
    joined=np.r_[list(s.previous_z),z]
    prediction=sliding_window_view(joined,6)[:,:-1][:,::-1]@s.ar_coefficients
    r=np.clip((z-prediction-s.residual_mean)/s.residual_sd,-20,20)
    zb=np.searchsorted(s.z_edges,z,side='right');rb=np.searchsorted(s.residual_edges,r,side='right')
    end=np.arange(1,len(z)+1);features=[np.c_[np.log1p(end),np.sqrt(end),
       np.full(len(z),np.log1p(s.hist_len)),np.full(len(z),s.hist_skew),np.full(len(z),s.hist_kurt),
       np.tile(s.hist_acf,(len(z),1)),np.full(len(z),s.hist_abs_acf1),np.abs(z),z*z,np.abs(r),r*r]]
    families=[]
    for w in WINDOWS:
        start=np.maximum(end-w,0);n=(end-start).astype(float);root=np.sqrt(n)
        mean=rolling_sum(z,start,end)/n;m2=rolling_sum(z*z,start,end)/n
        var=np.maximum(m2-mean*mean,1e-8)
        m3=rolling_sum(z**3,start,end)/n-3*mean*m2+2*mean**3
        m4=rolling_sum(z**4,start,end)/n-4*mean*rolling_sum(z**3,start,end)/n+6*mean*mean*m2-3*mean**4
        mean_stat=mean*root;variance_stat=(m2-1)*root/s.hist_z2_sd
        skew_stat=(m3/np.maximum(var**1.5,1e-8)-s.hist_skew)*np.sqrt(n/6)
        kurt_stat=(m4/np.maximum(var*var,1e-8)-3-s.hist_kurt)*np.sqrt(n/24)
        chi,ks,edge=pit_statistics(zb,start,end,n)
        acfs=[]
        for lag,hacf in zip(LAGS,s.hist_acf):
            product=np.r_[np.zeros(lag),z[lag:]*z[:-lag]]
            cross=rolling_sum(product,np.minimum(start+lag,end),end)/np.maximum(n-lag,1)
            stat=(cross-mean*mean)/var-hacf
            acfs.append(np.where(n>lag+1,stat*root,0))
        am=rolling_sum(np.abs(z),start,end)/n;av=np.maximum(rolling_sum(np.abs(z)**2,start,end)/n-am*am,1e-8)
        across=rolling_sum(np.r_[0,np.abs(z[1:]*z[:-1])],np.minimum(start+1,end),end)/np.maximum(n-1,1)
        aa=np.where(n>2,((across-am*am)/av-s.hist_abs_acf1)*root,0)
        rm=rolling_sum(r,start,end)/n;rm2=rolling_sum(r*r,start,end)/n;rv=np.maximum(rm2-rm*rm,1e-8)
        rmean=rm*root;rvar=(rm2-1)*root;rchi,rks,redge=pit_statistics(rb,start,end,n)
        rcross=rolling_sum(np.r_[0,r[1:]*r[:-1]],np.minimum(start+1,end),end)/np.maximum(n-1,1)
        ra=np.where(n>2,((rcross-rm*rm)/rv-s.hist_residual_acf1)*root,0)
        values=np.c_[mean_stat,abs(mean_stat),variance_stat,abs(variance_stat),skew_stat,abs(skew_stat),
            kurt_stat,abs(kurt_stat),chi,abs(chi),ks,edge,abs(edge),acfs[0],abs(acfs[0]),
            acfs[1],abs(acfs[1]),acfs[2],abs(acfs[2]),aa,abs(aa),rmean,abs(rmean),
            rvar,abs(rvar),rchi,abs(rchi),rks,redge,abs(redge),ra,abs(ra)]
        family=np.c_[abs(mean_stat),abs(variance_stat),np.maximum.reduce([ks,abs(chi),abs(edge)]),
            np.maximum(abs(skew_stat),abs(kurt_stat)),np.max(np.abs(np.stack(acfs)),axis=0),abs(aa),
            np.maximum.reduce([abs(rmean),abs(rvar),rks,abs(rchi)]),abs(ra)]
        values[n<2]=0;family[n<2]=0;features.append(values);families.append(family)
    current=np.max(np.stack(families),axis=0);factor=.995**np.arange(len(z))[:,None]
    decayed=np.maximum.accumulate(current/factor,axis=0)*factor
    features.extend([current,decayed,np.maximum.accumulate(current,axis=0)])
    return np.nan_to_num(np.concatenate(features,axis=1).astype(np.float32),nan=0,posinf=50,neginf=-50)

class FeatureStream(ResidualStream):
    def __init__(self,historical):
        super().__init__(historical);self.legacy=StreamingFeaturesV2(historical)

    def consume(self,point):
        residual=super().consume(point)
        legacy=self.legacy.update(point)
        return np.r_[residual,np.clip(signed_log(legacy),-8,8)].astype(np.float32)

    def batch(self,online):
        residual=super().batch(online);legacy=legacy_batch(self.legacy,online)
        return np.c_[residual,np.clip(signed_log(legacy),-8,8)].astype(np.float32)

def feature_indices(kind):
    return residual_indices(kind)+list(range(168,FEATURE_DIM))
