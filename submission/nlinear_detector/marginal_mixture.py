"""Mixture likelihood evidence integrating all possible past change times."""
import copy
import numpy as np
from scipy.special import logsumexp
from .v8_features import FeatureStream as BaseStream,feature_indices as base_indices,AUGMENT_ODD_INDICES,CONTEXT_DIM
from .residual_features import FeatureStream as ResidualStream
from .features_base import signed_log

VERSION='nlinear-mixture-evidence-v13'
COLUMNS=np.array([0,1,2,4,6,7,8,9,12,13,19,20,26,27])
MAGNITUDES=np.array([.025,.05,.1,.2,.4,.8])
GROUPS=([0,1,6,7],[2,3,8,9],[4,5,10,11],list(range(12)))
EXTRA_DIM=112
FEATURE_DIM=397+EXTRA_DIM

class MixtureBank:
    def __init__(self,host,history):
        self.host=copy.copy(host);self.host.buffer=list(host.buffer)
        z=(np.asarray(history,dtype=float)-host.mu)/host.sd;phi=host._all_channels(z);cut=int(.7*len(z))
        self.center=np.zeros(14);self.theta=np.zeros((14,12));self.penalty=np.zeros((14,12))
        for j,col in enumerate(COLUMNS):
            c=(phi[128 if col<12 else cut:,col]-host.ref_mean[col])/host.ref_sd[col]
            c=np.clip(c,-8,8);self.center[j]=c.mean();c=c-self.center[j]
            hac=np.clip((host.ewm_scale[2,col]*16)**2,.5,64)
            self.theta[j]=np.r_[MAGNITUDES,-MAGNITUDES]/hac
            self.penalty[j]=hac*(logsumexp(c[:,None]*self.theta[j],axis=0)-np.log(len(c)))
        self.logsum=np.full((14,12),-np.inf);self.peak=np.zeros((14,4));self.t=0

    def source(self,phi):
        h=self.host
        return np.clip((phi[...,COLUMNS]-h.ref_mean[COLUMNS])/h.ref_sd[COLUMNS],-8,8)-self.center

    def consume(self,point):
        h=self.host;z=(float(point)-h.mu)/h.sd;c=self.source(h._point_channels(z))
        self.logsum=np.logaddexp(0,self.logsum)+c[:,None]*self.theta-self.penalty;self.t+=1
        evidence=np.stack([logsumexp(self.logsum[:,g],axis=1)-np.log(len(g))-np.log(self.t) for g in GROUPS],axis=-1)
        self.peak=np.maximum(self.peak,evidence);h.buffer.append(z);h.buffer.pop(0)
        return np.clip(np.r_[signed_log(evidence.ravel()),np.log1p(self.peak.ravel())],-8,8).astype(np.float32)

    def batch(self,online):
        h=self.host;z=(np.asarray(online,dtype=float)-h.mu)/h.sd
        c=self.source(h._all_channels(np.r_[h.buffer,z])[129:])
        llr=c[:,:,None]*self.theta-self.penalty;cumulative=np.cumsum(llr,axis=0)
        previous=np.concatenate([np.zeros_like(cumulative[:1]),cumulative[:-1]],axis=0)
        logsum=cumulative+np.logaddexp.accumulate(-previous,axis=0)
        evidence=np.stack([logsumexp(logsum[:,:,g],axis=2)-np.log(len(g))-np.log(np.arange(1,len(z)+1))[:,None] for g in GROUPS],axis=-1)
        peak=np.maximum.accumulate(np.maximum(evidence,0),axis=0)
        return np.clip(np.c_[signed_log(evidence.reshape(len(z),-1)),np.log1p(peak.reshape(len(z),-1))],-8,8).astype(np.float32)

class FeatureStream(BaseStream):
    def __init__(self,history):
        super().__init__(history);self.mixture=MixtureBank(self,history)
    def consume(self,point):return np.r_[super().consume(point),self.mixture.consume(point)].astype(np.float32)
    def batch(self,online):return np.c_[super().batch(online),self.mixture.batch(online)].astype(np.float32)

def feature_indices(kind):return base_indices(kind)+list(range(397,FEATURE_DIM))
