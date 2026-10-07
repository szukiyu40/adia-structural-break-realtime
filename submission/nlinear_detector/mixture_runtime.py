"""Small causal mixture over change sizes, times, and NLinear statistics."""
import copy
import numpy as np
from scipy.special import logsumexp

COLUMNS=np.array([0,1,2,4,6,7,8,9,12,13,19,20,26,27])
MAGNITUDES=np.array([.025,.05,.1,.2,.4,.8])

class MixtureCorrection:
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
        self.logsum=np.full((14,12),-np.inf);self.t=0

    def source(self,phi):
        h=self.host
        return np.clip((phi[...,COLUMNS]-h.ref_mean[COLUMNS])/h.ref_sd[COLUMNS],-8,8)-self.center

    def consume(self,point):
        h=self.host;z=(float(point)-h.mu)/h.sd;c=self.source(h._point_channels(z))
        self.logsum=np.logaddexp(0,self.logsum)+c[:,None]*self.theta-self.penalty;self.t+=1
        evidence=logsumexp(self.logsum)-np.log(14*12)-np.log(self.t)
        h.buffer.append(z);h.buffer.pop(0)
        return float(np.clip(evidence,-4,4))

    def batch(self,online):
        h=self.host;z=(np.asarray(online,dtype=float)-h.mu)/h.sd
        c=self.source(h._all_channels(np.r_[h.buffer,z])[129:]);llr=c[:,:,None]*self.theta-self.penalty
        cumulative=np.cumsum(llr,axis=0);previous=np.concatenate([np.zeros_like(cumulative[:1]),cumulative[:-1]])
        logsum=cumulative+np.logaddexp.accumulate(-previous,axis=0)
        evidence=logsumexp(logsum,axis=(1,2))-np.log(14*12)-np.log(np.arange(1,len(z)+1))
        return np.clip(evidence,-4,4)
