"""Independent, resettable GRU correction of a pointwise NLinear logit."""
import numpy as np
import torch
from torch import nn
from scipy.signal import lfilter


class EvidenceGRU(nn.Module):
    def __init__(self,hidden=12,bound=.5,context_dim=43):
        super().__init__();self.bound=bound
        self.initial=nn.Sequential(nn.Linear(context_dim,hidden),nn.Tanh())
        self.gru=nn.GRU(14,hidden,batch_first=True)
        self.output=nn.Linear(hidden,1)
    def forward(self,x,c,state=None):
        if state is None:state=self.initial(c).unsqueeze(0)
        h,state=self.gru(x,state)
        return self.bound*torch.tanh(self.output(h).squeeze(-1)),state


class Correction:
    def __init__(self,bundle):
        config=bundle['config'];self.context_dim=bundle['state_dict']['initial.0.weight'].shape[1]
        self.model=EvidenceGRU(config['hidden'],config['bound'],self.context_dim)
        self.model.load_state_dict(bundle['state_dict']);self.model.eval()
    def begin(self,context):
        c=np.asarray(context,dtype=np.float32)[:self.context_dim].copy();c[[2,16,20,21]]=np.abs(c[[2,16,20,21]])
        self.context=torch.from_numpy(np.clip(c*.2,-4,4))[None,:]
        self.state=None;self.t=0;self.previous=None;self.means=np.zeros(3);self.movements=np.zeros(3)
    def consume(self,z):
        value=float(np.clip(z,-6,6));delta=0. if self.previous is None else value-self.previous
        x=[value/3,delta,abs(delta)]
        for i,width in enumerate([8,32,128]):
            a=2/(width+1);self.means[i]=(1-a)*self.means[i]+a*value
            self.movements[i]=(1-a)*self.movements[i]+a*max(delta,0)
            x.extend([self.means[i]/3,(value-self.means[i])/3,self.movements[i]])
        x.extend([np.log1p(self.t)/np.log(1001),1/np.sqrt(self.t+1)])
        with torch.inference_mode():
            correction,self.state=self.model(torch.tensor(x,dtype=torch.float32)[None,None,:],self.context,self.state)
        self.t+=1;self.previous=value
        return float(correction.item())
    def batch(self,z):
        z=np.clip(np.asarray(z,dtype=np.float32),-6,6)
        delta=np.diff(z,prepend=z[:1]);cols=[z/3,delta,np.abs(delta)]
        for width in [8,32,128]:
            a=2/(width+1);m=lfilter([a],[1,-(1-a)],z).astype(np.float32)
            movement=lfilter([a],[1,-(1-a)],np.maximum(delta,0)).astype(np.float32)
            cols.extend([m/3,(z-m)/3,movement])
        t=np.arange(len(z),dtype=np.float32);cols.extend([np.log1p(t)/np.log(1001),1/np.sqrt(t+1)])
        x=np.stack(cols,axis=-1).astype(np.float32)
        with torch.inference_mode():return self.model(torch.from_numpy(x)[None,:],self.context)[0][0].numpy()
