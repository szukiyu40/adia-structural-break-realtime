"""Small causal, history-conditioned innovation quantile predictor."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.special import ndtri
from common import QUANTILES

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(4,16,1)
        self.condition=nn.Linear(16,16)
        self.layers=nn.ModuleList([nn.Conv1d(16,16,3,dilation=d) for d in [1,2,4,8,16]])
        self.output=nn.Conv1d(16,9,1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.register_buffer('normal',torch.from_numpy(ndtri(QUANTILES).astype('f4'))[None,:,None])
        self.register_buffer('levels',torch.from_numpy(QUANTILES)[None,:,None])

    def forward(self,x,static):
        v=F.silu(self.input(x)+self.condition(static.clamp(-5,5))[:,:,None])
        for layer in self.layers:
            v=v+.3*F.silu(layer(F.pad(v,(2*layer.dilation[0],0))))
        q=self.normal+self.output(v)
        return torch.sort(q,dim=1).values

    def loss(self,q,target):
        diff=target[:,None,:]-q
        return torch.maximum(self.levels*diff,(self.levels-1)*diff).mean()

def inputs(values,residual):
    return np.stack([np.clip(residual/3,-8,8),np.sign(residual)*np.log1p(np.minimum(abs(residual),100)),
                     np.clip(values/3,-8,8),np.clip(np.diff(values,axis=-1,prepend=values[...,:1])/3,-8,8)],axis=-2).astype('f4')
