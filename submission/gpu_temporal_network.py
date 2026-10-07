"""Causal temporal residual models; per-step LayerNorm never mixes time."""
import torch
from torch import nn
import torch.nn.functional as F

class CausalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.dilation=dilation
        self.conv=nn.Conv1d(width,width,3,dilation=dilation)
        self.norm=nn.LayerNorm(width)
    def forward(self,x):
        y=self.conv(F.pad(x,(2*self.dilation,0)))
        y=F.silu(self.norm(y.transpose(1,2))).transpose(1,2)
        return x+y*.5

class TemporalResidual(nn.Module):
    def __init__(self,features,width=32):
        super().__init__();self.features=features;self.width=width
        self.project=nn.Linear(features,width)
        self.blocks=nn.Sequential(*[CausalBlock(width,d) for d in [1,4,16]])
        self.head=nn.Linear(width,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        h=F.silu(self.project(x)).transpose(1,2)
        h=self.blocks(h).transpose(1,2)
        return 2*torch.tanh(self.head(h).squeeze(-1)/2)
