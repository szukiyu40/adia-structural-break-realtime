"""Small history-conditioned pointwise/GRU detectors."""
import torch
from torch import nn
from features import feature_indices,CONTEXT_DIM

class Detector(nn.Module):
    def __init__(self,kind='hybrid',hidden=48,dropout=.1):
        super().__init__();self.kind=kind;self.hidden=hidden
        self.register_buffer('indices',torch.tensor(feature_indices(kind)))
        self.context_encoder=nn.Sequential(nn.Linear(CONTEXT_DIM,48),nn.SiLU(),nn.Linear(48,hidden),nn.Tanh())
        self.input_encoder=nn.Sequential(nn.Linear(len(feature_indices(kind))+hidden,hidden),nn.SiLU())
        self.rnn=nn.GRU(hidden,hidden,batch_first=True) if kind not in ('nlinear','mlp') else None
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,features,context,state=None):
        c=self.context_encoder(context)
        x=features.index_select(-1,self.indices)
        encoded=self.input_encoder(torch.cat([x,c[:,None,:].expand(-1,x.shape[1],-1)],dim=-1))
        if self.rnn is not None:
            if state is None:state=c.unsqueeze(0)
            sequence,state=self.rnn(encoded,state)
        else:sequence=encoded
        logits=self.head(torch.cat([sequence,encoded],dim=-1)).squeeze(-1)
        return logits,state
