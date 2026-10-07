"""Load the exact pointwise checkpoint shapes independently of feature width."""
import torch
from torch import nn

class FrozenDetector(nn.Module):
    def __init__(self,bundle):
        super().__init__();state=bundle['state_dict'];config=bundle['config']
        assert config['kind'] in ('mlp','nlinear')
        assert not any(k.startswith('rnn.') for k in state)
        self.kind=config['kind'];self.hidden=config['hidden'];self.rnn=None
        self.register_buffer('indices',state['indices'].clone())
        c0=state['context_encoder.0.weight'];c2=state['context_encoder.2.weight']
        inp=state['input_encoder.0.weight'];h0=state['head.0.weight'];h3=state['head.3.weight']
        self.context_encoder=nn.Sequential(nn.Linear(c0.shape[1],c0.shape[0]),nn.SiLU(),nn.Linear(c2.shape[1],c2.shape[0]),nn.Tanh())
        self.input_encoder=nn.Sequential(nn.Linear(inp.shape[1],inp.shape[0]),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(h0.shape[1],h0.shape[0]),nn.SiLU(),nn.Dropout(config['dropout']),nn.Linear(h3.shape[1],h3.shape[0]))
        self.load_state_dict(state);self.eval()

    def forward(self,features,context,state=None):
        c=self.context_encoder(context[...,:self.context_encoder[0].in_features]);x=features.index_select(-1,self.indices)
        encoded=self.input_encoder(torch.cat([x,c[:,None,:].expand(-1,x.shape[1],-1)],dim=-1))
        return self.head(torch.cat([encoded,encoded],dim=-1)).squeeze(-1),None
