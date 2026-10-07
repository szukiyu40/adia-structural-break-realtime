"""NLinear models of log energy distinguish normal volatility from a break."""
import numpy as np
from .marginal_mixture import FeatureStream as MarginalStream, MixtureBank as MarginalBank
from .v8_features import feature_indices as base_indices, AUGMENT_ODD_INDICES, CONTEXT_DIM
from .residual_features import FeatureStream as ResidualStream

VERSION='nlinear-volatility-innovations-v15'
EXTRA_DIM=112+168+43
FEATURE_DIM=397+EXTRA_DIM

class VolatilityBank:
    def __init__(self,host,history):
        self.mu=host.mu;self.sd=host.sd
        energy=np.log1p(((np.asarray(history,dtype=float)-self.mu)/self.sd)**2)
        self.model=ResidualStream(energy)
    def consume(self,point):
        energy=np.log1p(((float(point)-self.mu)/self.sd)**2)
        return np.r_[self.model.consume(energy),self.model.ctx].astype(np.float32)
    def batch(self,online):
        energy=np.log1p(((np.asarray(online,dtype=float)-self.mu)/self.sd)**2)
        return np.c_[self.model.batch(energy),np.tile(self.model.ctx,(len(energy),1))].astype(np.float32)

class MixtureBank:
    def __init__(self,host,history):
        self.marginal=MarginalBank(host,history);self.volatility=VolatilityBank(host,history)
    def consume(self,point):return np.r_[self.marginal.consume(point),self.volatility.consume(point)]
    def batch(self,online):return np.c_[self.marginal.batch(online),self.volatility.batch(online)]

class FeatureStream(MarginalStream):
    def __init__(self,history):
        super().__init__(history);self.volatility=VolatilityBank(self,history)
    def consume(self,point):return np.r_[super().consume(point),self.volatility.consume(point)].astype(np.float32)
    def batch(self,online):return np.c_[super().batch(online),self.volatility.batch(online)].astype(np.float32)

def feature_indices(kind):return base_indices(kind)+list(range(397,FEATURE_DIM))
