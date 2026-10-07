"""V204 plus matched heads measuring the added adaptive-variance features."""
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
import conditional_bayes_runtime_v199 as base
from adaptive_features import compute

class Resources(base.Resources):
    def __init__(self,folder):
        super().__init__(folder)
        self.extra=base.Resources(Path(folder).parent/'adaptive_variance')

class State(base.State):
    def __init__(self,history,resources):
        super().__init__(history,resources)
        self.extra_correction=0.
    def apply(self,parent,core):
        previous=super().apply(parent,core)
        if self.tick%4==0:
            extra=self.resources.extra;m=extra.meta
            row=np.r_[logit(np.clip(core,1e-7,1-1e-7)),self.last_features[m['context_columns']+m['dynamic_columns']]][None,:]
            augmented=np.c_[row,compute(self.history,np.asarray(self.online))[-1:]]
            a=float(extra.models['augmented'].predict(augmented,raw_score=True,num_threads=1)[0])
            b=float(extra.models['control'].predict(row,raw_score=True,num_threads=1)[0])
            self.extra_correction=.5*float(np.clip(a-b,-1,1))
        return float(expit(logit(np.clip(previous,1e-7,1-1e-7))+self.extra_correction))
