"""Fixed CPU inference for GPU-trained causal temporal residuals."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from scipy.special import logit,expit
import adaptive_variance_runtime as parent
from adaptive_features import compute
from gpu_temporal_network import TemporalResidual

class TemporalBank:
    def __init__(self,folder):
        root=Path(folder);self.meta=json.loads((root/'metadata.json').read_text())
        assert self.meta['features']==130 and self.meta['stride']==4 and self.meta['buffer_checkpoints']==43
        torch.set_num_threads(1);self.models=[]
        for info in self.meta['model_files']:
            path=root/info['file']
            with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256']
            with np.load(path) as z:
                model=TemporalResidual(130,32)
                state={k:torch.from_numpy(z[k].copy()) for k in model.state_dict()}
                model.load_state_dict(state,strict=True);model.eval()
                self.models.append((model,z['mean'].copy(),z['std'].copy()))
    def predict(self,rows):
        values=[]
        with torch.inference_mode():
            for model,mean,std in self.models:
                x=np.clip((rows-mean)/std,-8,8).astype(np.float32)
                values.append(float(model(torch.from_numpy(x[None]))[0,-1]))
        return self.meta['alpha']*float(np.clip(np.mean(values),-self.meta['clip'],self.meta['clip']))

class Resources(parent.Resources):
    def __init__(self,folder):
        super().__init__(folder)
        self.temporal=TemporalBank(Path(folder).parent/'gpu_temporal')

class State(parent.State):
    def __init__(self,history,resources):
        super().__init__(history,resources)
        self.temporal_rows=[];self.temporal_correction=0.
    def apply(self,previous,core):
        result=super().apply(previous,core)
        if self.tick%4==0:
            extra=compute(self.history,np.asarray(self.online))[-1].copy()
            for k in range(0,72,12):extra[k:k+6]=abs(extra[k:k+6])
            row=np.r_[logit(np.clip(core,1e-7,1-1e-7)),self.last_features,extra].astype(np.float32)
            self.temporal_rows.append(row)
            if len(self.temporal_rows)>43:self.temporal_rows.pop(0)
            self.temporal_correction=self.resources.temporal.predict(np.stack(self.temporal_rows))
        return float(expit(logit(np.clip(result,1e-7,1-1e-7))+self.temporal_correction))
