"""Shared causal feature stream and CPU neural ensemble."""
import json
from pathlib import Path
import numpy as np
import torch
from .frozen_network import FrozenDetector
from .features import FeatureStream,AUGMENT_ODD_INDICES

class Ensemble:
    def __init__(self,path):
        torch.set_num_threads(1)
        path=Path(path);self.recipe=json.loads((path/'manifest.json').read_text())
        self.models=[];self.weights=[];self.corrections=[]
        for entry in self.recipe['models']:
            bundle=torch.load(path/entry['file'],map_location='cpu',weights_only=False);c=bundle['config']
            model=FrozenDetector(bundle)
            assert model.rnn is None,'This ensemble runtime accepts pointwise heads only'
            self.models.append(model);self.weights.append(entry['weight'])
            if entry.get('evidence_gru'):
                from .evidence_runtime import Correction
                self.corrections.append(Correction(torch.load(path/entry['evidence_gru'],map_location='cpu',weights_only=False)))
            else:self.corrections.append(None)
        self.weights=np.asarray(self.weights,dtype=float);self.weights/=self.weights.sum()
        self.tta=bool(self.recipe.get('sign_tta',False));self.max_blend=float(self.recipe.get('max_blend',0))
        self.null_scale=float(self.recipe.get('null_scale',0))
        self.mixture_weight=float(self.recipe.get('mixture_weight',0))

    def begin(self,history):
        history=np.asarray(history,dtype=float)
        self.offsets=np.zeros(len(self.models),dtype=float)
        if self.null_scale:
            k=min(512,max(200,len(history)//5))
            if len(history)-k>=256:
                normal=FeatureStream(history[:-k]);f=normal.batch(history[-k:])
                x=np.repeat(f[None],2 if self.tta else 1,axis=0)
                c=np.repeat(normal.ctx[None],2 if self.tta else 1,axis=0)
                if self.tta:x[1,:,AUGMENT_ODD_INDICES]*=-1;c[1,[2,16,20,21]]*=-1
                with torch.inference_mode():
                    for i,model in enumerate(self.models):
                        p=torch.sigmoid(model(torch.from_numpy(x),torch.from_numpy(c))[0]).mean(0).numpy()[64:]
                        p=np.clip(p,1e-7,1-1e-7)
                        self.offsets[i]=self.null_scale*float(np.quantile(np.log(p/(1-p)),.9))
        self.features=FeatureStream(history);context=np.repeat(self.features.ctx[None],2 if self.tta else 1,axis=0)
        if self.tta:context[1,[2,16,20,21]]*=-1
        with torch.inference_mode():self.contexts=[m.context_encoder(torch.from_numpy(context[:,:m.context_encoder[0].in_features])) for m in self.models]
        for correction in self.corrections:
            if correction is not None:correction.begin(self.features.ctx)
        self.peak=0.
        self.mixture=None
        if self.mixture_weight:
            from .mixture_runtime import MixtureCorrection
            self.mixture=MixtureCorrection(self.features,history)

    def consume(self,point):
        features=np.repeat(self.features.consume(point)[None,None,:],2 if self.tta else 1,axis=0)
        if self.tta:features[1,:,AUGMENT_ODD_INDICES]*=-1
        x=torch.from_numpy(features);scores=[]
        with torch.inference_mode():
            for model,context,offset,correction in zip(self.models,self.contexts,self.offsets,self.corrections):
                selected=x.index_select(-1,model.indices)
                encoded=model.input_encoder(torch.cat([selected,context[:,None,:]],dim=-1))
                logits=model.head(torch.cat([encoded,encoded],dim=-1)).squeeze(-1)
                probability=float(torch.sigmoid(logits).mean())
                if self.null_scale or correction is not None:
                    probability=float(np.clip(probability,1e-7,1-1e-7))
                    z=float(np.log(probability/(1-probability)))
                    if correction is not None:z+=correction.consume(z)
                    probability=float(1/(1+np.exp(-(z-offset))))
                scores.append(probability)
        score=float(np.dot(self.weights,scores));self.peak=max(self.peak,score)
        score=(1-self.max_blend)*score+self.max_blend*self.peak
        if self.mixture is not None:
            evidence=self.mixture.consume(point);score=float(np.clip(score,1e-7,1-1e-7))
            score=float(1/(1+np.exp(-(np.log(score/(1-score))+self.mixture_weight*evidence))))
        return score
