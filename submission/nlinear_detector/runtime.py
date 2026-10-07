"""NumPy implementation of frozen NLinear MLPs and bounded GRU score heads."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from .features import FeatureStream,AUGMENT_ODD_INDICES


def silu(x):return x*expit(x)


class PointModel:
    def __init__(self,bundle):
        self.weights={k:v.detach().cpu().numpy() for k,v in bundle['state_dict'].items()}
        assert not any(k.startswith('rnn.') for k in self.weights)
        self.indices=self.weights['indices'];self.context_dim=self.weights['context_encoder.0.weight'].shape[1]
    def linear(self,x,key):return x@self.weights[key+'.weight'].T+self.weights[key+'.bias']
    def context(self,c):return np.tanh(self.linear(silu(self.linear(c[...,:self.context_dim],'context_encoder.0')),'context_encoder.2'))
    def point(self,x,c):
        z=silu(self.linear(np.concatenate([x[...,self.indices],c],axis=-1),'input_encoder.0'))
        return self.linear(silu(self.linear(np.concatenate([z,z],axis=-1),'head.0')),'head.3')[...,0]
    def batch(self,x,c):
        context=self.context(c)
        return self.point(x,np.broadcast_to(context[:,None,:],(*x.shape[:2],context.shape[-1])))


class NumpyCorrection:
    def __init__(self,bundle):
        self.weights={k:v.detach().cpu().numpy() for k,v in bundle['state_dict'].items()}
        self.bound=bundle['config']['bound'];self.hidden=bundle['config']['hidden']
        self.context_dim=self.weights['initial.0.weight'].shape[1]
    def begin(self,context):
        c=np.asarray(context,dtype=np.float32)[:self.context_dim].copy();c[[2,16,20,21]]=np.abs(c[[2,16,20,21]])
        c=np.clip(c*.2,-4,4)
        self.state=np.tanh(self.weights['initial.0.weight']@c+self.weights['initial.0.bias'])
        self.t=0;self.previous=None;self.means=np.zeros(3);self.movements=np.zeros(3)
    def consume(self,z):
        value=float(np.clip(z,-6,6));delta=0. if self.previous is None else value-self.previous
        x=[value/3,delta,abs(delta)]
        for i,width in enumerate([8,32,128]):
            a=2/(width+1);self.means[i]=(1-a)*self.means[i]+a*value
            self.movements[i]=(1-a)*self.movements[i]+a*max(delta,0)
            x.extend([self.means[i]/3,(value-self.means[i])/3,self.movements[i]])
        x.extend([np.log1p(self.t)/np.log(1001),1/np.sqrt(self.t+1)])
        w=self.weights;gx=w['gru.weight_ih_l0']@np.asarray(x,dtype=np.float32)+w['gru.bias_ih_l0'];gh=w['gru.weight_hh_l0']@self.state+w['gru.bias_hh_l0']
        xr,xz,xn=np.split(gx,3);hr,hz,hn=np.split(gh,3);reset=expit(xr+hr);update=expit(xz+hz);candidate=np.tanh(xn+reset*hn)
        self.state=(1-update)*candidate+update*self.state
        result=self.bound*float(np.tanh(w['output.weight']@self.state+w['output.bias']).item())
        self.t+=1;self.previous=value
        return result


class Ensemble:
    def __init__(self,path):
        import torch
        torch.set_num_threads(1)
        path=Path(path);self.recipe=json.loads((path/'manifest.json').read_text());self.models=[];self.weights=[];self.corrections=[]
        for entry in self.recipe['models']:
            self.models.append(PointModel(torch.load(path/entry['file'],map_location='cpu',weights_only=False)));self.weights.append(entry['weight'])
            self.corrections.append(NumpyCorrection(torch.load(path/entry['evidence_gru'],map_location='cpu',weights_only=False)) if entry.get('evidence_gru') else None)
        self.weights=np.asarray(self.weights,dtype=float);self.weights/=self.weights.sum()
        self.tta=bool(self.recipe.get('sign_tta',False));self.max_blend=float(self.recipe.get('max_blend',0));self.null_scale=float(self.recipe.get('null_scale',0));self.mixture_weight=float(self.recipe.get('mixture_weight',0))
        assert self.mixture_weight==0
    def begin(self,history):
        history=np.asarray(history,dtype=float);self.offsets=np.zeros(len(self.models),dtype=float)
        if self.null_scale:
            k=min(512,max(200,len(history)//5))
            if len(history)-k>=256:
                normal=FeatureStream(history[:-k]);f=normal.batch(history[-k:]);x=np.repeat(f[None],2 if self.tta else 1,axis=0);c=np.repeat(normal.ctx[None],len(x),axis=0)
                if self.tta:x[1,:,AUGMENT_ODD_INDICES]*=-1;c[1,[2,16,20,21]]*=-1
                for i,model in enumerate(self.models):
                    p=expit(model.batch(x,c)).mean(0)[64:];p=np.clip(p,1e-7,1-1e-7)
                    self.offsets[i]=self.null_scale*float(np.quantile(np.log(p/(1-p)),.9))
        self.features=FeatureStream(history);context=np.repeat(self.features.ctx[None],2 if self.tta else 1,axis=0)
        if self.tta:context[1,[2,16,20,21]]*=-1
        self.contexts=[m.context(context) for m in self.models]
        for correction in self.corrections:
            if correction is not None:correction.begin(self.features.ctx)
        self.peak=0.;self.mixture=None
    def consume(self,point):
        features=np.repeat(self.features.consume(point)[None,:],2 if self.tta else 1,axis=0)
        if self.tta:features[1,AUGMENT_ODD_INDICES]*=-1
        scores=[]
        for model,c,offset,correction in zip(self.models,self.contexts,self.offsets,self.corrections):
            p=float(expit(model.point(features,c)).mean())
            if self.null_scale or correction is not None:
                p=float(np.clip(p,1e-7,1-1e-7));z=float(np.log(p/(1-p)))
                if correction is not None:z+=correction.consume(z)
                p=float(expit(z-offset))
            scores.append(p)
        score=float(np.dot(self.weights,scores));self.peak=max(self.peak,score)
        return (1-self.max_blend)*score+self.max_blend*self.peak
