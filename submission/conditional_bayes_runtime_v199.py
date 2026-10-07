"""Serving-only conditional Bayes correction with current-prefix inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.special import expit,logit
import lightgbm as lgb
import discord_bayes_features_v198 as features

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Resources:
    def __init__(self,folder):
        root=Path(folder);self.meta=json.loads((root/'metadata.json').read_text())
        assert self.meta['alpha']==.5 and self.meta['stride']==4 and self.meta['clip']==1.
        assert self.meta['feature_names']==features.FEATURE_NAMES
        self.models={}
        for name in ['control','augmented']:
            path=root/f'{name}.txt';spec=self.meta['models'][name]
            assert sha(path)==spec['sha256']
            self.models[name]=lgb.Booster(model_file=str(path))
            assert self.models[name].num_feature()==spec['features'] and self.models[name].current_iteration()==spec['trees']

class State:
    def __init__(self,history,resources):
        self.history=np.asarray(history,np.float64).copy();self.online=[];self.resources=resources
        self.tick=-1;self.correction=0.;self.last_features=None
    def update(self,point):
        self.online.append(float(point));self.tick+=1
    def apply(self,parent,core):
        if self.tick%4==0:
            row=features.features(self.history,np.asarray(self.online),[self.tick])[0]
            meta=self.resources.meta
            control=np.r_[logit(np.clip(core,1e-7,1-1e-7)),row[meta['context_columns']]][None,:]
            augmented=np.c_[control,row[meta['dynamic_columns']][None,:]]
            a=float(self.resources.models['augmented'].predict(augmented,raw_score=True,num_threads=1)[0])
            b=float(self.resources.models['control'].predict(control,raw_score=True,num_threads=1)[0])
            self.correction=.5*float(np.clip(a-b,-1,1));self.last_features=row
        return float(expit(logit(np.clip(parent,1e-7,1-1e-7))+self.correction))
