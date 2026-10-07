"""Per-series serving adapter for a selected V108 tree and frozen encoder."""
import json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.special import expit
import encoder_change_v108 as features

SUMMARY=np.r_[np.arange(768,778),np.arange(1546,1558)]

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

class EncoderTree:
    def __init__(self,resources):
        root=Path(resources);self.metadata=json.loads((root/'metadata.json').read_text())
        assert self.metadata['arm'] in ('summary','full')
        assert self.metadata['calibration_coefficient']==.25
        model_path=root/self.metadata['model_file']
        assert sha(model_path)==self.metadata['model_sha256']
        checkpoint=root/self.metadata['checkpoint_directory']
        for name,digest in self.metadata['checkpoint_sha256'].items():assert sha(checkpoint/name)==digest
        features.CHECKPOINT=checkpoint
        self.encoder=features.Encoder()
        self.tree=lgb.Booster(model_file=str(model_path))
        self.count=int(self.metadata['rounds'])
        assert 0<self.count<=self.tree.current_iteration()
        assert self.tree.num_feature()==(22 if self.metadata['arm']=='summary' else features.DIM)
        self.stream=None;self.normal_q90=None

    def values(self,x):
        return x[:,SUMMARY] if self.metadata['arm']=='summary' else x

    def begin(self,history):
        history=np.asarray(history,np.float64)
        k=min(512,max(200,len(history)//5))
        reference=features.Reference(self.encoder,history[:-k])
        update=reference.at(history[-k:],np.arange(0,k,features.STEP))
        normal=features.expand_updates(update,k)
        logits=self.tree.predict(self.values(normal),num_iteration=self.count,raw_score=True,num_threads=1)
        self.normal_q90=float(np.quantile(logits[64:],.9))
        self.stream=features.Stream(self.encoder,history)

    def consume(self,point):
        assert self.stream is not None
        values=self.values(self.stream.consume(point)[None])
        logit=float(self.tree.predict(values,num_iteration=self.count,raw_score=True,num_threads=1)[0])
        probability=float(expit(logit-.25*self.normal_q90))
        assert np.isfinite(probability) and 0<=probability<=1
        return probability
