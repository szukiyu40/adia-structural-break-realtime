"""Deterministic one-observation-at-a-time prediction API."""
from pathlib import Path
import numpy as np
import torch
from network import Detector
from features import FeatureStream

class StreamingDetector:
    def __init__(self,checkpoint_path):
        torch.set_num_threads(1)
        bundle=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        c=bundle['config'];self.model=Detector(c['kind'],c['hidden'],c['dropout'])
        self.model.load_state_dict(bundle['state_dict']);self.model.eval()
        self.kind=c['kind']

    def begin(self,history):
        self.features=FeatureStream(history)
        self.context=torch.from_numpy(self.features.ctx.copy()).unsqueeze(0)
        self.state=None

    def consume(self,point):
        feature=self.features.consume(point)
        with torch.inference_mode():
            score,self.state=self.model(torch.from_numpy(feature).reshape(1,1,-1),self.context,self.state)
        return float(torch.sigmoid(score[0,0]))

def infer(datasets,model_directory_path):
    model=StreamingDetector(Path(model_directory_path)/'best.pt')
    yield
    for history,online in datasets:
        model.begin(history)
        for point in online:yield model.consume(point)
