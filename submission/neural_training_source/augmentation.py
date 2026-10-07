"""Attach paired synthetic examples only for permitted original parent IDs."""
from pathlib import Path
import hashlib,json
import numpy as np
from refit import JoinedFeatures

class SelectedFeatures:
    def __init__(self,base,rows,width):
        self.base=base;self.rows=np.asarray(rows);self.width=width
        self.shape=(len(rows),base.shape[1],width)
    def __len__(self):return self.shape[0]
    def __getitem__(self,key):
        rows,steps=key
        return self.base[self.rows[rows],steps,:self.width]

def augment_training(real,synthetic_cache,weight=.35,prefix=False):
    path=Path(synthetic_cache)/'all'
    metadata=json.loads((path/'metadata.json').read_text())
    parent=np.load(path/'parent_ids.npy',mmap_mode='r');allowed=np.asarray(real['ids'])
    rows=np.flatnonzero(np.isin(parent,allowed));selected_parent=np.asarray(parent[rows])
    assert np.isin(selected_parent,allowed).all()
    source={k:np.load(path/f'{k}.npy',mmap_mode='r') for k in ['features','context','labels','ids','lengths']}
    width=real['features'].shape[-1]
    if prefix:assert width<=source['features'].shape[-1]
    else:assert width==source['features'].shape[-1]
    assert not np.intersect1d(real['ids'],source['ids'][rows]).size
    data={k:np.concatenate([np.asarray(real[k]),np.asarray(source[k][rows])]) for k in ['context','labels','ids','lengths']}
    data['features']=JoinedFeatures([real['features'],SelectedFeatures(source['features'],rows,width)])
    data['series_weights']=np.r_[np.ones(len(real['ids']),np.float32),np.full(len(rows),weight,np.float32)]
    protocol={'real_training_series':len(allowed),'synthetic_examples':len(rows),'synthetic_parent_count':len(np.unique(selected_parent)),
       'synthetic_weight':weight,'synthetic_cache':str(Path(synthetic_cache).resolve()),'prefix_feature_selection':prefix,
       'feature_width':width,'source_feature_width':source['features'].shape[-1],
       'parent_ids_sha256':hashlib.sha256(np.sort(allowed).tobytes()).hexdigest(),
       'no_validation_parents_in_synthetic_training':True,'source_metadata':metadata}
    return data,protocol
