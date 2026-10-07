"""Offline full-data fit for the extended V120 tree with four ordered CPU threads.

Accept only organizer training parquets. The frozen encoder checkpoint, selected
tree count, feature arm, and LightGBM parameters come from an explicit recipe.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import argparse,json,hashlib,time,importlib.metadata
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import encoder_change_v108 as features

SUMMARY=np.r_[np.arange(768,778),np.arange(1546,1558)]
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')
def labels_from_raw(data,ids=None):
    filter_=None if ids is None else [('id','in',np.asarray(ids).tolist())]
    index=pq.read_table(Path(data)/'y_train_index.parquet',columns=['id','tau_index'],filters=filter_).to_pandas(ignore_metadata=True)
    index=index.sort_values('id');canonical=index.id.to_numpy(np.int64)
    assert len(canonical)==len(set(canonical.tolist()))
    if ids is not None:assert np.array_equal(canonical,ids)
    frame=pq.read_table(Path(data)/'y_train.parquet',columns=['id','time','target'],filters=filter_).to_pandas(ignore_metadata=True)
    labels=np.full((len(canonical),1000),-1,np.int8);lengths=np.empty(len(canonical),np.int64)
    groups=frame.groupby('id',sort=True);assert groups.ngroups==len(canonical)
    for i,(sid,g) in enumerate(groups):
        assert sid==canonical[i];g=g.sort_values('time');n=len(g);tau=int(index.tau_index.iloc[i])
        assert 1<=n<=1000 and -1<=tau<n and np.all(np.diff(g.time)>0)
        y=g.target.to_numpy(np.int8)
        assert np.array_equal(y,((tau>=0)&(np.arange(n)>=tau)).astype(np.int8))
        labels[i,:n]=y;lengths[i]=n
    return canonical,lengths,labels
def weighted_sampling(ids,labels):
    positives,negatives=(labels==1).sum(0),(labels==0).sum(0)
    weights=np.where(labels==1,negatives[None],positives[None]).astype(np.float32)
    weights[labels<0]=0;weights/=weights.sum()/np.count_nonzero(labels>=0)
    weights*=np.where(labels==1,np.minimum(np.cumsum(labels==1,axis=1)/8,1),1)
    selected=(labels>=0)&(weights>0)&((np.arange(1000)[None]%16)==(ids[:,None]%16))
    return selected,weights
def raw_series(data,ids):
    for lo in range(0,len(ids),25):
        chosen=ids[lo:lo+25]
        frame=pq.read_table(Path(data)/'X_train.parquet',columns=['id','period','time','value'],filters=[('id','in',chosen.tolist())]).to_pandas(ignore_metadata=True)
        groups={int(sid):g for sid,g in frame.groupby('id',sort=False)}
        assert set(groups)==set(chosen.tolist())
        for sid in chosen:
            g=groups[int(sid)].sort_values('time')
            h=g.loc[g.period==1,'value'].to_numpy(np.float64);x=g.loc[g.period==2,'value'].to_numpy(np.float64)
            assert len(h)>=1000 and 1<=len(x)<=1000
            yield int(sid),h,x
def fitting_features(data,encoder,ids,lengths,selected,arm):
    assert arm in ('summary','full');rows,ticks=np.nonzero(selected)
    x=np.empty((len(rows),len(SUMMARY) if arm=='summary' else features.DIM),np.float32)
    records=[];offset=0;began=time.perf_counter()
    def extract(item):
        i,(sid,h,online)=item
        assert sid==ids[i] and len(online)==lengths[i]
        reference=features.Reference(encoder,h)
        updates=reference.at(online,np.arange(0,len(online),features.STEP))
        chosen=np.flatnonzero(selected[i]);full=features.expand_updates(updates,len(online))[chosen]
        values=full[:,SUMMARY] if arm=='summary' else full
        record=dict(id=sid,history_sha256=array_sha(h),online_sha256=array_sha(online),
            update_features_sha256=array_sha(updates),ticks_sha256=array_sha(chosen),
            full_selected_features_sha256=array_sha(full),selected_features_sha256=array_sha(values),sampled_points=len(chosen))
        return i,values,record
    def ordered():
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending=deque()
            for item in enumerate(raw_series(data,ids)):
                pending.append(pool.submit(extract,item))
                if len(pending)>=8:yield pending.popleft().result()
            while pending:yield pending.popleft().result()
    for i,values,record in ordered():
        assert i==len(records)
        x[offset:offset+len(values)]=values;offset+=len(values);records.append(record)
        if (i+1)%100==0 or i+1==len(ids):print('FEATURES',i+1,len(ids),round(time.perf_counter()-began,2),'workers',4,flush=True)
    assert offset==len(rows) and np.isfinite(x).all()
    return x,records
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--checkpoint',required=True,type=Path);ap.add_argument('--recipe',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--ids',type=Path)
    args=ap.parse_args();recipe=json.loads(args.recipe.read_text());args.out.mkdir(parents=True)
    assert recipe['arm'] in ('summary','full') and recipe['rounds'] in (20,50,100,200,400,800,1600)
    assert recipe['calibration_coefficient']==.25 and recipe['feature_source_sha256']==sha(features.__file__)
    for name,digest in recipe['checkpoint_sha256'].items():assert sha(args.checkpoint/name)==digest
    manifest=dict(recipe=recipe,recipe_sha256=sha(args.recipe),source_sha256=sha(__file__),
        feature_source_sha256=sha(features.__file__),
        inputs={name:sha(args.data/name) for name in ('X_train.parquet','y_train.parquet','y_train_index.parquet')},
        versions={n:importlib.metadata.version(n) for n in ('numpy','lightgbm','torch','chronos-forecasting','transformers')},
        feature_workers=4,official_test_read=False,training_mode='full' if args.ids is None else 'explicit_subset_probe')
    write(args.out/'protocol.json',manifest)
    ids,lengths,labels=labels_from_raw(args.data,None if args.ids is None else np.load(args.ids))
    if args.ids is None:assert len(ids)==10000
    selected,weights=weighted_sampling(ids,labels);rows,ticks=np.nonzero(selected)
    features.CHECKPOINT=args.checkpoint;encoder=features.Encoder()
    x,records=fitting_features(args.data,encoder,ids,lengths,selected,recipe['arm'])
    y,w=labels[rows,ticks],weights[rows,ticks]
    write(args.out/'fitting.json',dict(ids=ids.tolist(),lengths=lengths.tolist(),records=records,
        features_sha256=array_sha(x),labels_sha256=array_sha(y),weights_sha256=array_sha(w),sampled_points=len(y)))
    np.save(args.out/'training_ids.npy',ids);began=time.perf_counter()
    model=lgb.train(recipe['lightgbm'],lgb.Dataset(x,label=y,weight=w,free_raw_data=True),num_boost_round=recipe['rounds'])
    assert model.current_iteration()==recipe['rounds']
    model.save_model(str(args.out/'model.txt'));loaded=lgb.Booster(model_file=str(args.out/'model.txt'))
    assert np.array_equal(model.predict(x[:129]),loaded.predict(x[:129]))
    result=dict(completed=True,training_series=len(ids),sampled_points=len(y),arm=recipe['arm'],rounds=recipe['rounds'],
        model_sha256=sha(args.out/'model.txt'),serialized_predictions_exact=True,tree_fit_seconds=time.perf_counter()-began,
        protocol_sha256=sha(args.out/'protocol.json'),fitting_sha256=sha(args.out/'fitting.json'),official_test_read=False)
    write(args.out/'result.json',result);print(result,flush=True)
if __name__=='__main__':main()
