"""Fit a break classifier on frozen Time-RCD representations; five held-series folds."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse, gc, hashlib, json, time, datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'outputs/models'
PARENT=MODEL/'full_parent_selected_v192'
LENGTHS=np.load(MODEL/'full_parent_scope_v179/v35/online_lengths.npy')
OFFSETS=np.r_[0,np.cumsum(LENGTHS)]
FEATURE_SOURCE_SHA='2098f01067f17992995bb5f1d531972fb2202db4cb77d1d04cf35882dc536b8b'
PARAMS=dict(objective='binary',learning_rate=.03,num_leaves=31,max_depth=-1,
    min_data_in_leaf=250,lambda_l2=10.,lambda_l1=.1,feature_fraction=.7,
    bagging_fraction=.8,bagging_freq=1,max_bin=127,num_threads=4,
    seed=19701,feature_fraction_seed=19701,bagging_seed=19701,data_random_seed=19701,
    deterministic=True,force_col_wise=True,verbosity=-1,boost_from_average=False)
ROUNDS=800

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())

def metric(y,p):
    wins=0.;pairs_total=0
    for t in range(y.shape[1]):
        keep=y[:,t]>=0;yy=y[keep,t];pp=p[keep,t]
        positive=int(yy.sum());negative=len(yy)-positive;pairs=positive*negative
        if pairs:
            wins+=float(rankdata(pp)[yy==1].sum()-positive*(positive+1)/2)
            pairs_total+=pairs
    return wins/pairs_total

def assemble(folder):
    rows=int(np.sum((LENGTHS+15)//16));x=np.empty((rows,1062),np.float32)
    sid_values=np.empty(rows,np.int32);ticks=np.empty(rows,np.int16)
    zero=np.empty((rows,2),np.float64);offset=0
    for sid,n in enumerate(LENGTHS):
        path=folder/f'{sid:05d}.npz'
        with np.load(path) as z:
            t=z['ticks'];e=z['embedding'].astype(np.float32);s=z['summary'].astype(np.float32)
            ne=z['normal_embedding'].astype(np.float32);ns=z['normal_summary'].astype(np.float32)
            assert int(z['online_length'])==n and np.array_equal(t,np.arange(0,n,16))
        assert e.shape==(len(t),512) and s.shape==(len(t),12) and ne.shape==(512,) and ns.shape==(12,)
        values=np.concatenate([e,e-ne,s,s-ns,np.broadcast_to(ns,(len(t),12)),
            (t/1000.)[:,None],(np.log1p(t)/np.log(1001.))[:,None]],axis=1)
        assert values.shape==(len(t),1062) and np.isfinite(values).all()
        span=slice(offset,offset+len(t));x[span]=values;sid_values[span]=sid;ticks[span]=t
        # Fixed zero-shot diagnostics: average 16-observation anomaly logit across two contexts.
        zero[span,0]=expit(.5*(s[:,1]+s[:,7]))
        zero[span,1]=expit(.5*(s[:,1]+s[:,7]-ns[1]-ns[7]))
        offset+=len(t)
    assert offset==rows
    return x,sid_values,ticks,zero

def expand(sampled,ids):
    result=np.zeros((len(ids),1000),np.float64)
    sample_offsets=np.r_[0,np.cumsum((LENGTHS+15)//16)]
    for i,sid in enumerate(ids):
        result[i,:LENGTHS[sid]]=np.repeat(sampled[sample_offsets[sid]:sample_offsets[sid+1]],16)[:LENGTHS[sid]]
    return result

def main(args):
    began=time.perf_counter();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    folder=Path(args.features);protocol=read(args.feature_protocol)
    assert protocol['source_sha256']==FEATURE_SOURCE_SHA and protocol['fixed_compute_batch']==32
    assert protocol['future_values_used'] is False and protocol['labels_read'] is False
    receipt=read(PARENT/'completion.json');verification=read(PARENT/'verification.json')
    assert receipt['observed_verification_exit_code']==0 and verification['full_parent_reconstruction_complete']
    assert receipt['verification_sha256']==sha(PARENT/'verification.json')
    parent_path=PARENT/'evaluation_probability.npy'
    assert sha(parent_path)==verification['evaluation_probability_sha256']
    parent_flat=np.load(parent_path,mmap_mode='r');parent=np.zeros((10000,1000),np.float64)
    for sid,n in enumerate(LENGTHS):parent[sid,:n]=parent_flat[OFFSETS[sid]:OFFSETS[sid+1]]
    tau_path=ROOT/'work/structural-break-real-time-dominant-wolf/data/y_train_index.parquet'
    tau=pd.read_parquet(tau_path).loc[np.arange(10000),'tau_index'].to_numpy(np.int64)
    clock=np.arange(1000)[None,:]
    labels=np.where(clock<LENGTHS[:,None],((tau[:,None]>=0)&(clock>=tau[:,None])).astype(np.int8),-1).astype(np.int8)
    run_protocol=dict(at_utc=now(),source_sha256=sha(__file__),feature_protocol_sha256=sha(args.feature_protocol),
        feature_source_sha256=FEATURE_SOURCE_SHA,parent_probability_sha256=sha(parent_path),tau_sha256=sha(tau_path),
        lightgbm_version=lgb.__version__,numpy_version=np.__version__,pandas_version=pd.__version__,
        parameters=PARAMS,rounds=ROUNDS,fold_rule='series_id modulo 5',training_series=8000,held_series=2000,
        feature_columns=1062,classification_target='break has occurred by current observation',
        feature_freezing='External anomaly-detection encoder stays frozen; train supervised LightGBM head.',
        direct_weight=.125,parent_weight=.875,loss_weights='Opposite-class counts at each tick from training series only; positive first-eight-observation ramp.',
        future_values_used=False,parent_predictions_used_in_fit=False,held_labels_used_in_fit=False,
        validation_reuses_development_series=True,new_unseen_test=False,official_goal_achieved=False)
    write(out/'protocol.json',run_protocol)
    x,row_ids,ticks,zero=assemble(folder);print('RCD_HEAD_INPUT_READY',x.shape,time.perf_counter()-began,flush=True)
    y=((tau[row_ids]>=0)&(ticks>=tau[row_ids])).astype(np.int8)
    sampled=np.empty(len(x),np.float64);folds=[]
    for outer in range(5):
        start=time.perf_counter();keep=np.arange(10000)%5!=outer;train=keep[row_ids];held=~train
        a=labels[keep];n1=(a==1).sum(axis=0);n0=(a==0).sum(axis=0)
        w=np.where(y[train]==1,n0[ticks[train]],n1[ticks[train]]).astype(np.float64)
        ramp=np.clip((ticks[train]-tau[row_ids[train]]+1)/8.,0,1)
        w*=np.where(y[train]==1,ramp,1.);w/=w.mean()
        fit_x=x[train];fit_y=y[train]
        dataset=lgb.Dataset(fit_x,label=fit_y,weight=w,free_raw_data=True)
        def progress(env):
            if (env.iteration+1)%100==0:
                r=dict(at_utc=now(),fold=outer,iteration=env.iteration+1,seconds=time.perf_counter()-start)
                write(out/'progress.json',r);print('RCD_HEAD_PROGRESS',json.dumps(r),flush=True)
        booster=lgb.train(PARAMS,dataset,num_boost_round=ROUNDS,callbacks=[progress])
        model_path=out/f'fold{outer}.txt';booster.save_model(str(model_path))
        values=booster.predict(x[held],num_threads=4)
        restored=lgb.Booster(model_file=str(model_path));probe=x[held][:min(128,int(held.sum()))]
        assert np.array_equal(booster.predict(probe),restored.predict(probe))
        sampled[held]=values
        np.savez_compressed(out/f'fold{outer}_predictions.npz',id=row_ids[held],tick=ticks[held],probability=values)
        ids=np.flatnonzero(~keep);direct=expand(sampled,ids);base=parent[ids]
        comparison=dict(parent=metric(labels[ids],base),direct=metric(labels[ids],direct),
            fixed_blend=metric(labels[ids],.875*base+.125*direct))
        record=dict(outer=outer,training_series=int(keep.sum()),held_series=len(ids),training_rows=int(train.sum()),
            held_rows=int(held.sum()),model_sha256=sha(model_path),seconds=time.perf_counter()-start,**comparison)
        record['blend_minus_parent']=record['fixed_blend']-record['parent']
        folds.append(record);write(out/f'fold{outer}_result.json',record);print('RCD_FOLD_SCORE',json.dumps(record),flush=True)
        del fit_x,fit_y,dataset,booster,restored;gc.collect()
    direct=expand(sampled,np.arange(10000));blend=.875*parent+.125*direct
    pooled=dict(parent=metric(labels,parent),direct=metric(labels,direct),fixed_blend=metric(labels,blend),
        zero_shot=metric(labels,expand(zero[:,0],np.arange(10000))),
        zero_shot_history_centered=metric(labels,expand(zero[:,1],np.arange(10000))))
    np.save(out/'oof_sampled.npy',sampled)
    result=dict(completed=True,at_utc=now(),series=10000,online_points=int(LENGTHS.sum()),
        source_sha256=sha(__file__),protocol_sha256=sha(out/'protocol.json'),folds=folds,pooled=pooled,
        fixed_blend_minus_parent=pooled['fixed_blend']-pooled['parent'],
        positive_folds=sum(f['blend_minus_parent']>0 for f in folds),seconds=time.perf_counter()-began,
        TS_AUC_computed=True,official_score=False,official_goal_achieved=False)
    result['eligible_for_serving_verification']=result['fixed_blend_minus_parent']>0 and result['positive_folds']>=3
    write(out/'result.json',result);print('RCD_COMPARISON_DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--feature-protocol',required=True);ap.add_argument('--out',required=True)
    main(ap.parse_args())
