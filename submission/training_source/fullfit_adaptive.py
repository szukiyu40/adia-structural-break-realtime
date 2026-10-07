"""Fit the accepted fixed conditional Bayes recipe on all series using OOF core inputs."""
from pathlib import Path
import gc,json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
from fit_adaptive import ROOT,OUT,CORE,FEATURES_ROOT,FEATURE_NAMES,STRIDE,PARAMS,ROUNDS,ALPHA,logits,sha,write,read,now

def main():
    started=time.perf_counter();result=read(OUT/'result.json')
    assert result['eligible_for_serving_verification'] and result['positive_folds']>=4
    out=OUT/'fullfit';out.mkdir(exist_ok=False)
    lengths=np.load(ROOT/'outputs/models/full_parent_scope_v179/v35/online_lengths.npy');counts=(lengths+STRIDE-1)//STRIDE;offsets=np.r_[0,np.cumsum(counts)]
    global_ids=np.repeat(np.arange(10000,dtype=np.int32),counts);ticks=np.concatenate([np.arange(0,n,STRIDE,dtype=np.int16) for n in lengths]);retain=ticks%16==0
    ids=global_ids[retain];tick=ticks[retain];all_rows=offsets[ids]+tick//STRIDE
    core=np.full(len(global_ids),np.nan);core_sources={}
    for fold in range(5):
        folder=CORE/f'fold{fold}';scope=read(folder/'scope.json');p=folder/'evaluation.npz'
        assert sha(p)==scope['files']['evaluation.npz'] and scope['common_outer_excluded_from_both_nested_bases']
        with np.load(p) as z:
            sid=z['id'];t=z['tick'];score=z['core']
        mask=t%STRIDE==0;row=offsets[sid[mask]]+t[mask]//STRIDE
        assert np.all(sid%5==fold) and np.isnan(core[row]).all();core[row]=score[mask]
        core_sources[str(p)]=sha(p)
    assert np.isfinite(core).all();core=core[retain]
    tau_path=ROOT/'work/structural-break-real-time-dominant-wolf/data/y_train_index.parquet';tau=pd.read_parquet(tau_path).loc[np.arange(10000),'tau_index'].to_numpy(np.int64)
    labels=np.where(np.arange(1000)[None,:]<lengths[:,None],((tau[:,None]>=0)&(np.arange(1000)[None,:]>=tau[:,None])).astype(np.int8),-1).astype(np.int8)
    y=((tau[ids]>=0)&(tick>=tau[ids])).astype(np.int8);n1=(labels==1).sum(0);n0=(labels==0).sum(0)
    w=np.where(y==1,n0[tick],n1[tick]).astype(np.float64);w*=np.where(y==1,np.clip((tick-tau[ids]+1)/8,0,1),1);w/=w.mean()
    e=read(FEATURES_ROOT/'extraction_result.json');assert e['features_sha256']==sha(FEATURES_ROOT/'features.npy')
    values=np.load(FEATURES_ROOT/'features.npy',mmap_mode='r')[all_rows]
    context=list(range(24,len(FEATURE_NAMES)));dyn=[i for i in range(24) if not FEATURE_NAMES[i].endswith('_mean')]
    phase=np.load(Path(__file__).resolve().parent/'adaptive_variance.npy',mmap_mode='r')[all_rows]
    xc=np.c_[logits(core),values[:,context+dyn]];xa=np.c_[xc,phase]
    protocol=dict(at_utc=now(),source_sha256=sha(__file__),comparison_result_sha256=sha(OUT/'result.json'),
        comparison_protocol_sha256=sha(OUT/'protocol.json'),features_sha256=e['features_sha256'],core_sources=core_sources,tau_sha256=sha(tau_path),
        training_series=10000,training_rows=len(y),parameters=PARAMS,rounds=ROUNDS,alpha=ALPHA,
        training_core_predictions='Out-of-fold evaluation scores; each series excluded from all components used to generate its core score.',
        serving_core='Unmodified submission64 full-fit core, applied as a second correction after V204; 8000-series OOF to10000-series full-fit training-size difference remains.',
        held_evaluation_or_official_labels_used=False,official_score=False,official_goal_achieved=False)
    write(out/'protocol.json',protocol);models={}
    for name,x in [('control',xc),('augmented',xa)]:
        d=lgb.Dataset(x,label=y,weight=w,init_score=logits(core),free_raw_data=True)
        model=lgb.train(PARAMS,d,num_boost_round=ROUNDS);path=out/f'{name}.txt';model.save_model(str(path))
        loaded=lgb.Booster(model_file=str(path));assert np.array_equal(model.predict(x[:128],raw_score=True,num_threads=1),loaded.predict(x[:128],raw_score=True,num_threads=1))
        models[name]=dict(sha256=sha(path),features=x.shape[1],trees=model.current_iteration())
        del d,model,loaded;gc.collect()
    meta=dict(version="adaptive_variance_20260916",decays=[.95,.99,.995],windows=[16,64,256],extra_feature_count=72,alpha=ALPHA,stride=STRIDE,clip=1.,context_columns=context,dynamic_columns=dyn,feature_names=FEATURE_NAMES,models=models,
        formula='sigmoid(logit(parent) + alpha * clip(augmented_raw - control_raw, -1, 1))',protocol_sha256=sha(out/'protocol.json'))
    write(out/'metadata.json',meta)
    result=dict(completed=True,at_utc=now(),series=10000,rows=len(y),models=models,metadata_sha256=sha(out/'metadata.json'),protocol_sha256=sha(out/'protocol.json'),seconds=time.perf_counter()-started,official_goal_achieved=False)
    write(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
