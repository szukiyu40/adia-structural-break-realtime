"""Test nonlinear PIT dependence as an addition to fixed Public-best71/V204."""
# Publication change: private paths use configurable ADIA_* locations.
import os
import sys
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import gc,json,time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit,logit
import sys
sys.path.insert(0,str(Path(os.environ.get('ADIA_LEGACY_ROOT', str(Path.cwd() / 'external-artifacts/legacy'))) / 'work'))
from discord_bayes_v198 import ROOT,OUT as FEATURES_ROOT,FEATURE_NAMES,STRIDE,sha,write,read,now
from fit_time_rcd_online_v197 import metric

HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'outputs/adaptive_variance_experiment'
CORE=ROOT/'outputs/models/full_parent_scope_v179/core'
PARENT=ROOT/'outputs/models/full_parent_selected_v192'
PARAMS=dict(objective='binary',learning_rate=.03,num_leaves=7,max_depth=3,min_data_in_leaf=10000,
    lambda_l2=100.,lambda_l1=1.,feature_fraction=1.,bagging_fraction=1.,max_bin=63,num_threads=4,
    seed=19901,deterministic=True,force_col_wise=True,verbosity=-1,boost_from_average=False)
ROUNDS=300
ALPHA=.5

def logits(p):return logit(np.clip(p,1e-7,1-1e-7))

def main():
    start_all=time.perf_counter();OUT.mkdir(exist_ok=False)
    initial=read(FEATURES_ROOT/'comparison/result.json')
    assert initial['completed']
    assert read(FEATURES_ROOT/'comparison_completion.json')['observed_exit_code']==0
    feature_result=read(FEATURES_ROOT/'extraction_result.json');assert sha(FEATURES_ROOT/'features.npy')==feature_result['features_sha256']
    features=np.load(FEATURES_ROOT/'features.npy',mmap_mode='r')
    lengths=np.load(ROOT/'outputs/models/full_parent_scope_v179/v35/online_lengths.npy');counts=(lengths+STRIDE-1)//STRIDE
    sample_offsets=np.r_[0,np.cumsum(counts)];full_offsets=np.r_[0,np.cumsum(lengths)]
    tau_path=ROOT/'work/structural-break-real-time-dominant-wolf/data/y_train_index.parquet'
    tau=pd.read_parquet(tau_path).loc[np.arange(10000),'tau_index'].to_numpy(np.int64)
    labels=np.where(np.arange(1000)[None,:]<lengths[:,None],((tau[:,None]>=0)&(np.arange(1000)[None,:]>=tau[:,None])).astype(np.int8),-1).astype(np.int8)
    verification=read(PARENT/'verification.json');receipt=read(PARENT/'completion.json')
    assert receipt['observed_verification_exit_code']==0 and receipt['verification_sha256']==sha(PARENT/'verification.json')
    assert verification['evaluation_probability_sha256']==sha(PARENT/'evaluation_probability.npy')
    parent_flat=np.load(ROOT/'outputs/models/bayes_strength_v204/primary_probability.npy');parent=np.zeros((10000,1000),np.float64)
    for sid,n in enumerate(lengths):parent[sid,:n]=parent_flat[full_offsets[sid]:full_offsets[sid+1]]
    dyn=[i for i in range(24) if not FEATURE_NAMES[i].endswith('_mean')]
    context=list(range(24,len(FEATURE_NAMES)))
    extra=read(HERE/'work/adaptive_extraction.json');assert extra['completed'] and extra['prefix_checks_exact']==12
    assert sha(HERE/'work/adaptive_variance.npy')==extra['features_sha256']
    history_features=np.load(HERE/'work/adaptive_variance.npy',mmap_mode='r')
    protocol=dict(at_utc=now(),source_sha256=sha(__file__),additional_features_sha256=extra['features_sha256'],additional_feature_source_sha256=extra['source_sha256'],baseline_submission=71,previous_result_sha256=sha(FEATURES_ROOT/'comparison/result.json'),
        feature_source_sha256=feature_result['source_sha256'],features_sha256=feature_result['features_sha256'],
        parent_probability_sha256=sha(ROOT/'outputs/models/bayes_strength_v204/primary_probability.npy'),tau_sha256=sha(tau_path),
        fold_rule='series_id modulo 5',training_series=8000,held_series=2000,training_stride=16,prediction_stride=STRIDE,
        control_feature_names=['core_logit']+[FEATURE_NAMES[i] for i in context+dyn],added_feature_names='72 adaptive variance normalized mean/energy features; raw and AR residual; lagged EWMA .95/.99/.995; windows16/64/256; causal',
        parameters=PARAMS,rounds=ROUNDS,alpha=ALPHA,initial_score='Core logit; base predictions exclude the outer and their own inner fold.',
        training_core_meta_is_in_sample=True,core_meta_parameter_count=3,
        control='Same history/time columns, loss, initial scores and training schedule; remove only asymmetric volatility columns; keep all existing Bayes features.',
        primary_formula='sigmoid(logit(V204) + 0.5 * clip(augmented_raw_residual - control_raw_residual, -1, 1))',
        secondary_diagnostic='sigmoid(logit(V204) + 0.5 * clip(augmented_raw_residual, -1, 1))',
        full_parent_predictions_used_in_fit=False,outer_held_labels_used_in_fit=False,future_values_used=False,
        validation_reuses_development_series=True,new_unseen_test=False,official_score=False,official_goal_achieved=False,
        numpy=np.__version__,lightgbm=lgb.__version__)
    write(OUT/'protocol.json',protocol)
    primary=parent.copy();augmented=parent.copy();folds=[]
    for fold in range(5):
        started=time.perf_counter();folder=OUT/f'fold{fold}';folder.mkdir()
        core_folder=CORE/f'fold{fold}';scope=read(core_folder/'scope.json')
        assert scope['common_outer_excluded_from_both_nested_bases'] and scope['meta_training_ids_exclude_outer']
        for name in ('training.npz','evaluation.npz'):assert sha(core_folder/name)==scope['files'][name]
        with np.load(core_folder/'training.npz') as z:
            ids=z['id'];ticks=z['tick'];core=z['core'];target=z['target']
        retained=ticks%16==0;ids=ids[retained];ticks=ticks[retained];core=core[retained];target=target[retained]
        assert np.all(ids%5!=fold) and len(np.unique(ids))==8000
        y=((tau[ids]>=0)&(ticks>=tau[ids])).astype(np.int8)
        assert np.array_equal(y,target)
        rows=sample_offsets[ids]+ticks//STRIDE;z=features[rows]
        xc=np.c_[logits(core),z[:,context+dyn]];xa=np.c_[xc,history_features[rows]]
        keep=np.arange(10000)%5!=fold;a=labels[keep];n1=(a==1).sum(0);n0=(a==0).sum(0)
        w=np.where(y==1,n0[ticks],n1[ticks]).astype(np.float64)
        w*=np.where(y==1,np.clip((ticks-tau[ids]+1)/8,0,1),1);w/=w.mean()
        with np.load(core_folder/'evaluation.npz') as z:
            eval_ids=z['id'];eval_ticks=z['tick'];eval_core=z['core']
        retain=eval_ticks%STRIDE==0;ei=eval_ids[retain];et=eval_ticks[retain];ec=eval_core[retain]
        assert np.all(ei%5==fold) and len(np.unique(ei))==2000
        ev=features[sample_offsets[ei]+et//STRIDE];ecx=np.c_[logits(ec),ev[:,context+dyn]];eax=np.c_[ecx,history_features[sample_offsets[ei]+et//STRIDE]]
        predictions={};models={}
        for name,x,xx in [('control',xc,ecx),('augmented',xa,eax)]:
            data=lgb.Dataset(x,label=y,weight=w,init_score=logits(core),free_raw_data=True)
            booster=lgb.train(PARAMS,data,num_boost_round=ROUNDS);p=folder/f'{name}.txt';booster.save_model(str(p))
            pred=booster.predict(xx,raw_score=True,num_threads=4)
            restored=lgb.Booster(model_file=str(p));assert np.array_equal(pred[:128],restored.predict(xx[:128],raw_score=True,num_threads=4))
            predictions[name]=pred;models[name]=sha(p)
            del data,booster,restored;gc.collect()
        difference=np.clip(predictions['augmented']-predictions['control'],-1,1)
        standalone=np.clip(predictions['augmented'],-1,1)
        np.savez_compressed(folder/'sampled_corrections.npz',id=ei,tick=et,difference=difference,augmented=standalone)
        cursor=0;held_ids=np.flatnonzero(~keep)
        for sid in held_ids:
            count=int(counts[sid]);assert np.all(ei[cursor:cursor+count]==sid)
            n=int(lengths[sid]);base=logits(parent[sid,:n])
            primary[sid,:n]=expit(base+ALPHA*np.repeat(difference[cursor:cursor+count],STRIDE)[:n])
            augmented[sid,:n]=expit(base+ALPHA*np.repeat(standalone[cursor:cursor+count],STRIDE)[:n]);cursor+=count
        assert cursor==len(ei)
        r=dict(fold=fold,at_utc=now(),training_rows=len(y),held_rows=len(ei),models=models,
            parent=metric(labels[held_ids],parent[held_ids]),conditional_difference=metric(labels[held_ids],primary[held_ids]),
            augmented_residual_diagnostic=metric(labels[held_ids],augmented[held_ids]),seconds=time.perf_counter()-started,
            core_scope_sha256=sha(core_folder/'scope.json'),outer_excluded=True,reload_exact=True)
        r['primary_minus_parent']=r['conditional_difference']-r['parent'];folds.append(r)
        write(folder/'result.json',r);print('CONDITIONAL_BAYES_FOLD',json.dumps(r),flush=True)
        del xc,xa,ecx,eax,ev,z;gc.collect()
    pooled=dict(parent=metric(labels,parent),conditional_difference=metric(labels,primary),augmented_residual_diagnostic=metric(labels,augmented))
    result=dict(completed=True,at_utc=now(),source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),series=10000,
        pooled=pooled,folds=folds,primary_minus_parent=pooled['conditional_difference']-pooled['parent'],positive_folds=sum(f['primary_minus_parent']>0 for f in folds),
        seconds=time.perf_counter()-start_all,official_score=False,official_goal_achieved=False)
    result['eligible_for_serving_verification']=result['primary_minus_parent']>.0002 and result['positive_folds']>=4
    write(OUT/'result.json',result);print('CONDITIONAL_BAYES_DONE',json.dumps(result),flush=True)

if __name__=='__main__':main()
