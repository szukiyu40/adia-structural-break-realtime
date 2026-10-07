"""Past-only small-change evidence inspired by the public Sept 4 Discord discussion.

Reimplementation from described mathematics, not the participant's code or result.
Each mean, scale and dependence channel stays separate until supervised fitting.
"""
# Publication change: private community link and participant metadata omitted.
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse, datetime, hashlib, json, time, gc
import numpy as np
from scipy.special import gammaln, logsumexp, ndtri
from scipy.signal import lfilter
from scipy.linalg import toeplitz

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'work/structural-break-real-time-dominant-wolf/data/X_train.parquet'
RAW_SHA='2aa61853ce0c3a8f6cdfe137ccde9a59f97cf0872779d2063b549c60a7b7ed54'
OUT=ROOT/'outputs/models/discord_bayes_v198'
DURATIONS=np.array([4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768])
MEAN_SD=np.array([.05,.12,.30])
SCALE_A=np.array([100.,30.,12.,6.,3.])
R_SHIFT=np.array([-.35,-.20,-.12,-.06,.06,.12,.20,.35])
LAGS=(1,2,4,8)
CHANNELS=('raw_z','innovation_z','raw_gaussian_rank','innovation_gaussian_rank')
FAMILIES=('mean','scale','dependence_lag1','dependence_lag2','dependence_lag4','dependence_lag8')
FEATURE_NAMES=[f'{ch}_{f}' for ch in CHANNELS for f in FAMILIES]
FEATURE_NAMES += [f'{ch}_{f}' for ch in CHANNELS for f in ('kurtosis','acf1','acf2','acf4','acf8')]
FEATURE_NAMES += [f'ar10_coefficient_{i+1}' for i in range(10)]+['history_log_length','tick','log_tick']
STRIDE=4

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())

def rank_gaussian(x,reference):
    ref=np.sort(reference)
    rank=(np.searchsorted(ref,x,'left')+np.searchsorted(ref,x,'right'))*.5
    return ndtri(np.clip((rank+.5)/(len(ref)+1),.5/(len(ref)+1),1-.5/(len(ref)+1)))

def acf(x,lag):
    a=x[lag:];b=x[:-lag]
    return float(np.clip(np.dot(a,b)/max(np.sqrt(np.dot(a,a)*np.dot(b,b)),1e-12),-.98,.98))

def prepare_channels(history,online):
    h=np.asarray(history,np.float64);o=np.asarray(online,np.float64)
    assert len(h)>=1000 and np.isfinite(h).all() and np.isfinite(o).all()
    center=h.mean();scale=max(h.std(),1e-8)
    hz=np.clip((h-center)/scale,-8,8);oz=np.clip((o-center)/scale,-8,8)
    # Fit only the normal historical interval; the filter itself is causal.
    ar_acf=np.array([1.]+[acf(hz,k) for k in range(1,11)])
    matrix=toeplitz(ar_acf[:-1]);matrix.flat[::11]+=.01
    coef=np.clip(np.linalg.solve(matrix,ar_acf[1:]),-1.5,1.5)
    e=lfilter(np.r_[1.,-coef],[1.],np.r_[hz,oz])
    eh=e[10:len(h)];eo=e[len(h):]
    ec=eh.mean();es=max(eh.std(),1e-8)
    eh=np.clip((eh-ec)/es,-8,8);eo=np.clip((eo-ec)/es,-8,8)
    hcols=[hz,eh,rank_gaussian(hz,hz),rank_gaussian(eh,eh)]
    ocols=[oz,eo,rank_gaussian(oz,hz),rank_gaussian(eo,eh)]
    context=[];r0=[]
    for col in hcols:
        r=[acf(col,lag) for lag in LAGS];r0.append(r)
        context.extend([float(np.mean(col**4)/max(np.mean(col**2)**2,1e-8)),*r])
    context.extend([*coef,np.log1p(len(h))/np.log(10001.)])
    return np.column_stack(ocols),np.asarray(r0),np.asarray(context)

def duration_grid(ticks):
    n=np.asarray(ticks,np.int64)+1
    # Include n itself: the prior weights then sum to every available tau.
    # Duplicate/end-clipped grid cells have zero mass, not extra short-window mass.
    m=np.c_[np.minimum(n[:,None],DURATIONS[None,:]),n]
    weight=np.diff(np.c_[np.zeros(len(n),np.int64),m],axis=1)
    with np.errstate(divide='ignore'):logw=np.log(weight)-np.log(n[:,None])
    assert np.all(weight.sum(axis=1)==n)
    return n,m,logw

def evidence(online_channels,r0,ticks):
    y=np.asarray(online_channels,np.float64);ticks=np.asarray(ticks,np.int64)
    n,m,logw=duration_grid(ticks);start=n[:,None]-m
    total=np.r_[np.zeros((1,4)),np.cumsum(y,axis=0)]
    sq=np.r_[np.zeros((1,4)),np.cumsum(y*y,axis=0)]
    s=total[n[:,None]]-total[start];q=sq[n[:,None]]-sq[start]
    count=m[:,:,None,None].astype(np.float64)
    var=MEAN_SD**2
    mean=-.5*np.log1p(count*var)+.5*var*s[:,:,:,None]**2/(1+count*var)
    mean=logsumexp(mean,axis=-1)-np.log(len(var))
    a=SCALE_A;b=a-1
    scale=gammaln(a+count/2)-gammaln(a)+a*np.log(b)-(a+count/2)*np.log(b+q[:,:,:,None]/2)+q[:,:,:,None]/2
    scale=logsumexp(scale,axis=-1)-np.log(len(a))
    families=[mean,scale]
    for j,lag in enumerate(LAGS):
        # Pairs both inside each candidate post-change suffix.
        cross=np.zeros_like(y);prevsq=np.zeros_like(y)
        cross[lag:]=y[lag:]*y[:-lag];prevsq[lag:]=y[:-lag]**2
        cs=np.r_[np.zeros((1,4)),np.cumsum(cross,axis=0)]
        ps=np.r_[np.zeros((1,4)),np.cumsum(prevsq,axis=0)]
        lo=np.minimum(start+lag,n[:,None]);pairs=np.maximum(m-lag,0)[:,:,None,None]
        c=cs[n[:,None]]-cs[lo];qx=ps[n[:,None]]-ps[lo];qy=sq[n[:,None]]-sq[lo]
        null=r0[:,j][None,None,:,None];alt=np.clip(null+R_SHIFT,-.98,.98)
        v0=1-null**2;v1=1-alt**2
        rss0=(qy[:,:,:,None]-2*null*c[:,:,:,None]+null**2*qx[:,:,:,None])/v0
        rss1=(qy[:,:,:,None]-2*alt*c[:,:,:,None]+alt**2*qx[:,:,:,None])/v1
        bf=-.5*pairs*np.log(v1/v0)-.5*(rss1-rss0)
        families.append(logsumexp(bf,axis=-1)-np.log(len(R_SHIFT)))
    mixed=[logsumexp(f+logw[:,:,None],axis=1) for f in families]
    return np.clip(np.stack(mixed,axis=2).reshape(len(ticks),-1),-30,100)

def features(history,online,ticks=None):
    ticks=np.arange(0,len(online),STRIDE) if ticks is None else np.asarray(ticks)
    y,r0,context=prepare_channels(history,online)
    dynamic=evidence(y,r0,ticks)
    x=np.c_[dynamic,np.broadcast_to(context,(len(ticks),len(context))),ticks/1000.,np.log1p(ticks)/np.log(1001.)].astype(np.float32)
    assert x.shape==(len(ticks),len(FEATURE_NAMES)) and np.isfinite(x).all()
    return x

def probe():
    rng=np.random.default_rng(19801);h=rng.normal(size=3000);o=rng.normal(size=400)
    ticks=np.array([0,1,3,4,8,63,64,127,255,399]);full=features(h,o,ticks)
    for i,t in enumerate(ticks):
        short=features(h,o[:t+1],[t]);mut=o.copy();mut[t+1:]=1e5
        assert np.array_equal(full[i],short[0]) and np.array_equal(full[i],features(h,mut,[t])[0])
    n,m,lw=duration_grid(np.arange(1000));assert np.allclose(np.exp(lw).sum(1),1,atol=1e-14)
    # Independent scalar likelihood identities at one fixed suffix.
    y,r,c=prepare_channels(h,o);t=127;nn,mm,logw=duration_grid(np.array([t]));terms=[]
    for size,weight in zip(mm[0],np.exp(logw[0])):
        if weight==0:continue
        yy=y[t+1-size:t+1,1];ss=float(yy.sum());qq=float(yy@yy)
        mean=np.mean([np.exp(-.5*np.log1p(size*v)+.5*v*ss*ss/(1+size*v)) for v in MEAN_SD**2])
        terms.append(weight*mean)
    value=evidence(y,r,[t])[0,6]
    assert abs(value-np.log(sum(terms)))<1e-12
    return dict(completed=True,prefix_and_future_mutation_exact=True,ticks=ticks.tolist(),uniform_tau_mass_max_error=float(np.abs(np.exp(lw).sum(1)-1).max()),scalar_mean_logbf_error=float(abs(value-np.log(sum(terms)))),feature_columns=len(FEATURE_NAMES))

def extract():
    from run_time_rcd_online_v197 import series_stream
    OUT.mkdir(exist_ok=True);assert sha(RAW)==RAW_SHA
    protocol=dict(at_utc=now(),source_sha256=sha(__file__),raw_sha256=RAW_SHA,
        community_source='Competition community discussion (mathematical description only)',
        community_messages='Parameter grids independently implemented; no participant code or reported scores reproduced.',
        feature_names=FEATURE_NAMES,stride=STRIDE,durations=DURATIONS.tolist(),mean_sd=MEAN_SD.tolist(),scale_shape=SCALE_A.tolist(),dependence_shifts=R_SHIFT.tolist(),lags=list(LAGS),
        normalizer='Normal historical mean/std, ridge Yule-Walker AR10, historical empirical Gaussian rank.',
        duration_prior='Uniform over actual tau, grid masses m minus previous m, include observed n; divide by n.',
        hypothesis='Small amplitude scale and dependence priors plus correct duration mass may complement the parent.',
        deviations_from_public_description=['Add raw and innovation Gaussian-rank channels.','Include lag2,4,8 dependence.','Include endpoint duration n to cover all observed tau.'],
        labels_read=False,future_values_used=False,official_score=False,official_goal_achieved=False)
    write(OUT/'protocol.json',protocol);write(OUT/'probe_result.json',probe())
    lengths=np.load(ROOT/'outputs/models/full_parent_scope_v179/v35/online_lengths.npy');counts=(lengths+STRIDE-1)//STRIDE;offsets=np.r_[0,np.cumsum(counts)]
    x=np.lib.format.open_memmap(OUT/'features.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1]),len(FEATURE_NAMES)))
    start=time.perf_counter();records=0
    for item in series_stream(RAW):
        sid=int(item['id'][0]);h=item['value'][item['period']==1];o=item['value'][item['period']==2]
        assert len(o)==lengths[sid];x[offsets[sid]:offsets[sid+1]]=features(h,o);records+=1
        if records%250==0:
            x.flush();p=dict(at_utc=now(),series=records,seconds=time.perf_counter()-start);write(OUT/'progress.json',p);print('BAYES_PROGRESS',json.dumps(p),flush=True)
    assert records==10000;x.flush()
    result=dict(completed=True,at_utc=now(),series=records,sampled_rows=int(offsets[-1]),feature_columns=len(FEATURE_NAMES),source_sha256=sha(__file__),features_sha256=sha(OUT/'features.npy'),protocol_sha256=sha(OUT/'protocol.json'),seconds=time.perf_counter()-start,labels_read=False)
    write(OUT/'extraction_result.json',result);print('BAYES_EXTRACTED',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    if args.probe:print(json.dumps(probe()))
    else:extract()
