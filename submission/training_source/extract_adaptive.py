# Publication change: private paths use configurable ADIA_* locations.
import os
import sys
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
OLD=Path(os.environ.get('ADIA_LEGACY_ROOT', str(Path.cwd() / 'external-artifacts/legacy')))
sys.path.insert(0,str(OLD/'work'))
import numpy as np
from scipy.signal import lfilter
from discord_bayes_v198 import prepare_channels,sha
from run_time_rcd_online_v197 import series_stream
OUT=Path(__file__).resolve().parent
LAGS=[1,4]; WINDOWS=[32,128,512]

def compute(h,o):
 h=np.asarray(h,float);o=np.asarray(o,float)
 _,_,ctx=prepare_channels(h, o[:1]);coef=ctx[20:30]
 hz=np.clip((h-h.mean())/max(h.std(),1e-8),-8,8)
 oz=np.clip((o-h.mean())/max(h.std(),1e-8),-8,8)
 er=lfilter(np.r_[1.,-coef],[1.],np.r_[hz,oz]);eh=er[10:len(h)]
 er=np.clip((er-eh.mean())/max(eh.std(),1e-8),-8,8)
 out=[]
 for v,n in [(np.r_[hz,oz],len(h)),(er[10:],len(h)-10)]:
  for decay in [.95,.99,.995]:
   # The denominator for point j uses only values strictly before j.
   q=lfilter([1-decay],[1,-decay],v*v,zi=[decay])[0]
   denom=np.maximum(np.r_[1.,q[:-1]],.05)
   z=v/np.sqrt(denom)
   for transformed in [z,np.minimum(z*z,64)]:
    c=np.r_[0,np.cumsum(transformed)]
    for w in [16,64,256]:
     avg=(c[w:]-c[:-w])/w;hist=avg[:n-w+1]
     score=np.clip((avg[n-w+1:]-hist.mean())/max(hist.std(),.02),-10,10)
     out.extend([score,abs(score)])
 return np.stack(out,axis=1).astype('f4')

if __name__=='__main__':
 start=time.time();lengths=np.load(OLD/'outputs/models/full_parent_scope_v179/v35/online_lengths.npy');off=np.r_[0,np.cumsum((lengths+3)//4)]
 f=np.lib.format.open_memmap(OUT/'adaptive_variance.npy',mode='w+',dtype='f4',shape=(int(off[-1]),72));checks=0
 for item in series_stream(OLD/'work/structural-break-real-time-dominant-wolf/data/X_train.parquet'):
  sid=int(item['id'][0]);h=item['value'][item['period']==1];o=item['value'][item['period']==2];z=compute(h,o);assert len(z)==len(o);f[off[sid]:off[sid+1]]=z[::4]
  if sid in [0,1,6,128,4096,9999]:
   for k in [min(17,len(o)),min(65,len(o))]:assert np.array_equal(z[:k],compute(h,o[:k]));checks+=1
  if sid%2000==0:print('series',sid,'seconds',time.time()-start,flush=True)
 f.flush();r=dict(completed=True,series=10000,prefix_checks_exact=checks,features_sha256=sha(OUT/'adaptive_variance.npy'),source_sha256=sha(__file__),seconds=time.time()-start)
 (OUT/'adaptive_extraction.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
