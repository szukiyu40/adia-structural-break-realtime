import numpy as np
from scipy.signal import lfilter
from discord_bayes_features_v198 import prepare_channels

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

