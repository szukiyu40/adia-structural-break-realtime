import argparse,json,time
import numpy as np
import torch
from features import FeatureStream
from inference import StreamingDetector

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);args=ap.parse_args()
    rng=np.random.default_rng(428)
    h=rng.standard_t(5,1500);x=rng.normal(size=300);x[90:]*=1.5
    m=StreamingDetector(args.checkpoint);m.begin(h);start=time.perf_counter()
    actual=np.array([m.consume(v) for v in x]);elapsed=time.perf_counter()-start
    m.begin(h);repeat=np.array([m.consume(v) for v in x])
    m.begin(rng.normal(size=1100));[m.consume(v) for v in rng.normal(size=80)]
    m.begin(h);after_other=np.array([m.consume(v) for v in x])
    altered=x.copy();altered[150:]+=10
    m.begin(h);future_changed=np.array([m.consume(v) for v in altered])
    assert np.array_equal(actual,repeat)
    assert np.array_equal(actual,after_other)
    assert np.array_equal(actual[:150],future_changed[:150])
    f=FeatureStream(h);batch=f.batch(x)
    with torch.inference_mode():
        logits,_=m.model(torch.from_numpy(batch)[None],torch.from_numpy(f.ctx)[None])
        batch_pred=torch.sigmoid(logits[0]).numpy()
    delta=float(np.max(np.abs(actual-batch_pred)))
    assert delta<2e-6,delta
    assert np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all()
    result={'same_series_repeat_exact':True,'independent_of_other_series':True,
      'future_suffix_does_not_change_prefix':True,'streaming_vs_batch_max_delta':delta,
      'seconds_per_300_points':elapsed,'estimated_5m_point_hours_single_process':elapsed/300*5e6/3600}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
