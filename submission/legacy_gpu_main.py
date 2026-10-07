"""Public-best V204 plus adaptive-variance and GPU-trained temporal corrections.
Training recipe and scripts: training_source/README.md and gpu_training_source/README.md.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
from contextvars import ContextVar
import sys

_CORE_SINK=ContextVar('conditional_bayes_v199_core_sink',default=None)

def _install_core_tap():
    global _CORE_SINK
    runtime=sys.modules['base.v738_compact_stack_runtime']
    cls=runtime.CompactMeta
    if getattr(cls.predict,'_conditional_bayes_v199',False):
        _CORE_SINK=cls.predict._conditional_bayes_v199_sink
        return
    original=cls.predict
    def observed(self,probabilities):
        value=original(self,probabilities)
        sink=_CORE_SINK.get()
        if sink is not None:
            assert sink['pending'] and sink['core'] is None,'Core score does not match the current online observation.'
            sink['core']=float(value)
        return value
    observed._conditional_bayes_v199=True
    observed._conditional_bayes_v199_sink=_CORE_SINK
    cls.predict=observed

def train(datasets,model_directory_path):
    import selected64_main as parent
    from gpu_temporal_runtime import Resources
    parent.train(datasets,model_directory_path);Resources(Path(model_directory_path)/'conditional_bayes')

def infer(datasets,model_directory_path):
    import selected64_main as parent
    from gpu_temporal_runtime import Resources,State
    resources=Resources(Path(model_directory_path)/'conditional_bayes')
    slot={'state':None,'pending':False,'core':None}
    def proxy():
        for history,online in datasets:
            assert not slot['pending'];state=State(history,resources);slot['state']=state
            def points(source=online,current=state):
                for point in source:
                    assert not slot['pending'],'Parent consumed future data before producing the current score.'
                    current.update(point);slot['pending']=True;slot['core']=None;yield point
                assert not slot['pending'],'Parent omitted the final score.'
            yield history,points()
    stream=iter(parent.infer(proxy(),model_directory_path))
    ready=next(stream);assert not slot['pending'];_install_core_tap();yield ready
    try:
        while True:
            token=_CORE_SINK.set(slot)
            try:
                value=next(stream)
            except StopIteration:
                break
            finally:
                _CORE_SINK.reset(token)
            assert slot['pending'] and slot['core'] is not None
            result=slot['state'].apply(float(value),slot['core'])
            slot['pending']=False;slot['core']=None;yield result
        assert not slot['pending']
    finally:
        stream.close()

INFER_PARALLELISM=32
