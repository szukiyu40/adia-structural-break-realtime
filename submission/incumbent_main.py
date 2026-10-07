"""Causal ensemble: 87.5% baseline and 12.5% Colab full-fit Chronos-2."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path


def train(datasets, model_directory_path):
    import legacy_gpu_main as parent
    from chronos_adapter_runtime import Resources
    parent.train(datasets, model_directory_path)
    Resources(Path(model_directory_path)/'chronos_adapter')


def infer(datasets, model_directory_path):
    import legacy_gpu_main as parent
    from chronos_adapter_runtime import Resources, State
    resources = Resources(Path(model_directory_path)/'chronos_adapter')
    slot = {'pending':False,'probability':None}

    def proxy():
        for history, online in datasets:
            assert not slot['pending']
            state = State(history, resources)
            def points(source=online, current=state):
                for point in source:
                    assert not slot['pending'], 'Parent read ahead of its current output.'
                    slot['probability'] = current.update(point)
                    slot['pending'] = True
                    yield point
                assert not slot['pending'], 'Parent omitted the final prediction.'
            yield history, points()

    stream = iter(parent.infer(proxy(), model_directory_path))
    try:
        ready = next(stream)
        assert not slot['pending']
        yield ready
        for value in stream:
            assert slot['pending'], 'Unmatched parent prediction.'
            result = .875*float(value)+.125*slot['probability']
            slot['pending'] = False
            yield result
        assert not slot['pending']
    finally:
        stream.close()


INFER_PARALLELISM = 32
