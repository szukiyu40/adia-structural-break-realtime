"""Causal incumbent plus the recorded feature-integration candidate."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import json
import math
from pathlib import Path


def _resources(model_directory_path):
    from feature_serving import Resources
    root = Path(model_directory_path) / 'feature_integration'
    metadata = json.loads((root / 'metadata.json').read_text())
    resources = Resources(root, expected_hashes=metadata['model_sha256'])
    config = metadata['serving']
    assert config['family'] in ('probability_blend', 'logit_contrast')
    assert math.isfinite(config['coefficient'])
    assert 0 <= config['coefficient'] <= (0.5 if config['family'] == 'probability_blend' else 1.)
    assert config['epsilon'] == 1e-7 and config['delta_update_stride'] == 4
    if config['family'] == 'logit_contrast':
        assert resources.control_heads
    return resources, config


def blend(incumbent, control, augmented, config):
    coefficient = config['coefficient']
    if config['family'] == 'probability_blend':
        return (1.-coefficient)*incumbent + coefficient*augmented
    epsilon = config['epsilon']
    def logit(probability):
        p = min(max(probability, epsilon), 1.-epsilon)
        return math.log(p) - math.log1p(-p)
    z = logit(incumbent) + coefficient*(logit(augmented)-logit(control))
    if z >= 0:
        return 1./(1.+math.exp(-z))
    ez = math.exp(z)
    return ez/(1.+ez)


def train(datasets, model_directory_path):
    import incumbent_main as parent
    parent.train(datasets, model_directory_path)
    _resources(model_directory_path)


def infer(datasets, model_directory_path):
    import incumbent_main as parent
    from feature_serving import State
    resources, config = _resources(model_directory_path)
    slot = {'pending': False, 'pair': None}

    def proxy():
        for history, online in datasets:
            assert not slot['pending']
            state = State(history, resources)
            def points(source=online, current=state):
                for point in source:
                    assert not slot['pending'], 'Parent consumed the next observation before yielding.'
                    slot['pair'] = current.update_pair(point)
                    slot['pending'] = True
                    yield point
                assert not slot['pending'], 'Parent omitted its last prediction.'
            yield history, points()

    stream = iter(parent.infer(proxy(), model_directory_path))
    try:
        ready = next(stream)
        assert not slot['pending']
        yield ready
        for value in stream:
            assert slot['pending'], 'Parent output has no matching observation.'
            control, augmented = slot['pair']
            result = blend(float(value), control, augmented, config)
            assert math.isfinite(result) and 0 <= result <= 1
            slot['pending'] = False
            yield result
        assert not slot['pending']
    finally:
        stream.close()


INFER_PARALLELISM = 30
