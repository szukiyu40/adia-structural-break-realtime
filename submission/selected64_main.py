"""Blend submission60 with the unchanged V120 tree and selected normal correction."""
import os
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

def settings(root):
    import json
    from normal_calibration_v124 import COEFFICIENTS
    recipe = json.loads((root / 'encoder_blend.json').read_text())
    weight, coefficient = float(recipe['weight']), float(recipe['normal_calibration_coefficient'])
    assert weight == .125 and coefficient in COEFFICIENTS
    return weight, coefficient

def train(datasets, model_directory_path):
    import selected60_main as parent
    from encoder_submission_runtime_v108 import EncoderTree
    root = Path(model_directory_path)
    parent.train(datasets, str(root)); EncoderTree(root / 'encoder'); settings(root)

def infer(datasets, model_directory_path):
    import selected60_main as parent
    from encoder_submission_runtime_v108 import EncoderTree
    from normal_calibration_v124 import recalibrate
    root = Path(model_directory_path); weight, coefficient = settings(root)
    detector = EncoderTree(root / 'encoder'); slot = {'pending': False, 'probability': None}
    def proxy():
        for history, online in datasets:
            assert not slot['pending']; detector.begin(history)
            def points(source=online):
                for point in source:
                    assert not slot['pending'], 'Selected parent read ahead of its emitted score.'
                    original = detector.consume(point)
                    slot['probability'] = float(recalibrate(original, detector.normal_q90, coefficient))
                    slot['pending'] = True; yield point
                assert not slot['pending'], 'Selected parent omitted the final score.'
            yield history, points()
    stream = iter(parent.infer(proxy(), str(root)))
    ready = next(stream); assert not slot['pending']; yield ready
    for value in stream:
        assert slot['pending'], 'Selected parent emitted an unmatched score.'
        combined = (1-weight)*float(value)+weight*slot['probability']
        slot['pending'] = False; yield combined
    assert not slot['pending']

INFER_PARALLELISM = 16
