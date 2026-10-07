"""Selected official model blended with an independent NLinear neural model."""
import os
from pathlib import Path
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'

def train(datasets,model_directory_path):
    import json,hashlib
    from baseline import main as parent
    path=Path(model_directory_path);parent.train(datasets,str(path/'baseline'))
    neural=path/'neural';manifest=json.loads((neural/'manifest.json').read_text())
    for item in manifest['models']:
        assert hashlib.sha256((neural/item['file']).read_bytes()).hexdigest()==item['sha256']
        if item.get('evidence_gru'):
            assert hashlib.sha256((neural/item['evidence_gru']).read_bytes()).hexdigest()==item['evidence_gru_sha256']
    recipe=json.loads((path/'blend.json').read_text())
    for item in recipe.get('correction_resources',[]):
        assert hashlib.sha256((path/item['file']).read_bytes()).hexdigest()==item['sha256']

def infer(datasets,model_directory_path):
    import json
    from nlinear_detector.runtime import Ensemble
    from baseline import main as parent
    root=Path(model_directory_path);recipe=json.loads((root/'blend.json').read_text());weight=float(recipe['neural_weight'])
    detector=Ensemble(root/'neural');slot={'pending':False,'probability':None,'index':0}
    correction=None
    if recipe.get('correction'):
        from blend_runtime import BlendCorrection
        correction=BlendCorrection(root/recipe['correction'],root/recipe['correction_metadata'])
    def proxy():
        for history,online in datasets:
            assert not slot['pending'];detector.begin(history);slot['index']=0
            if correction is not None:correction.begin(detector.features.ctx)
            def one_point(source=online):
                for point in source:
                    assert not slot['pending'],'Parent consumed a future point before yielding its score'
                    slot['probability']=detector.consume(point);slot['pending']=True
                    yield point
                assert not slot['pending'],'Parent omitted the final score'
            yield history,one_point()
    stream=iter(parent.infer(proxy(),str(root/'baseline')));ready=next(stream)
    assert not slot['pending'];yield ready
    for value in stream:
        assert slot['pending'],'Parent yielded a score without an online point'
        current_weight=weight
        if recipe.get('late_neural_weight') is not None:
            t=slot['index'];current_weight=weight+(float(recipe['late_neural_weight'])-weight)*t/(t+float(recipe.get('transition',128)))
        combined=(1-current_weight)*float(value)+current_weight*slot['probability']
        if correction is not None:combined=correction.consume(float(value),slot['probability'],slot['index'])
        slot['pending']=False;slot['index']+=1
        yield combined
    assert not slot['pending']

INFER_PARALLELISM=16
