"""Apply the predeclared full-data recipe without changing either fit function."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import argparse
import ast
import copy
import gc
import importlib.metadata
import json
import math
import random
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LOCALIZATION = ROOT / 'outputs/chronos_localization'
ADAPTATION = ROOT / 'outputs/chronos_patch_adaptation'
sys.path.insert(0, str(ADAPTATION))
import common
import data as data_module
import network
import adaptation
from common import Data, Sampler, BASE_MODEL, write, sha, state_sha


def extract(path, names, namespace, cuda=False):
    original = ast.parse(path.read_text())
    nodes = [copy.deepcopy(n) for n in original.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    class DeviceOnly(ast.NodeTransformer):
        changed = 0
        def visit_Constant(self, node):
            if cuda and node.value == 'mps':
                self.changed += 1
                return ast.copy_location(ast.Constant('cuda'), node)
            return node
    transform = DeviceOnly()
    module = transform.visit(ast.Module(body=nodes, type_ignores=[]))
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), namespace)
    return dict(source=str(path.relative_to(ROOT)), sha256=sha(path),
                functions=names, cuda_device_literals=transform.changed,
                fit_math_changed=False)


def configure(save):
    assert torch.cuda.is_available(), 'CUDA required; no CPU model fallback.'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((ROOT / 'portable_manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha(Path(__file__)) == protocol['run_source_sha256']
    assert torch.__version__ == protocol['torch_cuda_version']
    for name, version in protocol['packages'].items():
        assert importlib.metadata.version(name) == version, name
    BASE_MODEL.mkdir(parents=True, exist_ok=True)
    for name, digest in manifest['base_files_sha256'].items():
        target = BASE_MODEL / name
        if not target.exists():
            from huggingface_hub import hf_hub_download
            source = hf_hub_download('amazon/chronos-2', name,
                                    revision=manifest['base_revision'])
            shutil.copyfile(source, target)
        assert sha(target) == digest, name
    save.mkdir(exist_ok=True, parents=True)
    for name in ('protocol.json', 'authorization_and_plan.json', 'train_full_cuda.py'):
        target = save / name
        if target.exists():
            assert sha(target) == sha(OUT / name)
        else:
            shutil.copyfile(OUT / name, target)
    shutil.copyfile(ROOT / 'portable_manifest.json', save / 'portable_manifest.json')
    write(save / 'environment.json', dict(python=sys.version, torch=torch.__version__,
          gpu=torch.cuda.get_device_name(0), device='cuda', tf32=False,
          deterministic_algorithms=True, cpu_gpu_output_comparison=False,
          packages={n: importlib.metadata.version(n) for n in protocol['packages']}))
    stage1, stage2 = save / 'stage1', save / 'stage2'
    stage1.mkdir(exist_ok=True); stage2.mkdir(exist_ok=True)
    for directory in (stage1, stage2):
        shutil.copyfile(OUT / 'protocol.json', directory / 'protocol.json')
    (stage1 / 'ancestor').mkdir(exist_ok=True)
    initializer = stage1 / 'ancestor/fold-1_adapter.pt'
    source = ROOT / manifest['initializer']
    if not initializer.exists():
        shutil.copyfile(source, initializer)
    assert sha(initializer) == protocol['initializer_adapter_sha256']
    network.OUT = stage1
    adaptation.ORIGINAL = stage1
    data_module.OUT = stage1
    shared = dict(np=np, torch=torch, copy=copy, json=json, math=math, time=time,
                  Data=Data, Sampler=Sampler, write=write, sha=sha, state_sha=state_sha,
                  PatchHead=network.PatchHead, raw_features=network.raw_features,
                  objective=network.objective)
    first = dict(shared, OUT=stage1, WORK=ROOT / 'work/full_reference',
                 CONDITIONS=('point', 'localization'), STEPS=2000, SEED=920601)
    second = dict(shared, OUT=stage2, CONDITIONS=('head_only', 'tail_and_head'),
                  STEPS=1000, SEED=920901, SharedPrefix=adaptation.SharedPrefix,
                  Tail=adaptation.Tail, retained_head=adaptation.retained_head,
                  reference=adaptation.reference)
    audits = [extract(LOCALIZATION / 'experiment.py', ['reference_cache', 'fit'], first),
              extract(ADAPTATION / 'run.py', ['cpu_state', 'fit'], second, cuda=True)]
    assert audits[0]['cuda_device_literals'] == 0
    assert audits[1]['cuda_device_literals'] == 6
    write(save / 'recipe_ast_audit.json', audits)
    sources = save / 'recipe_sources'
    sources.mkdir(exist_ok=True)
    for folder, names in ((LOCALIZATION, ('data.py', 'network.py', 'util.py', 'experiment.py')),
                          (ADAPTATION, ('common.py', 'adaptation.py', 'run.py'))):
        for name in names:
            shutil.copyfile(folder / name, sources / (folder.name + '_' + name))
    return manifest, first, second


def fit_with_checkpoints(recipe, args, folder, final_step):
    original = torch.save
    retained = folder / 'checkpoints'
    retained.mkdir(exist_ok=True, parents=True)
    def save(obj, path, *a, **kw):
        target = Path(path)
        checkpoint = target.name == 'checkpoint.tmp.pt'
        if checkpoint:
            obj = dict(obj, torch_cpu_rng_state=torch.get_rng_state(),
                       torch_cuda_rng_states=torch.cuda.get_rng_state_all(),
                       python_rng_state=random.getstate())
        result = original(obj, path, *a, **kw)
        if checkpoint:
            step = int(obj['step'])
            dest = retained / f'step{step}.pt'
            if step != final_step:
                assert not dest.exists()
                shutil.copyfile(target, dest)
            write(retained / f'step{step}.json', dict(step=step, bytes=target.stat().st_size,
                  sha256=sha(target), rng_retained=True, final_step=final_step))
        return result
    torch.save = save
    try:
        return recipe['fit'](*args)
    finally:
        torch.save = original


@torch.inference_mode()
def native_check(data, prefix):
    ids = np.array([0, 1, 2, 3, 30, 254, 676, 732])
    history = torch.from_numpy(data.history(ids)).cuda()
    z, mask = prefix(history)
    tail = adaptation.Tail(prefix, False)
    actual = tail(z, mask, prefix.model.encoder)
    native = prefix.retained.patches(history)
    error = float((actual - native).abs().max().cpu())
    ref_error = float((adaptation.reference(actual, history)
                       - prefix.retained.reference(history)).abs().max().cpu())
    assert error == 0 and ref_error == 0
    return dict(passed=True, native_split_error=error, reference_error=ref_error,
                cpu_gpu_output_comparison=False)


def run(save):
    manifest, first, second = configure(save)
    assert not (save / 'complete.json').exists(), 'Completed weights are immutable.'
    data = Data()
    write(save / 'started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
          series=10000, held_series=0, stage1_updates=2000, stage2_updates=1000))
    first_folder = first['OUT'] / 'fold-1'
    if not (first_folder / 'training.json').exists():
        encoder = network.Encoder(-1, 'cuda')
        refs = first['reference_cache'](-1, data, encoder, 'cuda')
        heads = fit_with_checkpoints(first, (-1, data, encoder, refs, 'cuda'), first_folder, 2000)
        del encoder, refs, heads
        gc.collect(); torch.cuda.empty_cache()
    training = json.loads((first_folder / 'training.json').read_text())
    assert training['steps_per_condition'] == 2000
    assert training['train_series'] == 10000 and training['held_series'] == 0
    prefix = adaptation.SharedPrefix(-1, 'cuda')
    write(save / 'native_check.json', native_check(data, prefix))
    folder = second['OUT'] / 'fold-1'
    heads, tails = fit_with_checkpoints(second, (-1, data, prefix), folder, 1000)
    split = np.load(folder / 'split.npz')
    assert np.array_equal(split['train'], np.arange(10000)) and len(split['held']) == 0
    final = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert final['step'] == 1000
    for i, condition in enumerate(second['CONDITIONS']):
        model = torch.load(folder / condition / 'model.pt', map_location='cpu', weights_only=True)
        for key in ('head', 'tail'):
            assert all(torch.equal(v, final[key + 's'][i][k]) for k, v in model[key].items())
    del final, model
    write(save / 'complete.json', dict(completed=True, series=10000, held_series=0,
          stage1_updates=2000, stage2_updates=1000, device='cuda',
          candidate='stage2/fold-1/tail_and_head/model.pt',
          candidate_sha256=sha(folder / 'tail_and_head/model.pt'),
          final_checkpoint_matches_models=True, quality_scored=False,
          cpu_gpu_output_comparison=False, at_utc=datetime.now(timezone.utc).isoformat()))
    recovery = save / 'recovery'; recovery.mkdir(exist_ok=True)
    archive = recovery / 'Chronos_full_training.zip'
    files = [p for p in save.rglob('*') if p.is_file() and recovery not in p.parents
             and 'ancestor' not in p.parts and p.name != 'checkpoint.pt'
             and not (p.parent.name == 'checkpoints' and p.suffix == '.pt')]
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in files:
            z.write(p, str(p.relative_to(save)))
    write(recovery / 'manifest.json', dict(completed=True, bytes=archive.stat().st_size,
          sha256=sha(archive), files_sha256={str(p.relative_to(save)): sha(p) for p in files}))
    print('CHRONOS_FULL_TRAINING_COMPLETE', json.dumps(json.loads((save / 'complete.json').read_text())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=Path, required=True)
    run(parser.parse_args().save)
