"""Checkpoint-aware orchestration; the frozen experiment stages stay unchanged.

Use instead of run.py after interruption. Only the current unfinished fold is
resumed; the original two-nonpositive-fold stop gate is retained. Contradictory
or corrupt checkpoints fail explicitly and are never deleted or replaced by
this wrapper. --self-test uses temporary fake checkpoints and mocked dispatch.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import torch
import integration_common as c
import run as frozen_run


class CheckpointError(RuntimeError):
    pass


def expected_fit_ids(fold, inner):
    return [sid for sid in range(10000) if sid % 5 != fold and (sid // 5) % 2 == inner]


def inspect_checkpoint(path, fold, inner, steps=800):
    """Read-only consistency check, including full completion SHA and exact step."""
    path = Path(path)
    complete_path = path.parent / 'complete.json'
    if steps != 800:
        raise CheckpointError('The frozen experiment requires exactly 800 predictor steps.')
    if not path.exists():
        if complete_path.exists():
            raise CheckpointError(f'Completion metadata exists but checkpoint is missing: {path}')
        return {'state': 'missing', 'path': str(path), 'step': 0}
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        required = {'model', 'optimizer', 'numpy_rng', 'torch_rng', 'step', 'steps', 'fit_ids', 'losses'}
        if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
            raise ValueError('Missing training-state fields.')
        step = checkpoint['step']
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= steps:
            raise ValueError('Invalid optimizer step.')
        if checkpoint['steps'] != steps or checkpoint['fit_ids'] != expected_fit_ids(fold, inner):
            raise ValueError('Step budget or predictor fit IDs differ from the frozen protocol.')
        if len(checkpoint['losses']) != step:
            raise ValueError('Loss-history length does not match the saved optimizer step.')
        digest = c.sha(path)
        if complete_path.exists():
            complete = c.read(complete_path)
            if complete.get('steps') != steps or step != steps:
                raise ValueError('Completion metadata requires exactly 800 completed steps.')
            if complete.get('checkpoint_sha256') != digest:
                raise ValueError('complete.json SHA does not match latest.pt.')
            state = 'complete'
        else:
            state = 'unmarked_complete' if step == steps else 'partial'
        return {'state': state, 'path': str(path), 'step': step, 'sha256': digest}
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f'Invalid predictor state at {path}: {exc}. Originals preserved.') from exc


def frozen_train(fold, inner, path):
    # Same frozen function, data cache, 800 steps, and batch 64 as the protocol.
    code = ('import sys; from pathlib import Path; '
            f'sys.path.insert(0,{str(c.OUT / "predictor_source")!r}); '
            'import common; '
            f'common.CACHE=Path({str(c.ARRAYS)!r}); '
            'import train_predictor; '
            f'train_predictor.train({fold},{inner},steps=800,batch=64,out={str(path.parent)!r})')
    subprocess.run([str(c.PYTHON), '-u', '-c', code], check=True)


def ensure_predictor(path, fold, inner, record=lambda event: None, train_fn=frozen_train):
    before = inspect_checkpoint(path, fold, inner)
    record(dict(event='predictor_checked', fold=fold, inner=inner, **before))
    if before['state'] == 'complete':
        return before
    record({'event': 'predictor_resume_or_train', 'fold': fold, 'inner': inner,
            'from_step': before['step'], 'to_step': 800})
    # A complete checkpoint with absent complete.json is also sent through the
    # original trainer: it runs zero optimizer updates and writes its metadata.
    train_fn(fold, inner, Path(path))
    after = inspect_checkpoint(path, fold, inner)
    if after['state'] != 'complete':
        raise CheckpointError(f'Frozen trainer returned without a verified completed checkpoint: {path}')
    record(dict(event='predictor_verified_after_train', fold=fold, inner=inner, **after))
    return after


def campaign(out, work, call_stage, ensure_current, record=lambda event: None):
    """Original run.py order and gate, with one safe predictor dispatch boundary."""
    call_stage('prepare.py')
    call_stage('verify.py')
    results = []
    for fold in range(5):
        result_path = out / f'fold{fold}' / 'result.json'
        if not result_path.exists():
            result_path.parent.mkdir(parents=True, exist_ok=True)
            for inner in (0, 1):
                ensure_current(fold, inner)
                call_stage('extract.py', '--fold', fold, '--inner', inner)
            call_stage('fit.py', '--fold', fold)
        results.append(c.read(result_path))
        # Same reproducible-feature cleanup as frozen run.py, after saved result.
        scratch = work / f'fold{fold}'
        if scratch.exists():
            shutil.rmtree(scratch)
        record({'event': 'fold_complete', 'fold': fold,
                'primary_delta': results[-1]['primary_delta']})
        if sum(r['primary_delta'] <= 0 for r in results) >= 2:
            c.write(out / 'screen.json', {'status': 'stopped',
                'reason': '4/5 positive folds impossible under fixed gate',
                'completed_folds': len(results)})
            record({'event': 'screen_stopped', 'completed_folds': len(results)})
            break
    call_stage('report.py')


def main():
    # A pre-existing frozen protocol is required; this wrapper cannot create or
    # change the experiment protocol or silently re-snapshot its dependencies.
    protocol = c.ensure_protocol()
    if protocol['neural_fit_steps'] != 800 or protocol['neural_fit_batch'] != 64:
        raise CheckpointError('Unexpected frozen predictor training configuration.')
    for name, expected in protocol['predictor_sources'].items():
        if c.sha(c.OUT / 'predictor_source' / name) != expected:
            raise CheckpointError(f'Frozen predictor source changed: {name}')
    record_path = c.WORK / 'resume_invocations' / f'{time.time_ns()}_{os.getpid()}.json'
    audit = {'wrapper_sha256': c.sha(Path(__file__)),
             'frozen_protocol_sha256': c.sha(c.OUT / 'protocol.json'),
             'frozen_experiment_sources': protocol['experiment_sources'],
             'frozen_predictor_sources': protocol['predictor_sources'],
             'python': sys.version, 'started_unix': time.time(), 'events': [], 'status': 'running'}
    def record(event):
        audit['events'].append(dict(event, unix_time=time.time()))
        c.write(record_path, audit)
    def ensure_current(fold, inner):
        path = c.predictor_path(fold, inner)
        reused = str(path.parent.relative_to(c.OUT))
        if reused in protocol['reused_predictors']:
            if not path.exists() or c.sha(path) != protocol['reused_predictors'][reused]:
                raise CheckpointError(f'Reused predictor identity changed or missing: {path}. Originals preserved.')
        return ensure_predictor(path, fold, inner, record)
    record({'event': 'start'})
    try:
        campaign(c.OUT, c.WORK, frozen_run.call, ensure_current, record)
        audit['status'] = 'complete'
        record({'event': 'finished'})
    except Exception as exc:
        audit['status'] = 'failed'
        record({'event': 'failed', 'error': str(exc)})
        raise


def self_test():
    """No real predictor fitting, feature extraction, input loading, or campaign."""
    checks = []
    with tempfile.TemporaryDirectory(prefix='resume_check_', dir=c.ROOT / 'work') as temporary:
        base = Path(temporary)
        path = base / 'predictor/latest.pt'
        def checkpoint(step, complete=False):
            path.parent.mkdir(parents=True, exist_ok=True)
            ck = {'model': {'fake': torch.zeros(1)}, 'optimizer': {}, 'numpy_rng': {},
                  'torch_rng': torch.get_rng_state(), 'step': step, 'steps': 800,
                  'fit_ids': expected_fit_ids(2, 0), 'losses': [1.] * step}
            torch.save(ck, path)
            marker = path.parent / 'complete.json'
            if marker.exists():
                marker.unlink()  # Only these temporary test fixtures are changed.
            if complete:
                c.write(marker, {'steps': 800, 'checkpoint_sha256': c.sha(path)})
        def rejects(fn):
            try:
                fn()
            except CheckpointError:
                return
            raise AssertionError('Invalid state was accepted.')
        assert inspect_checkpoint(path, 2, 0)['state'] == 'missing'
        checkpoint(200)
        assert inspect_checkpoint(path, 2, 0)['state'] == 'partial'
        calls = []
        def fake_train(fold, inner, target):
            calls.append((fold, inner, str(target)))
            checkpoint(800, complete=True)
        ensure_predictor(path, 2, 0, train_fn=fake_train)
        assert len(calls) == 1
        complete_hash = c.sha(path.parent / 'complete.json')
        ensure_predictor(path, 2, 0, train_fn=fake_train)
        assert len(calls) == 1 and complete_hash == c.sha(path.parent / 'complete.json')
        checks.append('partial_checkpoint_resumed_once_complete_checkpoint_skipped')
        checkpoint(800)
        assert inspect_checkpoint(path, 2, 0)['state'] == 'unmarked_complete'
        ensure_predictor(path, 2, 0, train_fn=fake_train)
        assert len(calls) == 2
        checks.append('completed_unmarked_checkpoint_finalized_via_frozen_trainer')
        c.write(path.parent / 'complete.json', {'steps': 800, 'checkpoint_sha256': 'wrong'})
        original = path.read_bytes()
        rejects(lambda: inspect_checkpoint(path, 2, 0))
        assert path.read_bytes() == original
        checkpoint(200, complete=True)
        rejects(lambda: inspect_checkpoint(path, 2, 0))
        checkpoint(800, complete=True)
        path.unlink()
        rejects(lambda: inspect_checkpoint(path, 2, 0))
        path.write_bytes(b'not a torch checkpoint')
        rejects(lambda: inspect_checkpoint(path, 2, 0))
        checkpoint(800, complete=True)
        rejects(lambda: inspect_checkpoint(path, 3, 0))
        checks.append('mismatched_sha_partial_completion_missing_corrupt_wrong_split_rejected')
        out, work = base / 'out', base / 'scratch'
        for fold, delta in [(0, .01), (1, -.01)]:
            c.write(out / f'fold{fold}/result.json', {'primary_delta': delta})
        dispatch, trained = [], []
        def stages(name, *args):
            dispatch.append((name, args))
            if name == 'fit.py':
                assert args == ('--fold', 2)
                c.write(out / 'fold2/result.json', {'primary_delta': 0.})
        campaign(out, work, stages, lambda f, i: trained.append((f, i)))
        assert trained == [(2, 0), (2, 1)]
        assert not (out / 'fold3').exists()
        assert c.read(out / 'screen.json')['completed_folds'] == 3
        assert [name for name, _ in dispatch] == ['prepare.py', 'verify.py', 'extract.py', 'extract.py', 'fit.py', 'report.py']
        trained.clear()
        c.write(out / 'fold0/result.json', {'primary_delta': 0.})
        campaign(out, work, stages, lambda f, i: trained.append((f, i)))
        assert trained == [] and c.read(out / 'screen.json')['completed_folds'] == 2
        checks.append('fixed_gate_preserved_no_future_fold_pretraining_or_dispatch')
    return {'passed': True, 'real_training': False, 'checks': checks,
            'wrapper_sha256': c.sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        main()
