"""Conditional full-data fit; frozen CV sources are imported, never modified.

Both authenticated five-fold bootstrap gates must pass before any directory,
checkpoint, cache, or model is created. No serving bundle or submission is made.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import integration_common as c
import numpy as np

STEPS, BATCH, SCRATCH_LIMIT = 800, 64, 900_000_000


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_gate_fields(primary, ensemble):
    for name, result, flag in [('primary', primary, 'passes_frozen_primary_adoption_gate'),
                               ('incumbent', ensemble, 'passes_frozen_incumbent_adoption_gate')]:
        require(result.get('status') == 'complete' and result.get('series') == 10000,
                f'{name}: a completed 10,000-series bootstrap is required.')
        require(result.get(flag) is True and result.get('bootstrap_95_lower_positive') is True
                and result.get('ci_95', [-1])[0] > 0, f'{name}: bootstrap gate has not passed.')
        require(result['gate']['eligible'] is True and result['gate']['completed_folds'] == list(range(5)),
                f'{name}: all five eligible folds are required.')
        require(result.get('replicates') == 300 and result.get('seed') == 919703,
                f'{name}: bootstrap configuration changed.')
    require(primary['point_delta'] >= .001 and primary['gate']['positive_folds'] >= 4,
            'Primary point gate failed.')
    require(ensemble.get('fixed_blend_weight') == .1 and ensemble.get('inputs_authenticated') is True,
            'Incumbent comparison must authenticate the fixed 10% blend.')
    point = ensemble['verified_point_gate']
    require(point['eligible'] is True and point['point_gain'] >= .001 and point['positive_folds'] >= 4,
            'Incumbent recomputed point gate failed.')
    require(ensemble.get('passes_incumbent_blend_gate') is True and
            ensemble.get('eligible_for_fullfit_and_runtime_work') is True,
            'Incumbent full-fit eligibility is absent.')


def authenticate_gates():
    protocol = c.ensure_protocol()
    paths = {name: c.OUT / f'{name}.json' for name in ('bootstrap', 'ensemble_bootstrap')}
    require(all(p.exists() for p in paths.values()), 'Both bootstrap result files must exist; no full fit was started.')
    primary, ensemble = (c.read(paths[n]) for n in ('bootstrap', 'ensemble_bootstrap'))
    check_gate_fields(primary, ensemble)
    require(primary['frozen_protocol_sha256'] == c.sha(c.OUT / 'protocol.json'), 'Primary protocol SHA changed.')
    require(primary['implementation_sha256'] == c.sha(c.OUT / 'bootstrap.py'), 'Primary bootstrap source SHA changed.')
    require(primary['checks_sha256'] == c.sha(c.OUT / 'checks.json'), 'Primary validation record SHA changed.')
    require(primary['gate']['combined_report_sha256'] == c.sha(c.OUT / 'result.json'), 'Combined report changed.')
    require(set(ensemble['source_sha256']) == {'ensemble_bootstrap.py', 'bootstrap.py', 'integration_common.py', 'protocol.json'},
            'Unrecognized incumbent bootstrap source contract.')
    for name, digest in ensemble['source_sha256'].items():
        require(c.sha(c.OUT / name) == digest, f'Incumbent source SHA changed: {name}')
    for result in (primary, ensemble):
        require(bool(result['inputs_sha256']), 'Missing bootstrap input provenance.')
        for name, digest in result['inputs_sha256'].items():
            path = Path(name) if Path(name).is_absolute() else c.OUT / name
            require(c.sha(path) == digest, f'Bootstrap input SHA changed: {name}')
    for name, digest in ensemble['gate']['report_sources_sha256'].items():
        require(c.sha(c.OUT / name) == digest, f'Incumbent report SHA changed: {name}')
    for name, digest in protocol['predictor_sources'].items():
        require(c.sha(c.OUT / 'predictor_source' / name) == digest, f'Predictor source changed: {name}')
    require(c.sha(c.RECIPE) == protocol['original_head_recipe_sha256'], 'Original tree recipe changed.')
    require(c.sha(c.MANIFEST) == protocol['feature_manifest_sha256'], 'Original feature manifest changed.')
    return protocol, {p.name: c.sha(p) for p in paths.values()}


def fit_ids(inner):
    ids = np.arange(10000)
    return ids[(ids // 5) % 2 == inner]


def rng_state(rng):
    import torch
    state = {'numpy': rng.bit_generator.state, 'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.backends.mps.is_available(): state['mps'] = torch.mps.get_rng_state()
    if torch.cuda.is_available(): state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, rng):
    import torch
    rng.bit_generator.state = state['numpy']; random.setstate(state['python']); torch.set_rng_state(state['torch'].cpu())
    if 'mps' in state: torch.mps.set_rng_state(state['mps'].cpu())
    if 'cuda' in state: torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def atomic_torch(path, state):
    import torch
    tmp = path.with_suffix('.tmp.pt'); torch.save(state, tmp); tmp.replace(path)


def train_predictor(inner, out, data, network, manifest_sha, device):
    """Derivative of the frozen trainer: only fit IDs/seed scope are expanded."""
    import torch
    out.mkdir(parents=True, exist_ok=True)
    ids = fit_ids(inner); require(len(ids) == 5000, 'Full-fit history partition must have 5,000 series.')
    seed = sys.modules['common'].SEED + 50 + inner  # Reserve fold-index 5 for the full fit.
    torch.set_num_threads(2); torch.manual_seed(seed); random.seed(seed)
    rng = np.random.default_rng(seed)
    model = network.Predictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    checkpoint, complete = out / 'latest.pt', out / 'complete.json'
    step0, losses = 0, []
    if complete.exists(): require(checkpoint.exists(), 'Predictor complete.json has no checkpoint.')
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(state['manifest_sha256'] == manifest_sha and state['fit_ids'] == ids.tolist()
                and state['steps'] == STEPS and 0 <= state['step'] <= STEPS,
                'Predictor checkpoint provenance or step count differs.')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], rng); step0, losses = state['step'], state['losses']
        require(len(losses) == step0, 'Predictor checkpoint loss history differs.')
        if complete.exists():
            require(step0 == STEPS and c.read(complete)['checkpoint_sha256'] == c.sha(checkpoint),
                    'Predictor completion SHA/step differs; originals preserved.')
            return checkpoint
    start = time.time(); model.train()
    for step in range(step0, STEPS):
        sid = rng.choice(ids, BATCH)
        h = data['hist_lengths'][sid]
        begin = h//2 + rng.integers(0, 2**30, BATCH) % (h-h//2-193)
        ix = begin[:, None] + np.arange(193)[None, :]
        require(np.all(ix[:, -1] < h), 'Training window escaped the known history.')
        v, r = (np.asarray(data[name][sid[:, None], ix]) for name in ('values', 'innovations'))
        x = torch.from_numpy(network.inputs(v[:, :-1], r[:, :-1])).to(device)
        static = torch.from_numpy(np.array(data['static'][sid])).to(device)
        target = torch.from_numpy(np.clip(r[:, 1:], -25, 25)).to(device)
        optimizer.zero_grad(set_to_none=True)
        q = model(x, static); loss = model.loss(q[:, :, 64:], target[:, 64:])
        require(bool(torch.isfinite(loss)), 'Nonfinite predictor loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step+1) % 100 == 0 or step+1 == STEPS:
            state = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     'optimizer': optimizer.state_dict(), 'rng': rng_state(rng), 'step': step+1,
                     'steps': STEPS, 'fit_ids': ids.tolist(), 'losses': losses, 'manifest_sha256': manifest_sha}
            atomic_torch(checkpoint, state)
            if (step+1) % 200 == 0: atomic_torch(out / f'step{step+1}.pt', state)
            c.write(out / 'progress.json', {'inner': inner, 'step': step+1, 'loss_recent': float(np.mean(losses[-100:])),
                                          'seconds_this_session': time.time()-start})
            print(json.dumps(c.read(out / 'progress.json')), flush=True)
    c.write(complete, {'steps': STEPS, 'fit_series': 5000, 'seed': seed, 'checkpoint_sha256': c.sha(checkpoint),
                      'manifest_sha256': manifest_sha, 'seconds_this_session': time.time()-start})
    return checkpoint


def extract_neural(scratch, checkpoints, data, ids, ticks, network, features, manifest_sha, device):
    """One 144-column cache; each series is processed by the opposite predictor."""
    import torch
    counts = np.bincount(ids, minlength=10000); offsets = np.r_[0, counts.cumsum()]
    required = len(ids)*144*4 + 10_000_000
    require(required < SCRATCH_LIMIT, 'Neural scratch exceeds the reserved <1 GB limit.')
    scratch.mkdir(parents=True, exist_ok=True); path = scratch / 'neural.npy'
    if not path.exists():
        require(not any(scratch.glob('progress*.json')), 'Missing partial feature cache; originals preserved.')
        require(shutil.disk_usage(scratch).free > required + 256_000_000, 'Insufficient free disk for bounded neural scratch.')
        matrix = np.lib.format.open_memmap(path, mode='w+', dtype='f4', shape=(len(ids), 144))
        matrix[:] = np.nan; matrix.flush()
    else:
        matrix = np.load(path, mmap_mode='r+')
        require(matrix.shape == (len(ids), 144) and matrix.dtype == np.float32, 'Unexpected feature cache.')
    for inner in (0, 1):
        sids = fit_ids(1-inner); progress_path = scratch / f'progress{inner}.json'
        signature = {'manifest_sha256': manifest_sha, 'checkpoint_sha256': c.sha(checkpoints[inner])}
        progress = c.read(progress_path) if progress_path.exists() else dict(signature, chunks=[])
        require(all(progress[k] == v for k, v in signature.items()), 'Extraction provenance changed.')
        def rows(batch):
            return np.concatenate([np.arange(offsets[s], offsets[s+1]) for s in batch])
        done = 0
        for chunk in progress['chunks']:
            require(chunk['start'] == done, 'Extraction progress has gaps.')
            batch = sids[chunk['start']:chunk['end']]
            require(c.array_sha(matrix[rows(batch)]) == chunk['sha256'], 'Partial feature cache SHA differs.')
            done = chunk['end']
        state = torch.load(checkpoints[inner], map_location='cpu', weights_only=False)
        require(state['step'] == STEPS and state['fit_ids'] == fit_ids(inner).tolist(), 'Incorrect full-fit predictor.')
        model = network.Predictor().to(device).eval(); model.load_state_dict(state['model'])
        for lo in range(done, len(sids), 16):
            batch = sids[lo:lo+16]; seq, meta = [], []
            for sid in batch:
                h, n = int(data['hist_lengths'][sid]), int(data['lengths'][sid]); cal = min(512, h//2-64)
                v = np.array(data['values'][sid, h-cal-64:h+n]); r = np.array(data['innovations'][sid, h-cal-64:h+n])
                seq.append(network.inputs(v[:-1], r[:-1])); meta.append((v[64:], r[64:], cal))
            bx = np.zeros((len(batch), 4, max(s.shape[1] for s in seq)), dtype='f4')
            for j, item in enumerate(seq): bx[j, :, :item.shape[1]] = item
            with torch.no_grad():
                q = model(torch.from_numpy(bx).to(device), torch.from_numpy(np.array(data['static'][batch])).to(device)).cpu().numpy()
            for j, sid in enumerate(batch):
                v, r, cal = meta[j]; tt = ticks[offsets[sid]:offsets[sid+1]]
                z, width = features.feature_matrix(v, r, q[j, :, 63:63+len(r)].T, cal, tt, data['static'][sid])
                require(width == 101 and z.shape == (len(tt), 245) and np.isfinite(z).all(), 'Neural feature shape/value differs.')
                matrix[offsets[sid]:offsets[sid+1]] = z[:, 101:]
            matrix.flush()
            progress['chunks'].append({'start': lo, 'end': lo+len(batch), 'sha256': c.array_sha(matrix[rows(batch)])})
            c.write(progress_path, progress)
        del model; gc.collect()
    require(np.isfinite(matrix).all(), 'Unassigned neural feature rows.')
    require(sum(p.stat().st_size for p in scratch.iterdir() if p.is_file()) < SCRATCH_LIMIT, 'Own scratch exceeded limit.')
    return matrix, c.sha(path)


def fit_heads(out, base, neural, ids, ticks, labels, manifest_sha):
    import lightgbm as lgb
    target = np.asarray(labels[ids, ticks]); require(np.isin(target, [0, 1]).all(), 'Invalid tree targets.')
    unique, counts = np.unique(ids, return_counts=True); require(len(unique) == 10000, 'Missing full-fit series.')
    weight = np.repeat(1./counts, counts); weight /= weight.mean(); weight = weight.astype('f4')
    x = np.empty((len(ids), 566), dtype='f4'); x[:, :422] = base; x[:, 422:] = neural
    require(np.isfinite(x).all(), 'Nonfinite tree inputs.')
    hashes = {}
    for seed in c.SEEDS:
        path, marker = out / f'augmented_seed{seed}.txt', out / f'augmented_seed{seed}.json'
        if path.exists():
            require(marker.exists(), 'Unmarked existing head; preserve it for review before resuming.')
            info = c.read(marker)
            require(info['manifest_sha256'] == manifest_sha and info['sha256'] == c.sha(path), 'Saved head provenance differs.')
            booster = lgb.Booster(model_file=str(path))
        else:
            require(not marker.exists(), 'Head marker exists but model is missing.')
            model = lgb.LGBMClassifier(**c.parameters(seed)); model.fit(x, target, sample_weight=weight)
            booster = model.booster_; temporary = path.with_suffix('.tmp.txt'); booster.save_model(str(temporary))
            probe = x[np.linspace(0, len(x)-1, 17).astype(int)]
            require(np.array_equal(booster.predict(probe, num_threads=c.THREADS),
                                   lgb.Booster(model_file=str(temporary)).predict(probe, num_threads=c.THREADS)), 'Head serialization parity failed.')
            temporary.replace(path)
            c.write(marker, {'seed': seed, 'manifest_sha256': manifest_sha, 'sha256': c.sha(path), 'training_rows': len(ids)})
            del model
        require(booster.num_feature() == 566 and booster.num_trees() == 500, 'Head architecture differs.')
        hashes[path.name] = c.sha(path); del booster; gc.collect()
    return hashes


def main(out, check_only=False, stage=None, device='auto'):
    protocol, gates = authenticate_gates()  # Must precede every fitting/cache mutation.
    prepared = c.read(c.OUT / 'prepared.json')
    require(protocol['neural_fit_steps'] == STEPS and protocol['neural_fit_batch'] == BATCH, 'Frozen training budget differs.')
    for name, expected in prepared['array_hashes_verified'].items():
        require(c.sha(c.ARRAYS / name) == expected, f'Full-fit input changed: {name}')
    for name in ('sampled_base', 'sample_ids', 'sample_ticks'):
        require(c.sha(c.WORK / f'{name}.npy') == prepared[f'{name}_sha256'], f'Shared sampled cache changed: {name}')
    base = np.load(c.WORK / 'sampled_base.npy', mmap_mode='r')
    ids = np.load(c.WORK / 'sample_ids.npy'); ticks = np.load(c.WORK / 'sample_ticks.npy')
    data = {n: np.load(c.ARRAYS / f'{n}.npy', mmap_mode='r') for n in ('values', 'innovations', 'hist_lengths', 'lengths', 'static')}
    expected_ticks = [c.log_steps(int(n)) for n in data['lengths']]
    require(np.array_equal(ids, np.repeat(np.arange(10000), [len(t) for t in expected_ticks]))
            and np.array_equal(ticks, np.concatenate(expected_ticks)) and base.shape == (len(ids), 422), 'Sample identities differ.')
    for seed in c.SEEDS:
        require(c.parameters(seed) == protocol['parameters_by_seed'][str(seed)], 'Head recipe differs from frozen CV.')
    if check_only:
        print(json.dumps({'eligible': True, 'authenticated': True, 'training_started': False})); return
    # Torch and LightGBM must never share a worker process on this Mac.
    if device == 'auto':
        code = "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"
        device = subprocess.check_output([str(c.PYTHON), '-c', code], text=True).strip()
    manifest = {'full_fit_source_sha256': c.sha(Path(__file__)), 'frozen_protocol_sha256': c.sha(c.OUT / 'protocol.json'),
        'gate_results_sha256': gates, 'prepared_sha256': c.sha(c.OUT / 'prepared.json'),
        'predictor_sources': protocol['predictor_sources'], 'steps': STEPS, 'batch': BATCH,
        'predictor_fit': '(sid//5)%2 == inner; 5000 histories; seed = legacy SEED + 50 + inner',
        'neural_training_view': 'opposite inner predictor; 144 columns', 'tree_features': 566,
        'tree_parameters': protocol['parameters_by_seed'], 'training_rows': len(ids), 'training_series': 10000,
        'device': device, 'torch': version('torch'), 'numpy': np.__version__, 'lightgbm': version('lightgbm'),
        'process_policy': 'Separate Torch training/extraction workers and LightGBM head worker.',
        'scratch_limit_bytes': SCRATCH_LIMIT, 'inference_packaged': False, 'official_submission': False}
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True); manifest_path = out / 'manifest.json'
    if manifest_path.exists(): require(c.read(manifest_path) == manifest, 'Full-fit manifest changed; use a separate run directory.')
    else: c.write(manifest_path, manifest)
    manifest_sha = c.sha(manifest_path)
    complete_path = out / 'complete.json'
    if complete_path.exists():
        complete = c.read(complete_path); require(complete['manifest_sha256'] == manifest_sha, 'Completed fit manifest differs.')
        for name, digest in complete['artifacts_sha256'].items(): require(c.sha(out / name) == digest, f'Completed artifact changed: {name}')
        print('Verified existing complete full fit.'); return
    key = hashlib.sha256(str(out).encode()).hexdigest()[:12]; scratch = c.WORK / f'full_fit_{key}'
    checkpoints = [out / f'predictor{i}/latest.pt' for i in (0, 1)]
    def verify_predictors():
        for i, checkpoint in enumerate(checkpoints):
            info = c.read(checkpoint.parent / 'complete.json')
            require(info['manifest_sha256'] == manifest_sha and info['steps'] == STEPS and info['fit_series'] == 5000
                    and info['checkpoint_sha256'] == c.sha(checkpoint), f'Full predictor {i} completion differs.')
    if stage in ('predictor0', 'predictor1'):
        network, _ = c.predictor_modules(); i = int(stage[-1])
        train_predictor(i, out / f'predictor{i}', data, network, manifest_sha, device); return
    if stage == 'extract':
        verify_predictors(); network, features = c.predictor_modules()
        neural, neural_sha = extract_neural(scratch, checkpoints, data, ids, ticks, network, features, manifest_sha, device)
        c.write(out / 'extraction_complete.json', {'manifest_sha256': manifest_sha, 'neural_features_sha256': neural_sha})
        del neural; return
    if stage == 'heads':
        verify_predictors(); info = c.read(out / 'extraction_complete.json')
        require(info['manifest_sha256'] == manifest_sha and info['neural_features_sha256'] == c.sha(scratch / 'neural.npy'),
                'Extracted neural cache changed.')
        neural = np.load(scratch / 'neural.npy', mmap_mode='r')
        hashes = fit_heads(out, base, neural, ids, ticks, np.load(c.ARRAYS / 'labels.npy', mmap_mode='r'), manifest_sha)
        c.write(out / 'heads_complete.json', {'manifest_sha256': manifest_sha, 'heads_sha256': hashes}); return
    for worker in ('predictor0', 'predictor1', 'extract', 'heads'):
        subprocess.run([str(c.PYTHON), '-u', str(Path(__file__).resolve()), '--out', str(out), '--device', device, '--stage', worker], check=True)
    verify_predictors(); head_info = c.read(out / 'heads_complete.json')
    require(head_info['manifest_sha256'] == manifest_sha, 'Head completion manifest differs.')
    hashes = dict(head_info['heads_sha256'])
    for name, digest in hashes.items(): require(c.sha(out / name) == digest, f'Final head SHA differs: {name}')
    neural_sha = c.read(out / 'extraction_complete.json')['neural_features_sha256']
    for i, checkpoint in enumerate(checkpoints):
        hashes[str(checkpoint.relative_to(out))] = c.sha(checkpoint)
        hashes[f'predictor{i}/complete.json'] = c.sha(checkpoint.parent / 'complete.json')
    c.write(complete_path, {'manifest_sha256': manifest_sha, 'artifacts_sha256': hashes, 'neural_features_sha256': neural_sha,
                          'inference_packaged': False, 'official_submission': False})
    shutil.rmtree(scratch)  # Only this run's own reproducible neural cache.
    print(json.dumps({'full_fit_complete': True, 'out': str(out), 'official_submission': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=c.OUT / 'full_fit_models')
    parser.add_argument('--check-only', action='store_true', help='Authenticate gates and reusable inputs; never fit or write models.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--stage', choices=('predictor0', 'predictor1', 'extract', 'heads'), help=argparse.SUPPRESS)
    args = parser.parse_args(); main(args.out, args.check_only, args.stage, args.device)
