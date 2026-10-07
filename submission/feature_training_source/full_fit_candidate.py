"""Authorized candidate full-data fit; statistical records remain unchanged.

Architecture and fitting stages derive from full_fit.py. A hash-bound explicit
promotion record selects an authenticated cross-fitted blend and its separately
fitted serving scalar. Prior statistical flags are preserved. No submission.
"""
import argparse
import gc
import hashlib
import importlib.util
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
DERIVED_SOURCE_SHA256 = 'e0e5e5c36cb335ccc891193653cdd6b2dde2e8de1c00646cf1ec1135117a635b'
FAMILIES = {'prior_90_10': 'probability_blend', 'fixed_contrast_0p1': 'logit_contrast',
            'learned_probability_blend': 'probability_blend', 'learned_logit_contrast': 'logit_contrast'}
PROMOTION_ARTIFACTS = {'result.json', 'bootstrap.json', 'ensemble_bootstrap.json',
    '../blend_crossfit/protocol.json', '../blend_crossfit/result.json',
    '../blend_crossfit/final_coefficient.json', '../blend_crossfit/selection_policy.json'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_promotion_fields(authorization):
    require(authorization.get('schema_version') == 1 and authorization.get('status') == 'authorized',
            'Explicit candidate promotion authorization is required.')
    require(authorization.get('selected_candidate') in FAMILIES, 'Unknown authorized blend candidate.')
    require(authorization.get('completed_folds') == list(range(5)) and authorization.get('series') == 10000,
            'Promotion must identify all five folds and 10,000 series.')
    require(authorization.get('require_positive_point_gain') is True and
            authorization.get('official_submission_authorized') is False, 'Promotion scope differs.')
    require(PROMOTION_ARTIFACTS <= set(authorization.get('artifacts_sha256', {})),
            'Promotion must bind the old records and all four comparison artifacts.')


def authenticate_hashes(mapping, base):
    require(bool(mapping), 'Empty provenance mapping.')
    for name, digest in mapping.items():
        path = Path(name) if Path(name).is_absolute() else base / name
        require(c.sha(path) == digest, f'Authenticated artifact changed: {path}')


def validate_final_coefficient(record, candidate, config):
    require(record['status'] == 'final_serving_coefficient_training_only' and record['candidate'] == candidate,
            'The serving scalar must belong to the authorized selected candidate.')
    family = FAMILIES[candidate]
    require(record['family'] == family and record['epsilon'] == config['epsilon'] == 1e-7
            and record['delta_update_stride'] == config['delta_update_stride'] == 4, 'Serving formula differs.')
    coefficient = record['coefficient']; upper = .5 if family == 'probability_blend' else 1.
    require(not isinstance(coefficient, bool) and np.isfinite(coefficient) and 0 <= coefficient <= upper,
            'Serving coefficient is outside the declared family bounds.')
    require(record['new_validation_score_computed'] is False and record['official_submission'] is False,
            'An all-OOF scalar refit must not be represented as a new validation or submission.')
    fit = record['fit']; require(fit['success'] is True, 'Scalar fitting did not finish successfully.')
    if candidate.startswith('learned_'):
        require(fit['family'] == family and fit['coefficient'] == coefficient and fit['series'] == 10000,
                'Learned serving scalar provenance differs.')
        require(fit['training_series_ids'] == list(range(10000)) and
                fit['training_series_ids_sha256'] == c.array_sha(np.arange(10000, dtype=np.int64)),
                'The serving scalar must use the recorded all-OOF training rows.')
        require(fit['pair_count'] == config['final_fit_pairs'] == 1000000 and
                fit['seed'] == config['final_fit_seed'] == 920003 and
                fit['objective_definition'] == config['objective'] and fit['l2'] == config['l2'] == .001,
                'Serving scalar objective or pair budget changed.')
        require(len(fit['pairs_global_ids_and_time_sha256']) == 64, 'Missing serving pair identity.')
    else:
        require(coefficient == .1, 'The fixed comparison requires coefficient 0.1.')
    return {'candidate': candidate, 'family': family, 'coefficient': coefficient,
            'epsilon': 1e-7, 'delta_update_stride': 4}


def authenticate_candidate_promotion():
    """Authenticate existing experiments and rescore saved CV outputs; never fit."""
    authorization_path = c.OUT / 'candidate_promotion.json'
    require(authorization_path.exists(), 'candidate_promotion.json is absent; no full fit was started.')
    authorization = c.read(authorization_path); check_promotion_fields(authorization)
    authenticate_hashes(authorization['artifacts_sha256'], c.OUT)
    require(c.sha(c.OUT / 'full_fit.py') == DERIVED_SOURCE_SHA256, 'Reviewed full-fit source changed.')
    protocol = c.ensure_protocol()
    primary, ensemble = (c.read(c.OUT / f'{name}.json') for name in ('bootstrap', 'ensemble_bootstrap'))
    require(primary['status'] == 'complete' and primary['series'] == 10000 and
            primary['gate']['completed_folds'] == list(range(5)), 'Primary evaluation record is incomplete.')
    require(primary['frozen_protocol_sha256'] == c.sha(c.OUT / 'protocol.json') and
            primary['implementation_sha256'] == c.sha(c.OUT / 'bootstrap.py') and
            primary['checks_sha256'] == c.sha(c.OUT / 'checks.json') and
            primary['gate']['combined_report_sha256'] == c.sha(c.OUT / 'result.json'),
            'Primary evaluation provenance changed.')
    authenticate_hashes(primary['inputs_sha256'], c.OUT)
    require(ensemble['status'] in ('complete', 'skipped'), 'Unexpected incumbent bootstrap state.')
    require(set(ensemble['source_sha256']) == {'ensemble_bootstrap.py', 'bootstrap.py', 'integration_common.py', 'protocol.json'},
            'Unexpected incumbent bootstrap source contract.')
    authenticate_hashes(ensemble['source_sha256'], c.OUT)
    authenticate_hashes(ensemble['saved_point_gate']['report_sources_sha256'], c.OUT)
    if ensemble.get('inputs_sha256'): authenticate_hashes(ensemble['inputs_sha256'], c.OUT)
    for name, digest in protocol['predictor_sources'].items():
        require(c.sha(c.OUT / 'predictor_source' / name) == digest, f'Predictor source changed: {name}')
    require(c.sha(c.RECIPE) == protocol['original_head_recipe_sha256'] and
            c.sha(c.MANIFEST) == protocol['feature_manifest_sha256'], 'Frozen tree recipe/features changed.')

    # The old skipped ensemble result has no authenticated inputs. Authenticate
    # its complete model/prediction lineage now, without changing that old file
    # or treating its old eligibility predicate as an adoption requirement.
    import ensemble_bootstrap as previous
    yy, pp, folds, _, incumbent_lineage = previous.authenticated_inputs(protocol)
    saved_gate = previous.saved_point_gate()
    require(saved_gate == ensemble['saved_point_gate'], 'Old saved point record changed.')
    previous_point = previous.recompute_point_gate(yy, pp, folds, saved_gate)
    del yy, pp, folds; gc.collect()

    blend_dir = c.OUT.parent / 'blend_crossfit'
    specification = importlib.util.spec_from_file_location('_candidate_blend_comparison', blend_dir / 'run.py')
    blend = importlib.util.module_from_spec(specification); specification.loader.exec_module(blend)
    comparison_protocol = blend.frozen_protocol(); comparison = blend.completed_result()
    require(comparison['status'] == 'completed_descriptive_crossfit_comparison' and
            comparison['series'] == 10000 and comparison['all_five_folds'] is True, 'Blend comparison is incomplete.')
    require(comparison['input_sha256'] == comparison_protocol['input_sha256'], 'Comparison input maps differ.')
    require(comparison_protocol['selection_policy_sha256'] == c.sha(blend_dir / 'selection_policy.json'),
            'Comparison selection policy changed.')
    require(Path(comparison_protocol['array_path']).resolve() == c.ARRAYS.resolve(), 'Comparison uses different data arrays.')
    candidate = authorization['selected_candidate']
    require(candidate == comparison['selected_candidate'] == blend.select_candidate(comparison['pooled_scores']),
            'Authorization must select the recorded comparison winner.')
    coefficient_path = blend_dir / 'final_coefficient.json'; coefficient_record = c.read(coefficient_path)
    require(coefficient_record['protocol_sha256'] == c.sha(blend_dir / 'protocol.json') and
            coefficient_record['result_sha256'] == c.sha(blend_dir / 'result.json') and
            coefficient_record['selection_policy_sha256'] == c.sha(blend_dir / 'selection_policy.json') and
            coefficient_record['input_sha256'] == comparison_protocol['input_sha256'], 'Serving scalar provenance differs.')
    serving = validate_final_coefficient(coefficient_record, candidate, comparison_protocol['config'])
    require(abs(coefficient_record['selected_validation_score']-comparison['pooled_scores'][candidate]) < 1e-12,
            'The serving scalar references a different validation score.')
    row_name = comparison['series_ids_file']
    require(row_name in comparison['output_sha256'] and
            np.array_equal(np.load(blend_dir / row_name, allow_pickle=False), np.arange(10000, dtype=np.int64)),
            'CV prediction row identities differ.')
    labels, baseline, augmented, control = blend.load_predictions(comparison_protocol)
    require(np.array_equal(labels >= 0, np.isfinite(baseline) & (labels >= 0)), 'Invalid incumbent on alive rows.')
    if candidate == 'prior_90_10':
        prediction = .9*baseline + .1*augmented
    else:
        name = comparison['candidate_prediction_files'][candidate]
        require(name in comparison['output_sha256'], 'Selected prediction is absent from output provenance.')
        prediction = np.load(blend_dir / name, mmap_mode='r', allow_pickle=False)
    require(prediction.shape == labels.shape == (10000, 1000), 'CV prediction shape differs.')
    # Reconstruct selected learned CV predictions from each recorded *opposite*
    # calibration half. Never apply the all-OOF serving scalar to call it CV.
    if candidate.startswith('learned_'):
        coverage = np.zeros(10000, dtype=np.int8)
        for row in comparison['crossfit']['fits']:
            fold, half = row['outer_fold'], row['calibration_half']
            ids = np.flatnonzero(np.arange(10000) % 5 == fold)
            cal, test = ids[(ids//5)%2 == half], ids[(ids//5)%2 != half]
            require(row['calibration_ids'] == cal.tolist() and row['test_ids'] == test.tolist() and
                    row['test_labels_used_in_pair_sampling_or_fitting'] is False and
                    row['base_model_training_excludes_entire_outer_fold'] is True, 'Invalid scalar CV partition.')
            fit = row['fits'][candidate]
            require(fit['success'] is True and fit['family'] == serving['family'], 'Invalid selected CV scalar fit.')
            expected = blend.apply_blend(baseline[test], augmented[test], control[test], serving['family'], fit['coefficient'])
            alive = labels[test] >= 0
            require(np.array_equal(prediction[test][alive], expected[alive]), 'Saved CV outputs do not match their excluded-half scalars.')
            coverage[test] += 1
        require(np.all(coverage == 1), 'Every series must receive exactly one excluded-half prediction.')
    elif candidate == 'fixed_contrast_0p1':
        expected = blend.apply_blend(baseline, augmented, control, 'logit_contrast', .1)
        require(np.array_equal(prediction[labels >= 0], expected[labels >= 0]), 'Fixed contrast outputs differ.')
    incumbent_score, score = c.metric(labels, baseline), c.metric(labels, prediction)
    require(abs(incumbent_score-comparison['pooled_scores']['incumbent']) < 1e-12 and
            abs(score-comparison['pooled_scores'][candidate]) < 1e-12, 'Selected recorded CV score did not reproduce.')
    gain = score-incumbent_score; require(gain > 0, 'Authorized candidate must have a positive pooled CV point gain.')
    verified_folds = []
    require([r['fold'] for r in comparison['folds']] == list(range(5)), 'Missing blend CV folds.')
    for row in comparison['folds']:
        mask = np.arange(10000) % 5 == row['fold']
        value, incumbent = c.metric(labels[mask], prediction[mask]), c.metric(labels[mask], baseline[mask])
        require(abs(value-row['scores'][candidate]) < 1e-12 and abs(incumbent-row['scores']['incumbent']) < 1e-12,
                'Selected fold score did not reproduce.')
        verified_folds.append({'fold': row['fold'], 'candidate': value, 'incumbent': incumbent, 'gain': value-incumbent})
    draws = comparison['bootstrap']['scores']
    differences = np.asarray(draws[candidate])-np.asarray(draws['incumbent'])
    require(differences.shape == (300,) and np.isfinite(differences).all(), 'Selected paired bootstrap record is incomplete.')
    interval = np.quantile(differences, [.025, .975], method='linear').tolist()
    key = f'{candidate}_minus_incumbent'
    if key in comparison['bootstrap']['comparisons']:
        require(np.allclose(interval, comparison['bootstrap']['comparisons'][key]['ci_95'], rtol=0, atol=1e-12),
                'Recorded selected uncertainty interval differs.')
    original_statistics = {
        'primary': {k: primary.get(k) for k in ('status', 'point_delta', 'ci_95', 'passes_frozen_primary_adoption_gate', 'decision')},
        'incumbent_90_10': {k: ensemble.get(k) for k in ('status', 'ci_95', 'passes_frozen_incumbent_adoption_gate', 'reason')},
    }
    evidence = {'authorization_sha256': c.sha(authorization_path), 'authorization': authorization,
        'serving': serving, 'head_arms': ['control', 'augmented'] if serving['family'] == 'logit_contrast' else ['augmented'],
        'incumbent_model_and_prediction_lineage_sha256': incumbent_lineage,
        'previous_90_10_point_recomputed': previous_point, 'original_statistics_unchanged': original_statistics,
        'comparison_protocol_sha256': c.sha(blend_dir / 'protocol.json'), 'comparison_result_sha256': c.sha(blend_dir / 'result.json'),
        'final_coefficient_sha256': c.sha(coefficient_path), 'coefficient_fit': coefficient_record['fit'],
        'validated_cv_point': {'incumbent': incumbent_score, 'candidate': score, 'gain': gain, 'folds': verified_folds},
        'selected_validation_ci_95': interval, 'selected_ci_source': 'Stored paired whole-series bootstrap draws; no refitting.',
        'confidence_lower_bound_is_not_a_promotion_gate': True, 'prior_0_001_threshold_is_not_a_promotion_gate': True,
        'full_oof_serving_coefficient_has_new_validation_score': False, 'official_submission_authorized': False}
    return protocol, evidence


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


def fit_heads(out, base, neural, ids, ticks, labels, manifest_sha, arms):
    """Use one row/target/weight plan for every required full-data head."""
    import lightgbm as lgb
    require(arms in (['augmented'], ['control', 'augmented']), 'Unsupported candidate head plan.')
    target = np.asarray(labels[ids, ticks]); require(np.isin(target, [0, 1]).all(), 'Invalid tree targets.')
    unique, counts = np.unique(ids, return_counts=True); require(len(unique) == 10000, 'Missing full-fit series.')
    weight = np.repeat(1./counts, counts); weight /= weight.mean(); weight = weight.astype('f4')
    hashes = {}
    for arm in arms:
        if arm == 'control':
            x = base
        else:
            x = np.empty((len(ids), 566), dtype='f4'); x[:, :422] = base; x[:, 422:] = neural
        require(np.isfinite(x).all(), 'Nonfinite tree inputs.')
        for seed in c.SEEDS:
            path, marker = out / f'{arm}_seed{seed}.txt', out / f'{arm}_seed{seed}.json'
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
                c.write(marker, {'arm': arm, 'seed': seed, 'features': x.shape[1], 'manifest_sha256': manifest_sha,
                                 'sha256': c.sha(path), 'training_rows': len(ids)})
                del model
            require(booster.num_feature() == x.shape[1] and booster.num_trees() == 500, 'Head architecture differs.')
            hashes[path.name] = c.sha(path); del booster; gc.collect()
        del x; gc.collect()
    return hashes


def main(out, check_only=False, stage=None, device='auto'):
    protocol, evidence = authenticate_candidate_promotion()  # Must precede every fitting/cache mutation.
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
        print(json.dumps({'promotion_authorized': True, 'authenticated': True, 'training_started': False,
                          'serving': evidence['serving'], 'validated_cv_point': evidence['validated_cv_point'],
                          'selected_validation_ci_95': evidence['selected_validation_ci_95'],
                          'original_statistics_unchanged': evidence['original_statistics_unchanged']})); return
    # Torch and LightGBM must never share a worker process on this Mac.
    if device == 'auto':
        code = "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"
        device = subprocess.check_output([str(c.PYTHON), '-c', code], text=True).strip()
    manifest = {'full_fit_source_sha256': c.sha(Path(__file__)), 'frozen_protocol_sha256': c.sha(c.OUT / 'protocol.json'),
        'promotion_evidence': evidence, 'derived_full_fit_source_sha256': DERIVED_SOURCE_SHA256, 'prepared_sha256': c.sha(c.OUT / 'prepared.json'),
        'predictor_sources': protocol['predictor_sources'], 'steps': STEPS, 'batch': BATCH,
        'predictor_fit': '(sid//5)%2 == inner; 5000 histories; seed = legacy SEED + 50 + inner',
        'neural_training_view': 'opposite inner predictor; 144 columns', 'tree_features': 566,
        'tree_parameters': protocol['parameters_by_seed'], 'head_arms': evidence['head_arms'], 'training_rows': len(ids), 'training_series': 10000,
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
        hashes = fit_heads(out, base, neural, ids, ticks, np.load(c.ARRAYS / 'labels.npy', mmap_mode='r'), manifest_sha, evidence['head_arms'])
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
    parser.add_argument('--out', type=Path, default=c.OUT / 'candidate_models')
    parser.add_argument('--check-only', action='store_true', help='Authenticate gates and reusable inputs; never fit or write models.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--stage', choices=('predictor0', 'predictor1', 'extract', 'heads'), help=argparse.SUPPRESS)
    args = parser.parse_args(); main(args.out, args.check_only, args.stage, args.device)
