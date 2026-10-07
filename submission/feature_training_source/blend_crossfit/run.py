"""Freeze and compare three blends, with within-outer-fold scalar cross-fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit

OUT = Path(__file__).resolve().parent
INTEGRATION = OUT.parent / 'feature_integration'
NEW_CANDIDATES = ('fixed_contrast_0p1', 'learned_probability_blend', 'learned_logit_contrast')
REFERENCES = ('incumbent', 'prior_90_10')
SELECTION_ORDER = ('prior_90_10', *NEW_CANDIDATES)
CONFIG = {
    'schema': 1, 'epsilon': 1e-7, 'fixed_coefficient': .1,
    'delta_update_stride': 4, 'baseline_update_stride': 1,
    'outer_folds': [0, 1, 2, 3, 4], 'series': 10000, 'online_max_steps': 1000,
    'calibration_half': '(global_series_id // 5) % 2',
    'scalar_fits_per_family': 10, 'calibration_series_per_fit': 1000,
    'probability_blend_bounds': [0., .5], 'logit_contrast_bounds': [0., 1.],
    'objective': 'mean(logaddexp(0, -(logit(p_positive)-logit(p_negative)))) + l2 * coefficient**2 / 2',
    'l2': .001, 'calibration_pairs': 200000,
    'calibration_seed': '919803 + 10 * outer_fold + calibration_half',
    'pair_sampling': 'Sample time proportional to n_positive*n_negative using calibration labels only; then draw positive and negative uniformly at that same time, with replacement.',
    'same_pairs_for_both_families': True,
    'probability_optimizer': 'scipy bounded minimize_scalar, xatol=1e-12, maxiter=500; compare both endpoints and interior result',
    'contrast_optimizer': 'Convex derivative endpoint checks, else scipy brentq root, xtol=1e-12, rtol=1e-12',
    'final_fit_pairs': 1000000, 'final_fit_seed': 920003,
    'bootstrap_replicates': 300, 'bootstrap_seed': 919703,
    'time_bands': [[0, 50], [50, 200], [200, 500], [500, 1000]],
    'A': 'Saved fold augmented probabilities, averaged across seeds and predictor views before logit.',
    'C': 'Saved fold control probabilities, averaged across seeds before logit.',
    'B': 'Existing GPU plus Chronos incumbent baseline.npy.',
    'probability_formula': '(1-coefficient)*B + coefficient*A',
    'contrast_formula': 'expit(logit(clip(B)) + coefficient*(logit(clip(A))-logit(clip(C))))',
    'cadence': 'A/C held from each four-step checkpoint; B and final output update at every step.',
    'clip': '[epsilon, 1-epsilon]; probability blend clips only for fitting logit loss',
    'base_model_training': False, 'hyperparameter_grid': False,
    'bootstrap_policy': 'Unconditional on score or sign; paired global whole-series resampling, all five folds pooled.',
    'selection_order_on_exact_tie': list(SELECTION_ORDER),
    'old_reference_delta': .001, 'old_reference_is_hard_rejection_rule': False,
    'development_reused': True, 'independent_holdout': False,
    'official_score': False, 'official_submission': False, 'deployment_adopted': False,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_array(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp.npy')
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def load_helpers():
    sys.path.insert(0, str(INTEGRATION))
    spec = importlib.util.spec_from_file_location('_blend_bootstrap', INTEGRATION/'bootstrap.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clipped(value):
    return np.clip(value, CONFIG['epsilon'], 1.-CONFIG['epsilon'])


def held(value):
    value = np.asarray(value, dtype=np.float64)
    ticks = np.arange(value.shape[-1]) // CONFIG['delta_update_stride'] * CONFIG['delta_update_stride']
    return np.take(value, ticks, axis=-1)


def apply_blend(baseline, augmented, control, family, coefficient):
    """Series-independent causal transform; final dimension is online time."""
    b, a, c = (np.asarray(v, dtype=np.float64) for v in (baseline, augmented, control))
    if b.ndim < 1 or b.shape != a.shape or b.shape != c.shape:
        raise ValueError('The three inputs must have equal non-scalar shapes.')
    if family == 'probability_blend':
        return (1.-coefficient)*b + coefficient*held(a)
    if family != 'logit_contrast':
        raise ValueError(f'Unknown family: {family}')
    delta = logit(clipped(held(a))) - logit(clipped(held(c)))
    b = clipped(b)
    result = expit(logit(b) + coefficient*delta)
    return np.where((delta == 0.) | (coefficient == 0.), b, result)


def sample_pairs(calibration_labels, count, seed):
    """This function receives only calibration-half labels, never test labels."""
    y = np.asarray(calibration_labels)
    if y.ndim != 2 or not np.isin(y, (-1, 0, 1)).all():
        raise ValueError('Expected a two-dimensional binary label array with -1 suffixes.')
    masses = (y == 1).sum(axis=0, dtype=np.int64)*(y == 0).sum(axis=0, dtype=np.int64)
    total = int(masses.sum())
    if total <= 0:
        raise ValueError('No positive-negative calibration pairs.')
    rng = np.random.default_rng(seed)
    ticks = rng.choice(y.shape[1], size=count, p=masses/total).astype(np.int32)
    positive = np.empty(count, dtype=np.int32)
    negative = np.empty(count, dtype=np.int32)
    for tick in np.unique(ticks):
        positions = np.flatnonzero(ticks == tick)
        pos = np.flatnonzero(y[:, tick] == 1)
        neg = np.flatnonzero(y[:, tick] == 0)
        positive[positions] = pos[rng.integers(len(pos), size=len(positions))]
        negative[positions] = neg[rng.integers(len(neg), size=len(positions))]
    return positive, negative, ticks


def pair_values(b, a, c, pairs):
    positive, negative, ticks = pairs
    checkpoint = ticks // CONFIG['delta_update_stride'] * CONFIG['delta_update_stride']
    return {'bp': np.asarray(b[positive, ticks]), 'bn': np.asarray(b[negative, ticks]),
            'ap': np.asarray(a[positive, checkpoint]), 'an': np.asarray(a[negative, checkpoint]),
            'cp': np.asarray(c[positive, checkpoint]), 'cn': np.asarray(c[negative, checkpoint])}


def fit_scalar(values, family):
    """Fit exactly one bounded scalar; no grids or validation labels."""
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite calibration pair probabilities.')
    penalty = CONFIG['l2']
    if family == 'probability_blend':
        def objective(w):
            pos = logit(clipped((1.-w)*values['bp'] + w*values['ap']))
            neg = logit(clipped((1.-w)*values['bn'] + w*values['an']))
            return float(np.logaddexp(0., -(pos-neg)).mean() + penalty*w*w/2.)
        lower, upper = CONFIG['probability_blend_bounds']
        optimized = minimize_scalar(objective, method='bounded', bounds=(lower, upper),
                                    options={'xatol': 1e-12, 'maxiter': 500})
        if not optimized.success or not np.isfinite(optimized.fun):
            raise ValueError(f'Probability coefficient optimizer failed: {optimized}')
        candidates = [(lower, objective(lower)), (upper, objective(upper)),
                      (float(optimized.x), float(optimized.fun))]
        coefficient, loss = min(candidates, key=lambda row: (row[1], row[0]))
        return {'family': family, 'coefficient': coefficient, 'objective': loss,
                'optimizer': 'bounded_minimize_scalar_plus_endpoints', 'success': True,
                'message': str(optimized.message), 'nfev': int(optimized.nfev),
                'interior_coefficient': float(optimized.x), 'interior_objective': float(optimized.fun),
                'endpoint_objectives': [candidates[0][1], candidates[1][1]],
                'bound_selected': coefficient in (lower, upper)}
    if family != 'logit_contrast':
        raise ValueError(f'Unknown family: {family}')
    offset = logit(clipped(values['bp'])) - logit(clipped(values['bn']))
    contrast = (logit(clipped(values['ap']))-logit(clipped(values['cp']))
                -logit(clipped(values['an']))+logit(clipped(values['cn'])))
    def objective(alpha):
        return float(np.logaddexp(0., -(offset+alpha*contrast)).mean() + penalty*alpha*alpha/2.)
    def derivative(alpha):
        return float((-contrast*expit(-(offset+alpha*contrast))).mean() + penalty*alpha)
    lower, upper = CONFIG['logit_contrast_bounds']
    dlo, dhi = derivative(lower), derivative(upper)
    calls, iterations = 2, 0
    if dlo >= 0.:
        coefficient, status = lower, 'lower_endpoint_nonnegative_derivative'
    elif dhi <= 0.:
        coefficient, status = upper, 'upper_endpoint_nonpositive_derivative'
    else:
        coefficient, optimized = brentq(derivative, lower, upper, xtol=1e-12,
                                        rtol=1e-12, full_output=True, disp=True)
        if not optimized.converged:
            raise ValueError('Contrast derivative root did not converge.')
        status, calls, iterations = 'bracketed_derivative_root', int(optimized.function_calls)+2, int(optimized.iterations)
    return {'family': family, 'coefficient': float(coefficient), 'objective': objective(coefficient),
            'optimizer': 'convex_derivative_endpoints_or_brentq', 'success': True,
            'message': status, 'derivative_at_coefficient': derivative(coefficient),
            'endpoint_derivatives': [dlo, dhi], 'endpoint_objectives': [objective(lower), objective(upper)],
            'function_calls': calls, 'iterations': iterations,
            'bound_selected': coefficient in (lower, upper)}


def synthetic_checks():
    rng = np.random.default_rng(919704)
    b, a, c = [rng.uniform(.01, .99, (8, 29)) for _ in range(3)]
    max_error = 0.
    for family, coefficient in [('probability_blend', .17), ('logit_contrast', .1)]:
        actual = apply_blend(b, a, c, family, coefficient)
        scalar = np.empty_like(actual)
        for row in range(len(b)):
            for tick in range(b.shape[1]):
                k = tick//4*4
                bb, aa, cc = b[row, tick], a[row, k], c[row, k]
                scalar[row, tick] = ((1.-coefficient)*bb+coefficient*aa if family == 'probability_blend'
                                    else expit(logit(clipped(bb))+coefficient*(logit(clipped(aa))-logit(clipped(cc)))))
        max_error = max(max_error, float(np.max(np.abs(actual-scalar))))
        assert max_error <= 2e-15
        order = np.array([7, 2, 0, 4, 1, 3, 6, 5])
        assert np.array_equal(actual[order], apply_blend(b[order], a[order], c[order], family, coefficient))
        assert np.array_equal(actual[[4, 1]], apply_blend(b[[4, 1]], a[[4, 1]], c[[4, 1]], family, coefficient))
        assert np.array_equal(actual, np.stack([apply_blend(bb, aa, cc, family, coefficient) for bb,aa,cc in zip(b,a,c)]))
        for cut in (1, 3, 4, 5, 17, 28):
            assert np.array_equal(actual[:, :cut], apply_blend(b[:, :cut], a[:, :cut], c[:, :cut], family, coefficient))
            changed = [v.copy() for v in (b, a, c)]
            for value in changed:
                value[:, cut:] = rng.uniform(0., 1., value[:, cut:].shape)
            assert np.array_equal(actual[:, :cut], apply_blend(*changed, family, coefficient)[:, :cut])
        altered_a, altered_c = a.copy(), c.copy()
        mask = np.arange(a.shape[1]) % 4 != 0
        altered_a[:, mask] = 0.; altered_c[:, mask] = 1.
        assert np.array_equal(actual, apply_blend(b, altered_a, altered_c, family, coefficient))
    assert np.array_equal(apply_blend(b, a, a, 'logit_contrast', .3), clipped(b))
    assert np.array_equal(apply_blend(b, a, c, 'logit_contrast', 0.), clipped(b))
    assert np.array_equal(apply_blend(b, a, c, 'probability_blend', 0.), b)
    boundary = np.array([[0., 1e-10, 1e-7, .5, 1.-1e-7, 1.-1e-10, 1.]])
    bounded = apply_blend(boundary, np.ones_like(boundary), np.zeros_like(boundary), 'logit_contrast', 1.)
    # The mathematical expit is open-interval; float64 may round extreme
    # contrast logits to exactly one. Do not add an undeclared output clip.
    assert np.isfinite(bounded).all() and ((bounded >= 0.) & (bounded <= 1.)).all()
    changing_b = np.array([[.15, .25, .4, .55, .65, .75, .8, .85]])
    cadence = apply_blend(changing_b, np.full_like(changing_b, .7), np.full_like(changing_b, .4), 'logit_contrast', .1)
    assert np.array_equal(cadence, expit(logit(changing_b)+.1*(logit(.7)-logit(.4))))
    assert len(np.unique(cadence[0, :4])) == 4
    # Pair oracle: exact class membership, absent labels excluded, repeatability.
    labels = rng.integers(0, 2, size=b.shape, dtype=np.int8)
    labels[0, -3:] = -1
    labels[1, -2:] = -1
    ids = np.array([0, 5, 10, 15, 20, 25, 30, 35])
    cal = np.flatnonzero((ids//5)%2 == 0)
    test = np.flatnonzero((ids//5)%2 == 1)
    pairs = sample_pairs(labels[cal], 2000, 17)
    assert all(np.array_equal(x, z) for x,z in zip(pairs, sample_pairs(labels[cal], 2000, 17)))
    pos, neg, ticks = pairs
    assert (labels[cal][pos, ticks] == 1).all() and (labels[cal][neg, ticks] == 0).all()
    assert not np.isin(ids[cal][pos], ids[test]).any()
    values = pair_values(b[cal], a[cal], c[cal], pairs)
    fits = {family: fit_scalar(values, family) for family in ('probability_blend', 'logit_contrast')}
    changed_labels = labels.copy(); changed_labels[test] = 1-(changed_labels[test] == 1)
    changed_b, changed_a, changed_c = [v.copy() for v in (b,a,c)]
    for value in (changed_b, changed_a, changed_c):
        value[test] = rng.uniform(.01, .99, value[test].shape)
    changed_pairs = sample_pairs(changed_labels[cal], 2000, 17)
    changed_values = pair_values(changed_b[cal], changed_a[cal], changed_c[cal], changed_pairs)
    assert all(fits[family] == fit_scalar(changed_values, family) for family in fits)
    # Independent dense synthetic oracle, never applied to experiment arrays.
    for family, upper in [('probability_blend', .5), ('logit_contrast', 1.)]:
        def direct_objective(coefficient):
            pp = apply_blend(b[cal], a[cal], c[cal], family, coefficient)
            margin = logit(clipped(pp[pos,ticks]))-logit(clipped(pp[neg,ticks]))
            # Contrast fitting uses unconstrained latent logits. These tiny
            # inputs do not reach output clipping, so this is an exact oracle.
            return float(np.logaddexp(0., -margin).mean()+CONFIG['l2']*coefficient**2/2.)
        optimum = fits[family]
        assert abs(direct_objective(optimum['coefficient'])-optimum['objective']) < 1e-12
        assert optimum['objective'] <= min(direct_objective(x) for x in np.linspace(0., upper, 101))+1e-8
    flat = {name: np.full(100, .5) for name in ('bp','bn','ap','an','cp','cn')}
    for family in fits:
        assert fit_scalar(flat, family)['coefficient'] == 0.
    return {'passed': True, 'real_experiment_arrays_read': False,
            'implementation_sha256': sha(Path(__file__)), 'scalar_batch_max_absolute_error': max_error,
            'series_order_subset_repeat_and_row_batch_exact': True,
            'prefix_and_changed_future_exact': True, 'A_C_held_four_steps': True,
            'B_and_output_update_every_step': True, 'zero_coefficient_and_zero_contrast_identity': True,
            'epsilon_boundaries_finite_and_within_closed_unit_interval': True,
            'sampled_pairs_same_time_correct_class_and_calibration_only': True,
            'test_half_labels_and_predictions_do_not_affect_fit': True,
            'fitting_objective_matches_independent_synthetic_oracle': True,
            'both_optimizers_match_synthetic_dense_oracle': True,
            'zero_signal_selects_zero_with_regularization': True}


def collect_inputs():
    """Authenticate and hash existing files without scoring new candidates."""
    inputs = {}
    def add(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Input differs from its recorded hash: {path}')
        inputs[str(path)] = digest
    for name in ('run.py', 'README.md', 'synthetic_checks.json', 'selection_policy.json'):
        add(OUT/name)
    source_protocol = read(INTEGRATION/'protocol.json')
    add(INTEGRATION/'protocol.json')
    for name, expected in source_protocol['experiment_sources'].items():
        add(INTEGRATION/name, expected)
    checked = read(INTEGRATION/'checks.json')
    if not checked.get('passed'):
        raise ValueError('Existing bootstrap numerical validation did not pass.')
    add(INTEGRATION/'checks.json')
    add(INTEGRATION/'bootstrap.py', checked['implementation_sha256'])
    helpers = load_helpers()
    prepared = read(INTEGRATION/'prepared.json'); add(INTEGRATION/'prepared.json')
    for name in ('labels', 'baseline', 'lengths'):
        add(helpers.ARRAYS/f'{name}.npy', prepared['array_hashes_verified'][f'{name}.npy'])
    combined = read(INTEGRATION/'result.json')
    if combined['completed_folds'] != 5 or combined['series'] != CONFIG['series']:
        raise ValueError('All five feature-integration folds must be complete.')
    add(INTEGRATION/'result.json', read(OUT/'selection_policy.json')['prior_result_sha256'])
    for fold in CONFIG['outer_folds']:
        folder = INTEGRATION/f'fold{fold}'
        result = read(folder/'result.json')
        if result['fold'] != fold or result['held_series'] != 2000:
            raise ValueError(f'Unexpected completed fold identity: {folder}')
        add(folder/'result.json'); add(folder/'held_ids.npy')
        for arm in ('control', 'augmented'):
            name = f'{arm}_probability.npy'
            add(folder/name, result['prediction_hashes'][name])
    return inputs, str(helpers.ARRAYS)


def frozen_protocol():
    path = OUT/'protocol.json'
    if not path.exists():
        raise ValueError('Run --prepare and review the frozen protocol before evaluating.')
    protocol = read(path)
    if protocol['config'] != CONFIG:
        raise ValueError('Configuration changed after preparation; use a new experiment.')
    for path, expected in protocol['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input or code changed: {path}')
    return protocol


def prepare():
    if (OUT/'protocol.json').exists():
        protocol = frozen_protocol()
        print(json.dumps({'status': 'already_prepared_inputs_verified', 'input_files': len(protocol['input_sha256'])}))
        return
    checks = synthetic_checks(); write(OUT/'synthetic_checks.json', checks)
    inputs, arrays_path = collect_inputs()
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'state': 'prepared_before_new_candidate_scores_or_real_coefficient_fits',
                'config': CONFIG, 'array_path': arrays_path, 'input_sha256': inputs,
                'selection_policy_sha256': sha(OUT/'selection_policy.json'),
                'input_hash_scope': 'All consumed source data, predictions, provenance records, and executable helpers.',
                'synthetic_checks_passed': True, 'new_candidate_scored': False,
                'real_coefficient_fitted': False, 'root_review_required_before_run': True,
                'existing_development_results_already_known': True}
    with (OUT/'protocol.json').open('x') as handle:
        handle.write(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'prepared_waiting_for_review', 'protocol': str(OUT/'protocol.json'),
                      'protocol_sha256': sha(OUT/'protocol.json'), 'input_files': len(inputs),
                      'new_candidate_scored': False, 'real_coefficient_fitted': False}))


def load_predictions(protocol):
    arrays = Path(protocol['array_path'])
    y = np.load(arrays/'labels.npy', mmap_mode='r', allow_pickle=False)
    b = np.load(arrays/'baseline.npy', mmap_mode='r', allow_pickle=False)
    lengths = np.load(arrays/'lengths.npy', mmap_mode='r', allow_pickle=False)
    if y.shape != (10000, 1000) or b.shape != y.shape or lengths.shape != (10000,):
        raise ValueError('Unexpected incumbent/label/length shape.')
    valid = np.arange(y.shape[1])[None, :] < lengths[:, None]
    if not np.isin(y, (-1, 0, 1)).all() or not np.array_equal(y >= 0, valid):
        raise ValueError('Labels disagree with observed contiguous online lengths.')
    predictions = {arm: np.full(y.shape, np.nan, dtype=np.float64) for arm in ('control', 'augmented')}
    coverage = np.zeros(len(y), dtype=np.int8)
    for fold in CONFIG['outer_folds']:
        folder = INTEGRATION/f'fold{fold}'
        ids = np.load(folder/'held_ids.npy', allow_pickle=False)
        if not np.array_equal(ids, np.flatnonzero(np.arange(len(y)) % 5 == fold)):
            raise ValueError(f'Wrong series order or fold IDs: {folder}')
        coverage[ids] += 1; alive = valid[ids]
        for arm in predictions:
            p = np.load(folder/f'{arm}_probability.npy', allow_pickle=False)
            if p.shape != (len(ids), 1000) or not np.isfinite(p[alive]).all():
                raise ValueError(f'Invalid prediction array: {folder}/{arm}')
            if not ((p[alive] >= 0.) & (p[alive] <= 1.)).all():
                raise ValueError('Expected probabilities between zero and one.')
            if not np.array_equal(p[alive], held(p)[alive]):
                raise ValueError('Saved A/C probabilities do not have the frozen four-step cadence.')
            predictions[arm][ids] = p
    if not (coverage == 1).all() or not np.isfinite(b[valid]).all() or not ((b[valid] >= 0.) & (b[valid] <= 1.)).all():
        raise ValueError('Incomplete held-series coverage or invalid incumbent probability.')
    return y, b, predictions['augmented'], predictions['control']


def crossfit(y, b, a, c):
    probabilities = {name: np.full(y.shape, np.nan) for name in NEW_CANDIDATES[1:]}
    assignment = np.zeros(len(y), dtype=np.int8)
    records = []
    all_ids = np.arange(len(y))
    for fold in CONFIG['outer_folds']:
        ids = all_ids[all_ids % 5 == fold]
        for calibration_half in (0, 1):
            cal = ids[(ids//5)%2 == calibration_half]
            test = ids[(ids//5)%2 != calibration_half]
            if len(cal) != 1000 or len(test) != 1000 or np.intersect1d(cal,test).size:
                raise ValueError('Unexpected calibration/test partition.')
            seed = 919803+10*fold+calibration_half
            pairs = sample_pairs(y[cal], CONFIG['calibration_pairs'], seed)
            values = pair_values(b[cal], a[cal], c[cal], pairs)
            fitted = {}
            for name, family in [('learned_probability_blend', 'probability_blend'),
                                 ('learned_logit_contrast', 'logit_contrast')]:
                fit = fit_scalar(values, family); fitted[name] = fit
                prediction = apply_blend(b[test], a[test], c[test], family, fit['coefficient'])
                prediction[y[test] < 0] = np.nan
                probabilities[name][test] = prediction
            assignment[test] += 1
            records.append({'outer_fold': fold, 'calibration_half': calibration_half,
                            'calibration_ids': cal.tolist(), 'test_ids': test.tolist(),
                            'calibration_ids_sha256': array_sha(cal), 'test_ids_sha256': array_sha(test),
                            'base_model_training_excludes_entire_outer_fold': True,
                            'calibration_pair_seed': seed, 'calibration_pair_count': CONFIG['calibration_pairs'],
                            'calibration_pairs_global_ids_and_time_sha256': array_sha(np.stack([cal[pairs[0]],cal[pairs[1]],pairs[2]],axis=1)),
                            'test_labels_used_in_pair_sampling_or_fitting': False,
                            'shared_identical_pairs_for_both_families': True, 'fits': fitted})
            print(json.dumps({'phase': 'scalar_crossfit', 'fold': fold, 'calibration_half': calibration_half,
                              'coefficients': {name: fit['coefficient'] for name,fit in fitted.items()}}), flush=True)
    if not (assignment == 1).all():
        raise ValueError('Every series must receive exactly one out-of-calibration prediction.')
    return probabilities, records


def summary(labels, probabilities, metric):
    scores = {name: metric(labels, p) for name,p in probabilities.items()}
    return {'scores': scores, 'deltas': {f'{name}_minus_{reference}': scores[name]-scores[reference]
             for name in NEW_CANDIDATES for reference in REFERENCES}}


def completed_result():
    result = read(OUT/'result.json')
    if result['protocol_sha256'] != sha(OUT/'protocol.json') or result['selection_policy_sha256'] != sha(OUT/'selection_policy.json'):
        raise ValueError('Result provenance differs from frozen protocol or selection policy.')
    for name, expected in result['output_sha256'].items():
        if sha(OUT/name) != expected:
            raise ValueError(f'Completed output changed: {name}')
    return result


def select_candidate(scores):
    best = max(SELECTION_ORDER, key=lambda name: scores[name])
    return best if scores[best] > scores['incumbent'] else None


def write_report(result):
    scores = result['pooled_scores']
    lines = ['# Blend cross-fitting comparison', '',
             'Each learned coefficient is fit on one half of an excluded outer fold and evaluated on the other half, then the halves swap. Base model training excludes the entire outer fold. All 10,000 series receive one out-of-calibration prediction.', '',
             '| Candidate | Pooled TS-AUC | Gain over incumbent |', '|---|---:|---:|']
    for name in ('incumbent', *SELECTION_ORDER):
        lines.append(f'| {name} | {scores[name]:.9f} | {scores[name]-scores["incumbent"]:+.9f} |')
    lines += ['', '| Comparison | Delta | Whole-series bootstrap 95% CI |', '|---|---:|---:|']
    for name,row in result['bootstrap']['comparisons'].items():
        lo,hi = row['ci_95']; lines.append(f'| {name} | {row["point_delta"]:+.9f} | [{lo:+.9f}, {hi:+.9f}] |')
    lines += ['', '| Fold | Incumbent | Prior 90/10 | Fixed contrast | Learned probability | Learned contrast |', '|---|---:|---:|---:|---:|---:|']
    for row in result['folds']:
        values = row['scores']
        lines.append(f'| {row["fold"]+1} | '+' | '.join(f'{values[name]:.9f}' for name in ('incumbent',*SELECTION_ORDER))+' |')
    lines += ['', '| Time band | Incumbent | Prior 90/10 | Fixed contrast | Learned probability | Learned contrast | Pair mass |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['time_bands']:
        lines.append(f'| [{row["start"]}, {row["end"]}) | '+' | '.join(f'{row["scores"][name]:.9f}' for name in ('incumbent',*SELECTION_ORDER))+f' | {row["pair_mass"]:,} |')
    lines += ['', '| Outer fold | Calibration half | Probability weight | Contrast alpha |', '|---|---|---:|---:|']
    for row in result['crossfit']['fits']:
        fits = row['fits']; lines.append(f'| {row["outer_fold"]+1} | {row["calibration_half"]} | {fits["learned_probability_blend"]["coefficient"]:.9f} | {fits["learned_logit_contrast"]["coefficient"]:.9f} |')
    lines += ['', f'Selected for candidate preparation: `{result["selected_candidate"]}`.',
              'The selection policy was frozen before new candidate scores. It selects the highest pooled score with positive incumbent gain; exact ties prefer the prior fixed 90/10 recipe. The old +0.001 reference and positive confidence-interval lower bound are reported descriptively, with no hard rejection rule.', '',
              'Bootstrap resamples whole series globally, using the same 300 multiplicity draws for all arms and times. Scores pool all series, rather than averaging folds. Intervals condition on saved models and fitted coefficients; models and coefficients are not refit in bootstrap draws.', '',
              'This repeatedly used development set is not an independent holdout. Family selection itself uses these development scores. The calibration experiment trains each scalar on 1,000 unseen series; a subsequent all-OOF serving refit is training and receives no new validation claim. No official score, deployment adoption or submission is performed.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def run():
    protocol = frozen_protocol()
    if (OUT/'result.json').exists():
        completed_result(); print(json.dumps({'status': 'already_complete_verified'})); return
    began = time.time(); helpers = load_helpers()
    y,b,a,c = load_predictions(protocol)
    learned, fits = crossfit(y,b,a,c)
    probabilities = {'incumbent': b, 'prior_90_10': .9*b+.1*a,
                     'fixed_contrast_0p1': apply_blend(b,a,c,'logit_contrast',.1), **learned}
    for name in NEW_CANDIDATES:
        probabilities[name][y < 0] = np.nan
    pooled = summary(y, probabilities, helpers.metric)
    previous = read(INTEGRATION/'result.json')
    if abs(pooled['scores']['incumbent']-previous['incumbent']) >= 1e-12 or abs(pooled['scores']['prior_90_10']-previous['fixed_10pct_blends']['augmented']) >= 1e-12:
        raise ValueError('Previously recorded comparator scores did not reproduce.')
    folds = []
    for fold in CONFIG['outer_folds']:
        mask = np.arange(len(y)) % 5 == fold
        folds.append({'fold': fold, 'series': int(mask.sum()), **summary(y[mask], {name:p[mask] for name,p in probabilities.items()}, helpers.metric)})
    bands = []
    for start,end in CONFIG['time_bands']:
        yy = y[:,start:end]
        bands.append({'start': start, 'end': end,
                      'pair_mass': int(((yy==1).sum(axis=0)*(yy==0).sum(axis=0)).sum()),
                      **summary(yy, {name:p[:,start:end] for name,p in probabilities.items()}, helpers.metric)})
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    draws = rng.multinomial(len(y), np.full(len(y),1./len(y)), size=CONFIG['bootstrap_replicates']).astype(np.int32)
    counts = np.concatenate([np.ones((1,len(y)),dtype=np.int32),draws])
    samples,pair_mass,details = helpers.weighted_scores(y,probabilities,counts,
        progress=lambda item: print(json.dumps(item),flush=True))
    for name in probabilities:
        if abs(float(samples[name][0])-pooled['scores'][name]) >= 1e-12:
            raise ValueError('Bootstrap identity row differs from direct metric.')
    comparisons = {}
    for name in NEW_CANDIDATES:
        for reference in REFERENCES:
            key = f'{name}_minus_{reference}'
            differences = samples[name][1:]-samples[reference][1:]
            lo,hi = np.quantile(differences,[.025,.975],method='linear')
            comparisons[key] = {'point_delta': pooled['deltas'][key], 'ci_95': [float(lo),float(hi)],
                                'bootstrap_mean': float(differences.mean()), 'bootstrap_std': float(differences.std(ddof=1)),
                                'fraction_positive': float((differences>0).mean()), 'delta_draws': differences.tolist()}
    frozen_protocol()
    save_array(OUT/'global_series_ids.npy', np.arange(len(y), dtype=np.int64))
    output_hashes = {'global_series_ids.npy': sha(OUT/'global_series_ids.npy')}
    for name in NEW_CANDIDATES:
        path = OUT/f'{name}_probability.npy'; save_array(path,probabilities[name]); output_hashes[path.name] = sha(path)
    result = {'status': 'completed_descriptive_crossfit_comparison', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(OUT/'protocol.json'), 'selection_policy_sha256': sha(OUT/'selection_policy.json'),
              'config': CONFIG, 'series': len(y), 'observations': int((y>=0).sum()), 'all_five_folds': True,
              'pooled_scores': pooled['scores'], 'pooled_deltas': pooled['deltas'], 'folds': folds, 'time_bands': bands,
              'crossfit': {'fits': fits, 'out_of_calibration_coverage': len(y),
                          'every_evaluation_series_excluded_from_its_base_model_and_scalar_fit': True,
                          'global_OOF_outer_training_offset_used': False},
              'selected_candidate': select_candidate(pooled['scores']),
              'selection_is_candidate_preparation_only': True,
              'previous_reference': {'delta': .001, 'hard_rejection_rule': False,
                                    'reached_by': {name: pooled['scores'][name]-pooled['scores']['incumbent']>=.001 for name in SELECTION_ORDER}},
              'bootstrap': {'replicates': CONFIG['bootstrap_replicates'], 'seed': CONFIG['bootstrap_seed'],
                            'unconditional_on_candidate_scores': True, 'models_and_coefficients_held_fixed': True,
                            'scheme': 'Global whole-series multinomial draws shared across all arms and ticks.',
                            'ci_method': 'Paired percentile interval with linear 2.5% and 97.5% quantiles.',
                            'identity_row_matches_unweighted_metric': True, 'comparisons': comparisons,
                            'draw_counts_sha256': array_sha(draws), 'minimum_pair_mass': int(pair_mass[1:].min()),
                            'maximum_pair_mass': int(pair_mass[1:].max()),
                            'scores': {name:p[1:].tolist() for name,p in samples.items()}, 'details': details},
              'output_sha256': output_hashes, 'series_ids_file': 'global_series_ids.npy',
              'candidate_prediction_files': {name: f'{name}_probability.npy' for name in NEW_CANDIDATES},
              'prediction_layout': 'Global series id 0..9999; online tick 0..999; invalid suffixes NaN.',
              'input_sha256': protocol['input_sha256'], 'seconds': time.time()-began,
              'development_reused': True, 'independent_holdout': False,
              'official_score': False, 'official_submission': False, 'deployment_adopted': False}
    write_report(result); result['output_sha256']['RESULTS.md'] = sha(OUT/'RESULTS.md')
    write(OUT/'result.json',result)
    print(json.dumps({'status': result['status'], 'pooled_scores': result['pooled_scores'],
                      'selected_candidate': result['selected_candidate'],
                      'ci_95': {name:row['ci_95'] for name,row in comparisons.items()}},ensure_ascii=False),flush=True)


def fit_final():
    protocol = frozen_protocol(); result = completed_result()
    selected = select_candidate(result['pooled_scores'])
    if selected != result['selected_candidate'] or selected is None:
        raise ValueError('No authenticated positive-gain selected family exists.')
    if (OUT/'final_coefficient.json').exists():
        prior = read(OUT/'final_coefficient.json')
        if prior['result_sha256'] != sha(OUT/'result.json') or prior['protocol_sha256'] != sha(OUT/'protocol.json'):
            raise ValueError('Existing final coefficient provenance differs.')
        print(json.dumps({'status': 'already_fit_final_verified', 'candidate': prior['candidate'], 'coefficient': prior['coefficient']})); return
    family = 'probability_blend' if selected in ('prior_90_10','learned_probability_blend') else 'logit_contrast'
    learned = selected.startswith('learned_')
    record = {'status': 'final_serving_coefficient_training_only', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'candidate': selected, 'family': family, 'protocol_sha256': sha(OUT/'protocol.json'),
              'result_sha256': sha(OUT/'result.json'), 'selection_policy_sha256': sha(OUT/'selection_policy.json'),
              'selected_validation_score': result['pooled_scores'][selected],
              'epsilon': CONFIG['epsilon'], 'delta_update_stride': CONFIG['delta_update_stride'],
              'input_sha256': protocol['input_sha256'],
              'new_validation_score_computed': False, 'development_reused': True,
              'official_submission': False, 'deployment_adopted': False}
    if learned:
        y,b,a,c = load_predictions(protocol)
        pairs = sample_pairs(y,CONFIG['final_fit_pairs'],CONFIG['final_fit_seed'])
        fit = fit_scalar(pair_values(b,a,c,pairs),family)
        record.update(coefficient=fit['coefficient'], fit={**fit, 'series': len(y),
                      'pair_count': CONFIG['final_fit_pairs'], 'seed': CONFIG['final_fit_seed'],
                      'training_series_ids': list(range(len(y))),
                      'training_series_ids_sha256': array_sha(np.arange(len(y), dtype=np.int64)),
                      'objective_definition': CONFIG['objective'], 'l2': CONFIG['l2'],
                      'pairs_global_ids_and_time_sha256': array_sha(np.stack(pairs,axis=1)),
                      'source': 'All stored out-of-fold training predictions; this is a serving refit, not heldout evaluation.'})
    else:
        record.update(coefficient=.1, fit={'source': 'Frozen fixed coefficient; no fitting performed.', 'success': True})
    frozen_protocol(); completed_result(); write(OUT/'final_coefficient.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true',help='Synthetic invariance/fitting checks only; no experiment arrays or output files.')
    mode.add_argument('--prepare',action='store_true',help='Check synthetically and freeze code/input hashes without real scoring or coefficient fitting.')
    mode.add_argument('--run',action='store_true',help='After review, verify frozen inputs and run the three-candidate experiment.')
    mode.add_argument('--fit-final',action='store_true',help='After family selection, fit the serving scalar on all OOF rows; training only.')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(synthetic_checks()))
    elif args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        fit_final()
