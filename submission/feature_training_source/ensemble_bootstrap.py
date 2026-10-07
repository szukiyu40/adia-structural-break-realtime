"""Conditional paired bootstrap: incumbent versus the fixed 10% augmented blend.

This follow-up leaves the original feature-comparison protocol unchanged. It
uses the same validated, exact weighted TS-AUC implementation as bootstrap.py.
No weights are tuned. Full computation requires the saved five-fold report to
show a pooled gain >= .001 and a positive gain on at least four folds.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import integration_common as C
from bootstrap import weighted_scores, validate_weighted_implementation


REPLICATES = 300
SEED = 919703
MINIMUM_GAIN = .001
MINIMUM_POSITIVE_FOLDS = 4
RESULT_PATH = C.OUT / 'ensemble_bootstrap.json'
ARMS = ('incumbent', 'fixed_blend')


def fixed_blend(incumbent, augmented):
    """Keep precisely the original NumPy expression and input dtypes."""
    assert incumbent.shape == augmented.shape
    return .9 * incumbent + .1 * augmented


def saved_point_gate():
    """Read small saved reports only; do not load full predictions at this gate."""
    completed = [fold for fold in range(5) if (C.OUT / f'fold{fold}/result.json').exists()]
    sources = {f'fold{fold}/result.json': C.sha(C.OUT / f'fold{fold}/result.json') for fold in completed}
    base = {'completed_folds': completed, 'minimum_gain': MINIMUM_GAIN,
            'minimum_positive_folds': MINIMUM_POSITIVE_FOLDS,
            'fixed_blend_weight': .1, 'report_sources_sha256': sources}
    if completed != list(range(5)):
        return dict(base, eligible=False, reason='All five completed fold reports are required.')
    if not (C.OUT / 'result.json').exists():
        return dict(base, eligible=False, reason='The combined five-fold report is required.')
    combined = C.read(C.OUT / 'result.json')
    sources['result.json'] = C.sha(C.OUT / 'result.json')
    if combined['completed_folds'] != 5 or combined['series'] != 10000:
        return dict(base, eligible=False, reason='The combined report must cover all 10,000 held series.')
    fold_gains = []
    for fold in completed:
        result = C.read(C.OUT / f'fold{fold}/result.json')
        delta = result['fixed_10pct_blends']['augmented'] - result['incumbent']
        assert abs(delta-result['augmented_blend_delta_vs_incumbent']) < 1e-12
        fold_gains.append(float(delta))
    pooled_gain = combined['fixed_10pct_blends']['augmented'] - combined['incumbent']
    positive = sum(delta > 0 for delta in fold_gains)
    eligible = pooled_gain >= MINIMUM_GAIN and positive >= MINIMUM_POSITIVE_FOLDS
    return dict(base, eligible=bool(eligible), reported_pooled_gain=float(pooled_gain),
                reported_fold_gains=fold_gains, reported_positive_folds=positive,
                reason='Fixed incumbent-blend point gate passed.' if eligible else
                'Fixed incumbent-blend pooled-gain or positive-fold gate failed.')


def authenticated_inputs(protocol):
    """Authenticate both probabilities and the five-fold model lineage."""
    prepared = C.read(C.OUT / 'prepared.json')
    provenance = {'prepared.json': C.sha(C.OUT / 'prepared.json')}
    for name, expected in protocol['predictor_sources'].items():
        path = C.OUT / 'predictor_source' / name
        actual = C.sha(path)
        assert actual == expected, f'Predictor source changed: {name}'
        provenance[str(path.relative_to(C.OUT))] = actual
    for name in ('labels', 'baseline'):
        path = C.ARRAYS / f'{name}.npy'
        actual = C.sha(path)
        assert actual == prepared['array_hashes_verified'][path.name], path.name
        provenance[str(path)] = actual
    ids_parts, augmented_parts, fold_results = [], [], []
    expected_heads = {f'{arm}_seed{seed}.txt' for arm in ('control', 'augmented') for seed in C.SEEDS}
    for fold in range(5):
        folder = C.OUT / f'fold{fold}'
        result = C.read(folder / 'result.json')
        assert result['fold'] == fold and result['held_series'] == 2000
        provenance[f'fold{fold}/result.json'] = C.sha(folder / 'result.json')
        held_path = folder / 'held_ids.npy'
        ids = np.load(held_path, allow_pickle=False)
        expected_ids = np.arange(10000)[np.arange(10000) % 5 == fold]
        assert np.array_equal(ids, expected_ids), f'Fold {fold} held identity differs.'
        provenance[f'fold{fold}/held_ids.npy'] = C.sha(held_path)
        assert set(result['model_hashes']) == expected_heads
        for filename, expected in result['model_hashes'].items():
            path = folder / filename
            actual = C.sha(path)
            assert actual == expected, f'Head changed: fold{fold}/{filename}'
            provenance[f'fold{fold}/{filename}'] = actual
        for inner in (0, 1):
            path = C.predictor_path(fold, inner)
            extraction_path = folder / f'extraction{inner}.json'
            extraction = C.read(extraction_path)
            complete_path = path.parent / 'complete.json'
            complete = C.read(complete_path)
            actual = C.sha(path)
            assert actual == extraction['checkpoint_sha256'] == complete['checkpoint_sha256']
            assert complete['steps'] == 800 and extraction['fit_and_feature_ids_disjoint'] is True
            provenance[str(path.relative_to(C.OUT))] = actual
            provenance[str(extraction_path.relative_to(C.OUT))] = C.sha(extraction_path)
            provenance[str(complete_path.relative_to(C.OUT))] = C.sha(complete_path)
        prediction_path = folder / 'augmented_probability.npy'
        actual = C.sha(prediction_path)
        assert actual == result['prediction_hashes'][prediction_path.name]
        probability = np.load(prediction_path, allow_pickle=False)
        assert probability.shape == (2000, 1000)
        provenance[f'fold{fold}/augmented_probability.npy'] = actual
        ids_parts.append(ids)
        augmented_parts.append(probability)
        fold_results.append(result)
    ids = np.concatenate(ids_parts)
    assert np.array_equal(np.sort(ids), np.arange(10000))
    labels = np.asarray(np.load(C.ARRAYS / 'labels.npy', mmap_mode='r')[ids])
    incumbent = np.asarray(np.load(C.ARRAYS / 'baseline.npy', mmap_mode='r')[ids])
    augmented = np.concatenate(augmented_parts)
    # No conversion to a common dtype occurs before this original expression.
    blend = fixed_blend(incumbent, augmented)
    predictions = {'incumbent': incumbent, 'fixed_blend': blend}
    dtypes = {'incumbent': str(incumbent.dtype), 'augmented': str(augmented.dtype),
              'fixed_blend': str(blend.dtype)}
    return labels, predictions, fold_results, dtypes, provenance


def recompute_point_gate(labels, predictions, fold_results, saved_gate):
    """Verify pooled and per-fold scores from authenticated saved arrays."""
    scores = {name: C.metric(labels, probability) for name, probability in predictions.items()}
    combined = C.read(C.OUT / 'result.json')
    assert abs(scores['incumbent']-combined['incumbent']) < 1e-12
    assert abs(scores['fixed_blend']-combined['fixed_10pct_blends']['augmented']) < 1e-12
    fold_gains = []
    for fold, result in enumerate(fold_results):
        selection = slice(2000*fold, 2000*(fold+1))
        incumbent_score = C.metric(labels[selection], predictions['incumbent'][selection])
        blend_score = C.metric(labels[selection], predictions['fixed_blend'][selection])
        assert abs(incumbent_score-result['incumbent']) < 1e-12
        assert abs(blend_score-result['fixed_10pct_blends']['augmented']) < 1e-12
        delta = blend_score-incumbent_score
        assert abs(delta-saved_gate['reported_fold_gains'][fold]) < 1e-12
        fold_gains.append(float(delta))
    point_gain = scores['fixed_blend']-scores['incumbent']
    assert abs(point_gain-saved_gate['reported_pooled_gain']) < 1e-12
    positive = sum(delta > 0 for delta in fold_gains)
    eligible = point_gain >= MINIMUM_GAIN and positive >= MINIMUM_POSITIVE_FOLDS
    return {'eligible': bool(eligible), 'point_scores': scores, 'point_gain': float(point_gain),
            'fold_gains': fold_gains, 'positive_folds': positive, 'series': len(labels)}


def source_record():
    return {'ensemble_bootstrap.py': C.sha(Path(__file__)),
            'bootstrap.py': C.sha(C.OUT / 'bootstrap.py'),
            'integration_common.py': C.sha(C.OUT / 'integration_common.py'),
            'protocol.json': C.sha(C.OUT / 'protocol.json')}


def main(check_only=False):
    started = time.time()
    checks = validate_weighted_implementation()
    if check_only:
        print(json.dumps({'phase': 'ensemble_bootstrap_checks', 'passed': checks['passed'],
                          'explicit_repeat_replicates': checks['total_explicit_repeat_replicates'],
                          'largest_absolute_difference': checks['largest_absolute_difference']}), flush=True)
        return checks
    protocol = C.ensure_protocol()
    assert protocol['bootstrap'] == {'replicates': REPLICATES, 'seed': SEED}
    gate = saved_point_gate()
    sources = source_record()
    if not gate['eligible']:
        result = {'status': 'skipped', 'heavy_bootstrap_run': False, 'saved_point_gate': gate, 'gate': gate,
                  'source_sha256': sources, 'weighted_implementation_checks': checks,
                  'inputs_authenticated': False, 'reason': gate['reason'],
                  'original_protocol_modified': False, 'alternative_weights_tried': False,
                  'passes_frozen_incumbent_adoption_gate': False,
                  'deployment_adopted': False, 'official_submission': False}
        C.write(RESULT_PATH, result)
        print(json.dumps({key: result[key] for key in ('status', 'heavy_bootstrap_run', 'reason')}), flush=True)
        return result
    labels, predictions, fold_results, dtypes, provenance = authenticated_inputs(protocol)
    point = recompute_point_gate(labels, predictions, fold_results, gate)
    if not point['eligible']:
        result = {'status': 'skipped', 'heavy_bootstrap_run': False, 'saved_point_gate': gate, 'gate': dict(gate, eligible=False),
                  'verified_point_gate': point, 'inputs_authenticated': True,
                  'reason': 'The recomputed incumbent-blend point gate failed.',
                  'inputs_sha256': provenance, 'source_sha256': sources,
                  'weighted_implementation_checks': checks, 'deployment_adopted': False,
                  'official_submission': False, 'alternative_weights_tried': False,
                  'original_protocol_modified': False, 'passes_frozen_incumbent_adoption_gate': False}
        C.write(RESULT_PATH, result)
        return result
    n = len(labels)
    assert n == 10000
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(n, np.full(n, 1./n), size=REPLICATES).astype(np.int32)
    assert np.array_equal(draws.sum(axis=1), np.full(REPLICATES, n))
    counts = np.concatenate([np.ones((1, n), dtype=np.int32), draws])
    scores, pairs, details = weighted_scores(
        labels, predictions, counts,
        progress=lambda item: print(json.dumps(dict(item, comparison='incumbent_fixed_10pct_blend')), flush=True))
    for arm in ARMS:
        assert abs(scores[arm][0]-point['point_scores'][arm]) < 1e-12
    gains = scores['fixed_blend'][1:] - scores['incumbent'][1:]
    lower, upper = np.quantile(gains, [.025, .975], method='linear')
    passed = bool(lower > 0)
    result = {
        'status': 'complete', 'comparison': 'incumbent_vs_fixed_90pct_incumbent_10pct_augmented',
        'heavy_bootstrap_run': True, 'series': n, 'replicates': REPLICATES, 'seed': SEED,
        'fixed_blend_weight': .1, 'saved_point_gate': gate, 'verified_point_gate': point,
        'gate': dict(gate, verified_point_gain=point['point_gain'], positive_folds=point['positive_folds']),
        'ci_95': [float(lower), float(upper)], 'ci_method': 'Paired percentile interval with linear 2.5% and 97.5% quantiles.',
        'bootstrap_mean_gain': float(gains.mean()), 'bootstrap_std_gain': float(gains.std(ddof=1)),
        'bootstrap_fraction_positive': float((gains > 0).mean()), 'bootstrap_95_lower_positive': passed,
        'passes_incumbent_blend_gate': passed, 'passes_frozen_incumbent_adoption_gate': passed,
        'decision': 'passed_pending_fullfit_and_complete_runtime_validation' if passed else 'failed_incumbent_blend_bootstrap_gate',
        'eligible_for_fullfit_and_runtime_work': passed, 'deployment_adopted': False,
        'official_submission': False, 'official_score': False,
        'original_protocol_modified': False, 'alternative_weights_tried': False,
        'arithmetic': '0.9 * incumbent + 0.1 * augmented, evaluated in NumPy with the original saved input dtypes before any conversion.',
        'prediction_dtypes': dtypes,
        'resampling': 'Uniform whole-series multinomial sampling over the 10,000 held series across five folds; identical counts for both arms and every tick.',
        'metric': 'Pair-weighted within-time weighted Mann-Whitney TS-AUC; dead suffixes excluded and tied cross-class scores receive half credit.',
        'uncertainty_scope': 'Conditional on fixed fitted models and reused development data. Models are not retrained within draws; this is not independent-test or deployment certification.',
        'development_reused': True, 'independent_holdout': False,
        'weighted_full_data_identity_matches_unweighted_metric': True,
        'minimum_bootstrap_pair_mass': int(pairs[1:].min()), 'maximum_bootstrap_pair_mass': int(pairs[1:].max()),
        'draw_counts_sha256': C.array_sha(draws), 'bootstrap_gain': gains.tolist(),
        'bootstrap_scores': {arm: scores[arm][1:].tolist() for arm in ARMS},
        'counting_details': details, 'inputs_authenticated': True,
        'inputs_sha256': provenance, 'source_sha256': sources,
        'weighted_implementation_checks': checks, 'numpy_version': np.__version__,
        'seconds': time.time()-started,
    }
    C.write(RESULT_PATH, result)
    print(json.dumps({key: result[key] for key in ('status', 'ci_95', 'decision', 'seconds')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Run tiny shared implementation checks only; do not write an ensemble result or run the real bootstrap.')
    args = parser.parse_args()
    main(check_only=args.check_only)
