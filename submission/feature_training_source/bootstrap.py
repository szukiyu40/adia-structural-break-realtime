"""Conditional, paired whole-series bootstrap of the frozen five-fold comparison.

Each draw samples 10,000 whole series with replacement. Per-time AUC is computed
with its positive-negative pair mass, including exact half credit for ties. The
same multiplicities are used for both prediction arms and every online tick.
Only negative scores are sorted, once per arm and tick, for all 300 draws.
"""
from integration_common import *
import argparse
import time


REPLICATES = 300
BOOTSTRAP_SEED = 919703
ARMS = ('control', 'augmented')


def weighted_scores(labels, predictions, counts, progress=None):
    """Return pair-weighted TS-AUC for every row of integer series counts.

    A positive observation's contribution is its multiplicity times the total
    negative multiplicity below its score, plus half the tied multiplicity.
    Left/right search boundaries calculate twice this contribution exactly.
    Alive series and labels are fixed per tick; absent series contribute zero.
    """
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    assert labels.ndim == 2 and counts.ndim == 2
    assert counts.shape[1] == labels.shape[0] and len(counts) > 0
    assert np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all()
    # Counts/cumulative counts fit int32; total pair sums fit int64 comfortably.
    assert np.max(counts.sum(axis=1, dtype=np.int64)) <= 1_000_000
    assert labels.shape[1] <= 100_000
    assert np.isin(labels, (-1, 0, 1)).all()
    counts = counts.astype(np.int32, copy=False)
    valid = labels >= 0
    for name, probability in predictions.items():
        assert probability.shape == labels.shape, name
        assert np.isfinite(probability[valid]).all(), name
    assert predictions
    doubled_wins = {name: np.zeros(len(counts), dtype=np.int64) for name in predictions}
    pairs = np.zeros(len(counts), dtype=np.int64)
    tied_pairs = {name: 0 for name in predictions}
    useful_ticks = 0
    started = time.time()
    for tick in range(labels.shape[1]):
        positive = np.flatnonzero(labels[:, tick] == 1)
        negative = np.flatnonzero(labels[:, tick] == 0)
        if len(positive) and len(negative):
            useful_ticks += 1
            positive_count = counts[:, positive]
            positive_mass = positive_count.sum(axis=1, dtype=np.int64)
            for arm_number, (name, probability) in enumerate(predictions.items()):
                negative_order = negative[np.argsort(probability[negative, tick], kind='stable')]
                negative_scores = probability[negative_order, tick]
                positive_scores = probability[positive, tick]
                left = np.searchsorted(negative_scores, positive_scores, side='left')
                right = np.searchsorted(negative_scores, positive_scores, side='right')
                tied_pairs[name] += int((right-left).sum(dtype=np.int64))
                cumulative = np.zeros((len(counts), len(negative)+1), dtype=np.int32)
                np.cumsum(counts[:, negative_order], axis=1, dtype=np.int32,
                          out=cumulative[:, 1:])
                if arm_number == 0:
                    pairs += positive_mass * cumulative[:, -1].astype(np.int64)
                twice_negative_below = cumulative[:, left] + cumulative[:, right]
                doubled_wins[name] += np.einsum(
                    'ij,ij->i', positive_count, twice_negative_below,
                    dtype=np.int64, optimize=False)
        if progress is not None and ((tick+1) % 100 == 0 or tick+1 == labels.shape[1]):
            progress({'phase': 'paired_series_bootstrap', 'ticks_done': tick+1,
                      'total_ticks': labels.shape[1], 'replicates_including_identity': len(counts),
                      'seconds': time.time()-started})
    scores = {}
    for name, wins in doubled_wins.items():
        assert (wins >= 0).all() and (wins <= 2*pairs).all()
        score = np.full(len(counts), .5, dtype=np.float64)
        np.divide(wins, 2*pairs, out=score, where=pairs > 0)
        scores[name] = score
    return scores, pairs, {'useful_ticks': useful_ticks, 'unweighted_cross_class_tied_pairs': tied_pairs}


def _explicit_sklearn_metric(labels, probability):
    """Independent, explicit-repetition oracle for the tiny validation cases."""
    from sklearn.metrics import roc_auc_score
    wins = 0.
    pairs = 0
    for tick in range(labels.shape[1]):
        alive = labels[:, tick] >= 0
        y = labels[alive, tick]
        positive = int((y == 1).sum())
        negative = int((y == 0).sum())
        mass = positive * negative
        if mass:
            wins += float(roc_auc_score(y, probability[alive, tick])) * mass
            pairs += mass
    return wins/pairs if pairs else .5


def validate_weighted_implementation():
    """Compare exact weighted results with physically repeated whole series."""
    ragged = np.array([
        [0, 0, 1, 1, -1],
        [0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0],
        [1, 1, -1, -1, -1],
    ], dtype=np.int8)
    control = np.array([
        [.2, .5, .5, .8, np.nan],
        [.2, .5, .7, np.nan, np.nan],
        [.2, .1, .5, .8, .9],
        [.2, .5, np.nan, np.nan, np.nan],
    ])
    augmented = np.array([
        [.3, .2, .8, .8, np.nan],
        [.1, .7, .8, np.nan, np.nan],
        [.3, .2, .2, .8, .1],
        [.7, .7, np.nan, np.nan, np.nan],
    ])
    fixed_counts = np.array([
        [1, 1, 1, 1], [2, 0, 1, 1], [0, 3, 0, 1],
        [0, 0, 4, 0], [4, 0, 0, 0], [0, 0, 0, 0],
    ], dtype=np.int32)
    cases = [('fixed_ragged_tied_and_zero_pairs', ragged,
              {'control': control, 'augmented': augmented}, fixed_counts)]
    rng = np.random.default_rng(428109)
    for case_number in range(4):
        n, ticks = 7, 6
        lengths = np.array([6, 5, 4, 3, 2, 1, 6])
        tau = rng.integers(0, ticks+2, n)
        yy = np.where(np.arange(ticks)[None, :] < lengths[:, None],
                      np.arange(ticks)[None, :] >= tau[:, None], -1).astype(np.int8)
        pp = {name: rng.integers(0, 4, (n, ticks)).astype(np.float64)/3 for name in ARMS}
        for probability in pp.values():
            probability[yy < 0] = np.nan
        multiplicities = rng.multinomial(n, np.full(n, 1./n), size=17).astype(np.int32)
        cases.append((f'random_ragged_tied_{case_number}', yy, pp, multiplicities))
    checks = []
    largest_difference = 0.
    for name, labels, probabilities, multiplicities in cases:
        scores, pairs, details = weighted_scores(labels, probabilities, multiplicities)
        for replicate, count in enumerate(multiplicities):
            repeated = np.repeat(np.arange(len(labels)), count)
            repeated_labels = labels[repeated]
            explicit_pairs = sum(int((repeated_labels[:, t] == 0).sum()) *
                                 int((repeated_labels[:, t] == 1).sum())
                                 for t in range(labels.shape[1]))
            assert pairs[replicate] == explicit_pairs
            for arm, probability in probabilities.items():
                expected = metric(repeated_labels, probability[repeated])
                sklearn_expected = _explicit_sklearn_metric(repeated_labels, probability[repeated])
                error = max(abs(scores[arm][replicate]-expected),
                            abs(scores[arm][replicate]-sklearn_expected))
                assert error < 2e-14, (name, replicate, arm, error)
                largest_difference = max(largest_difference, error)
            expected_delta = (metric(repeated_labels, probabilities['augmented'][repeated]) -
                              metric(repeated_labels, probabilities['control'][repeated]))
            assert abs(scores['augmented'][replicate]-scores['control'][replicate]-expected_delta) < 2e-14
        checks.append({'case': name, 'replicates': len(multiplicities),
                       'whole_series_repeat_match': True, 'sklearn_match': True,
                       'pair_mass_match': True, **details})
    # All cross-class pairs tied must receive exactly half credit for every draw.
    tied_labels = np.array([[0, 0], [1, 1], [0, -1]], dtype=np.int8)
    tied = np.full(tied_labels.shape, .37)
    tied_scores, _, _ = weighted_scores(tied_labels, {'control': tied, 'augmented': tied},
                                        np.array([[1, 1, 1], [5, 2, 3], [0, 2, 0]], dtype=np.int32))
    assert np.array_equal(tied_scores['control'], np.full(3, .5))
    assert np.array_equal(tied_scores['augmented'], tied_scores['control'])
    return {'passed': True, 'implementation_sha256': sha(Path(__file__)),
            'total_explicit_repeat_replicates': sum(x['replicates'] for x in checks),
            'arms_per_replicate': 2, 'largest_absolute_difference': largest_difference,
            'all_ties_half_credit': True, 'zero_pair_score_is_half': True,
            'invalid_suffix_predictions_ignored': True, 'cases': checks}


def gate_status():
    completed = [fold for fold in range(5) if (OUT / f'fold{fold}/result.json').exists()]
    if completed != list(range(5)):
        return {'eligible': False, 'reason': 'Five completed folds required.', 'completed_folds': completed}
    if not (OUT / 'result.json').exists():
        return {'eligible': False, 'reason': 'Combined five-fold report required.', 'completed_folds': completed}
    combined = read(OUT / 'result.json')
    positive = sum(read(OUT / f'fold{fold}/result.json')['primary_delta'] > 0 for fold in completed)
    eligible = (combined['completed_folds'] == 5 and combined['series'] == 10000 and
                positive >= 4 and combined['primary_delta'] >= .001)
    return {'eligible': bool(eligible), 'reason': 'Fixed full-five-fold gate passed.' if eligible else
            'Fixed full-five-fold delta or positive-fold gate failed.',
            'completed_folds': completed, 'positive_folds': positive,
            'reported_primary_delta': combined['primary_delta'], 'combined_report_sha256': sha(OUT / 'result.json')}


def load_predictions():
    ids_parts = []
    parts = {name: [] for name in ARMS}
    provenance = {}
    for fold in range(5):
        folder = OUT / f'fold{fold}'
        result = read(folder / 'result.json')
        ids = np.load(folder / 'held_ids.npy', allow_pickle=False)
        assert np.array_equal(ids, np.arange(10000)[np.arange(10000) % 5 == fold])
        ids_parts.append(ids)
        provenance[f'fold{fold}/held_ids.npy'] = sha(folder / 'held_ids.npy')
        provenance[f'fold{fold}/result.json'] = sha(folder / 'result.json')
        for name in ARMS:
            path = folder / f'{name}_probability.npy'
            digest = sha(path)
            assert digest == result['prediction_hashes'][path.name]
            probability = np.load(path, allow_pickle=False)
            assert probability.shape == (2000, 1000)
            parts[name].append(probability)
            provenance[f'fold{fold}/{path.name}'] = digest
    ids = np.concatenate(ids_parts)
    assert np.array_equal(np.sort(ids), np.arange(10000))
    label_path = ARRAYS / 'labels.npy'
    digest = sha(label_path)
    assert digest == read(OUT / 'prepared.json')['array_hashes_verified']['labels.npy']
    labels = np.asarray(np.load(label_path, mmap_mode='r')[ids])
    probabilities = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    provenance[str(label_path)] = digest
    return labels, probabilities, provenance


def main(check_only=False):
    started = time.time()
    checks = validate_weighted_implementation()
    write(OUT / 'checks.json', checks)
    if check_only:
        print(json.dumps({'phase': 'bootstrap_checks', 'passed': True,
                          'replicates': checks['total_explicit_repeat_replicates'],
                          'largest_absolute_difference': checks['largest_absolute_difference']}), flush=True)
        return checks
    protocol = ensure_protocol()
    assert protocol['bootstrap'] == {'replicates': REPLICATES, 'seed': BOOTSTRAP_SEED}
    gate = gate_status()
    if not gate['eligible']:
        result = {'status': 'skipped', 'heavy_compute_run': False, 'gate': gate,
                  'checks_sha256': sha(OUT / 'checks.json'),
                  'implementation_sha256': sha(Path(__file__))}
        write(OUT / 'bootstrap.json', result)
        print(json.dumps(result), flush=True)
        return result
    labels, predictions, provenance = load_predictions()
    baseline = {name: metric(labels, prediction) for name, prediction in predictions.items()}
    combined = read(OUT / 'result.json')
    for name in ARMS:
        assert abs(baseline[name]-combined['scores'][name]) < 1e-12
    point_delta = baseline['augmented'] - baseline['control']
    assert abs(point_delta-gate['reported_primary_delta']) < 1e-12 and point_delta >= .001
    n = len(labels)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(n, np.full(n, 1./n), size=REPLICATES).astype(np.int32)
    assert np.array_equal(draws.sum(axis=1), np.full(REPLICATES, n))
    # One identity row validates the weighted implementation on the full dataset.
    counts = np.concatenate([np.ones((1, n), dtype=np.int32), draws])
    scores, pair_mass, details = weighted_scores(
        labels, predictions, counts, progress=lambda message: print(json.dumps(message), flush=True))
    for name in ARMS:
        assert abs(scores[name][0]-baseline[name]) < 1e-12
    delta = scores['augmented'][1:] - scores['control'][1:]
    lower, upper = np.quantile(delta, [.025, .975], method='linear')
    passed = bool(lower > 0)
    result = {
        'status': 'complete', 'heavy_compute_run': True, 'series': n, 'replicates': REPLICATES,
        'seed': BOOTSTRAP_SEED, 'gate': gate, 'point_scores': baseline, 'point_delta': point_delta,
        'resampling': 'Uniform multinomial sampling of whole series, globally across the five held folds; identical counts for both arms and all ticks.',
        'metric': 'Sum of within-tick weighted Mann-Whitney wins / sum of positive-negative pair masses; tied scores receive half credit.',
        'ci_method': 'Paired percentile interval, 2.5% and 97.5% quantiles with linear interpolation.',
        'ci_95': [float(lower), float(upper)],
        'bootstrap_mean_delta': float(delta.mean()), 'bootstrap_std_delta': float(delta.std(ddof=1)),
        'bootstrap_fraction_positive': float((delta > 0).mean()),
        'bootstrap_95_lower_positive': passed, 'passes_frozen_primary_adoption_gate': passed,
        'decision': 'passed_primary_gate_pending_incumbent_and_sequential_validation' if passed else 'failed_bootstrap_gate',
        'weighted_full_data_identity_matches_unweighted_metric': True,
        'minimum_bootstrap_pair_mass': int(pair_mass[1:].min()),
        'maximum_bootstrap_pair_mass': int(pair_mass[1:].max()),
        'draw_counts_sha256': array_sha(draws), 'bootstrap_delta': delta.tolist(),
        'bootstrap_scores': {name: scores[name][1:].tolist() for name in ARMS},
        'counting_details': details, 'inputs_sha256': provenance,
        'checks_sha256': sha(OUT / 'checks.json'), 'implementation_sha256': sha(Path(__file__)),
        'frozen_protocol_sha256': sha(OUT / 'protocol.json'), 'numpy_version': np.__version__,
        'development_reused': True, 'independent_holdout': False,
        'uncertainty_scope': 'Conditional on the fixed fitted models and reused development series; models are not retrained within bootstrap draws.',
        'deployment_adopted': False, 'official_submission': False, 'seconds': time.time()-started,
    }
    write(OUT / 'bootstrap.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'point_delta', 'ci_95', 'decision', 'seconds')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Run tiny validation cases without any full-data bootstrap.')
    args = parser.parse_args()
    main(check_only=args.check_only)
