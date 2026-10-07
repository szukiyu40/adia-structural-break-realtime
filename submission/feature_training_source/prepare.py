"""Authenticate inputs, snapshot dependencies, and freeze the comparison."""
from integration_common import *
import shutil
import time
from datetime import datetime, timezone

def prepare():
    if (OUT / 'prepared.json').exists():
        ensure_protocol()
        return
    start = time.time(); WORK.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / 'predictor_source'; snapshot.mkdir(exist_ok=True)
    sources = ('common.py', 'network.py', 'features.py', 'train_predictor.py')
    for name in sources:
        shutil.copy2(PRED_SOURCE / name, snapshot / name)
    copied = {}
    for fold in (0, 1):
        for inner in (0, 1):
            old = PRED_SOURCE / f'fold{fold}/predictor{inner}'
            dest = predictor_path(fold, inner).parent; dest.mkdir(parents=True, exist_ok=True)
            for name in ('latest.pt', 'protocol.json', 'complete.json'):
                shutil.copy2(old / name, dest / name)
            copied[str(dest.relative_to(OUT))] = sha(dest / 'latest.pt')
    script_names = ('integration_common.py', 'prepare.py', 'extract.py', 'fit.py', 'run.py', 'verify.py', 'report.py')
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'primary_question': 'Does adding 144 cross-fitted neural features improve the same 422-feature detector?',
        'outer_split': 'id % 5', 'inner_predictor_fit_split': '(id // 5) % 2',
        'train_ticks': 'Original rounded logarithmic grid, at most 40 ticks per series.',
        'train_weights': 'Equal total weight per series; global mean normalized to one.',
        'held_stride': INFER_STRIDE, 'interpolation': 'Previous value held, including all alive steps in metric.',
        'arms': {'control': 422, 'augmented': 566},
        'neural_columns': 'Existing feature_matrix output columns 101:245 only.',
        'training_neural_view': 'Only predictor that excludes the training series.',
        'held_neural_views': 'Predict each excluded predictor view separately, then average probabilities.',
        'head_seeds': list(SEEDS), 'parameters_by_seed': {str(s): parameters(s) for s in SEEDS},
        'neural_fit_steps': 800, 'neural_fit_batch': 64,
        'screen': 'Evaluate folds in fixed order; stop as soon as two non-positive primary fold deltas make 4/5 positives impossible.',
        'primary_adoption_gate': {'full_10000_delta_at_least': .001, 'positive_folds_at_least': 4, 'paired_series_bootstrap_95_lower_positive': True},
        'bootstrap': {'replicates': 300, 'seed': 919703},
        'ensemble_diagnostic': 'Fixed 10% blend with existing GPU+Chronos probabilities; descriptive, not the primary question.',
        'deployment_adoption': 'Requires subsequent improvement of the full incumbent plus sequential inference certification.',
        'development_reused': True, 'independent_holdout': False, 'official_submission': False,
        'experiment_sources': {name: sha(OUT / name) for name in script_names},
        'predictor_sources': {name: sha(snapshot / name) for name in sources},
        'reused_predictors': copied,
        'feature_manifest_sha256': sha(MANIFEST), 'original_head_recipe_sha256': sha(RECIPE),
        'resource_policy': 'One fold at a time; remove only this experiment generated feature scratch after saved predictions. Preserve weights and outputs.',
    }
    write(OUT / 'protocol.json', protocol)
    d = arrays(); manifest = read(MANIFEST)
    assert [r['id'] for r in manifest] == list(range(10000))
    assert np.array_equal([r['rows'] for r in manifest], d['lengths'])
    provenance = read(PRED_SOURCE / 'data_manifest.json')
    verified = {}
    for name in d:
        filename = f'{name}.npy'; actual = sha(ARRAYS / filename)
        assert actual == provenance['arrays'][filename], filename
        verified[filename] = actual
    ticks = [log_steps(int(n)) for n in d['lengths']]
    counts = np.array([len(t) for t in ticks], dtype='i4'); offsets = np.r_[0, counts.cumsum()]
    base = np.lib.format.open_memmap(WORK / 'sampled_base.npy', mode='w+', dtype='f4', shape=(int(offsets[-1]), 422))
    for sid, row in enumerate(manifest):
        assert sha(row['path']) == row['sha256'], f'Feature identity failed for {sid}'
        feature = np.load(row['path'], mmap_mode='r')
        assert feature.shape == (int(d['lengths'][sid]), row['stored_columns'])
        base[offsets[sid]:offsets[sid+1]] = feature[ticks[sid], :422]
        if sid % 2000 == 0:
            print(json.dumps({'phase': 'prepare', 'verified_series': sid, 'seconds': time.time()-start}), flush=True)
    base.flush(); assert np.isfinite(base).all()
    np.save(WORK / 'sample_ids.npy', np.repeat(np.arange(10000, dtype='i4'), counts))
    np.save(WORK / 'sample_ticks.npy', np.concatenate(ticks).astype('i4'))
    np.save(WORK / 'sample_offsets.npy', offsets)
    write(OUT / 'prepared.json', {
        'series': 10000, 'online_points': int(d['lengths'].sum()), 'sampled_rows': int(offsets[-1]),
        'array_hashes_verified': verified, 'all_10000_feature_hashes_verified': True,
        'sampled_base_sha256': sha(WORK / 'sampled_base.npy'),
        'sample_ids_sha256': sha(WORK / 'sample_ids.npy'), 'sample_ticks_sha256': sha(WORK / 'sample_ticks.npy'),
        'seconds': time.time()-start,
    })
    print('Inputs authenticated and fixed comparison prepared.', flush=True)

if __name__ == '__main__':
    prepare()
