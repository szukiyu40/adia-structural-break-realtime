"""Assemble the selected candidate from authenticated completed training."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main(parent, fit, out):
    source = Path(__file__).resolve().parent
    complete = read(fit / 'complete.json')
    manifest = read(fit / 'manifest.json')
    assert complete['manifest_sha256'] == sha(fit / 'manifest.json')
    assert manifest['training_series'] == 10000
    assert manifest['full_fit_source_sha256'] == sha(source / 'full_fit_candidate.py')
    for name, digest in complete['artifacts_sha256'].items():
        assert sha(fit / name) == digest, name
    serving = manifest['promotion_evidence']['serving']
    assert serving['family'] in ('probability_blend', 'logit_contrast')
    parent_manifest = read(parent / 'package_manifest.json')
    for name, info in parent_manifest.items():
        assert sha(parent / name) == info['sha256'], name
    assert not out.exists(), 'Refuse to overwrite an existing package.'
    assert shutil.disk_usage(out.parent).free > 1_300_000_000
    ignore = shutil.ignore_patterns('.crunchdao', '__pycache__', '*.pyc', 'data', 'prediction', '.cache')
    shutil.copytree(parent, out, ignore=ignore, symlinks=False)
    (out / 'main.py').rename(out / 'incumbent_main.py')
    shutil.copy2(source / 'candidate_main.py', out / 'main.py')
    shutil.copytree(source / 'serving', out / 'feature_serving', ignore=ignore)
    model_root = out / 'resources' / 'feature_integration'
    model_root.mkdir()
    model_hashes = {}
    for name, digest in complete['artifacts_sha256'].items():
        target = model_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fit / name, target)
        assert sha(target) == digest
        if name.endswith('.pt') or name.endswith('.txt'):
            model_hashes[name] = digest
    write(model_root / 'metadata.json', {'serving': serving, 'model_sha256': model_hashes,
          'full_fit_manifest_sha256': sha(fit / 'manifest.json')})
    training = out / 'feature_training_source'
    training.mkdir()
    for name in ('integration_common.py', 'prepare.py', 'extract.py', 'fit.py', 'run.py',
                 'full_fit.py', 'full_fit_candidate.py', 'candidate_promotion.json', 'protocol.json',
                 'prepared.json', 'bootstrap.json', 'ensemble_bootstrap.json', 'result.json',
                 'bootstrap.py', 'ensemble_bootstrap.py', 'verify.py', 'report.py', 'resume.py',
                 'checks.json', 'bootstrap_checks.json', 'bootstrap_crosscheck.json',
                 'CANDIDATE_PROMOTION.md', 'candidate_main.py', 'build_submission.py'):
        if (source / name).exists():
            shutil.copy2(source / name, training / name)
    shutil.copytree(source / 'predictor_source', training / 'predictor_source', ignore=ignore)
    shutil.copytree(source / 'training_input_source', training / 'training_input_source', ignore=ignore)
    for name in ('complete.json', 'manifest.json', 'heads_complete.json', 'extraction_complete.json'):
        shutil.copy2(fit / name, training / ('full_fit_' + name))
    (training / 'README.md').write_text(
        '# Feature model training source\n\n'
        'These files preserve the exact sources used for the feature experiment and full fit. '
        'The archived integration_common.py contains original workspace paths; configure '
        'ARRAYS, MANIFEST, RECIPE, PYTHON and the local output/scratch roots to reproduce '
        'in another workspace. protocol.json records the complete tree parameters and '
        'source hashes. predictor_source contains the normal-history predictor and feature code.\n\n'
        'training_input_source archives the exact raw-Parquet preprocessing and original '
        'tree recipe. Its preprocessing also reconstructs the incumbent OOF predictions '
        'used for validation; final feature-model training itself consumes only the '
        'six non-baseline arrays. The V4 streaming feature implementation is included '
        'under ../feature_serving/legacy and can reconstruct the causal feature cache.\n\n'
        'Inputs: organizer training data transformed into values, innovations, hist_lengths, '
        'lengths, static and labels arrays, plus the causal 422-column V4 feature cache. '
        'prepared.json binds the original input hashes and sampled training rows. '
        'The first half of each known history fits an AR12 model; predictor training uses '
        'only later known-history observations. Two predictors each use a disjoint half '
        'of the 10,000 histories, 800 updates each. Tree training uses the opposite '
        'predictor for each series, 40 logarithmically sampled online ticks at most, '
        'and equal total weight per series. Three seeds each fit 500 trees.\n\n'
        'full_fit_candidate.py orchestrates the exact training stages and preserves RNG '
        'states and resumable checkpoints. candidate_promotion.json records the decision '
        'to prepare this candidate; it does not change the earlier statistical tests. '
        'blend_crossfit contains the scalar-fitting source and saved selection evidence. '
        'The final scalar is trained on OOF predictions after choosing the family; '
        'its in-sample score is never presented as validation.\n')
    blend_source = source.parent / 'blend_crossfit'
    blend_evidence = training / 'blend_crossfit'
    blend_evidence.mkdir()
    for name in ('run.py', 'README.md', 'selection_policy.json', 'protocol.json', 'result.json',
                 'final_coefficient.json', 'coefficients.json', 'bootstrap.json',
                 'synthetic_checks.json', 'RESULTS.md'):
        if (blend_source / name).exists():
            shutil.copy2(blend_source / name, blend_evidence / name)
    requirements = ['numpy', 'lightgbm', 'xgboost', 'scipy', 'pandas', 'pyarrow', 'torch',
                    'chronos-forecasting', 'transformers', 'accelerate', 'scikit-learn', 'threadpoolctl']
    (out / 'requirements.txt').write_text(''.join(f'{name}=={version(name)}\n' for name in requirements))
    (out / 'README.md').write_text(
        '# Feature integration candidate\n\n'
        f'Selected formula: {serving["family"]}; coefficient {serving["coefficient"]:.12g}.\n\n'
        'The incumbent and added detector consume only history and revealed observations. '
        'Added detector outputs refresh every four observations; blending uses the current '
        'incumbent output at every observation. All models are local.\n\n'
        'The added model was fitted on all 10,000 training series after a saved, cross-fitted '
        'blend comparison. The comparison reuses development data and is not an independent '
        'holdout. See feature_training_source for scores, uncertainty and training provenance. '
        'Earlier statistical gate results remain unchanged.\n\n'
        'This is a submission candidate. Local SDK validation is recorded separately; '
        'no official submission is performed by this builder.\n')
    write(out / 'assembly.json', {'selected': serving,
          'parent_manifest_sha256': sha(parent / 'package_manifest.json'),
          'fit_complete_sha256': sha(fit / 'complete.json'),
          'builder_sha256': sha(Path(__file__)), 'official_submitted': False})
    assert not any(p.is_symlink() for p in out.rglob('*'))
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p != out / 'package_manifest.json')
    inventory = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}
    write(out / 'package_manifest.json', inventory)
    print(json.dumps({'files': len(inventory), 'bytes': sum(p['bytes'] for p in inventory.values()),
                      'out': str(out), 'official_submitted': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.parent.resolve(), args.fit.resolve(), args.out.resolve())
