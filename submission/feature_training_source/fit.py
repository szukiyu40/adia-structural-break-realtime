"""Matched tree heads; neural outputs are averaged only on excluded series."""
from integration_common import *
import argparse
import gc
import time
import lightgbm as lgb

def fit(fold):
    ensure_protocol(); out = OUT / f'fold{fold}'; out.mkdir(exist_ok=True)
    if (out / 'result.json').exists():
        return read(out / 'result.json')
    start = time.time(); d = arrays(); manifest = read(MANIFEST); scratch = WORK / f'fold{fold}'
    row_ids = np.load(WORK / 'sample_ids.npy'); row_ticks = np.load(WORK / 'sample_ticks.npy')
    selected = row_ids % 5 != fold; ids = row_ids[selected]; ticks = row_ticks[selected]
    held = np.arange(10000)[np.arange(10000) % 5 == fold]
    xbase = np.asarray(np.load(WORK / 'sampled_base.npy', mmap_mode='r')[selected])
    xlearned = np.empty((len(ids), 144), dtype='f4'); assigned = np.zeros(len(ids), dtype=bool)
    encoded = ids.astype('i8') * 1000 + ticks
    for inner in (0, 1):
        identity = np.load(scratch / f'train{inner}.npz'); ii = identity['ids']; tt = identity['ticks']
        assert np.all(ii % 5 != fold) and np.all((ii // 5) % 2 != inner)
        index = np.searchsorted(encoded, ii.astype('i8') * 1000 + tt)
        assert np.array_equal(encoded[index], ii.astype('i8') * 1000 + tt) and not assigned[index].any()
        xlearned[index] = np.load(scratch / f'train{inner}_x.npy', mmap_mode='r'); assigned[index] = True
    assert assigned.all() and xbase.shape[1] == 422 and xlearned.shape[1] == 144
    target = np.asarray(d['labels'][ids, ticks]); assert np.isin(target, [0, 1]).all()
    unique, counts = np.unique(ids, return_counts=True); assert len(unique) == 8000
    weight = np.repeat(1./counts, counts); weight /= weight.mean()
    assert np.ptp(np.add.reduceat(weight, np.r_[0, counts.cumsum()[:-1]])) < 1e-10
    weight = weight.astype('f4')
    np.savez_compressed(out / 'training_plan.npz', ids=ids, ticks=ticks, target=target, weight=weight)
    models = {}; timings = {}
    for arm in ('control', 'augmented'):
        x = xbase if arm == 'control' else np.concatenate([xbase, xlearned], axis=1)
        arm_models = []; t0 = time.time()
        for seed in SEEDS:
            path = out / f'{arm}_seed{seed}.txt'
            if path.exists():
                booster = lgb.Booster(model_file=str(path))
                assert booster.num_feature() == x.shape[1] and booster.num_trees() == 500
            else:
                model = lgb.LGBMClassifier(**parameters(seed)); model.fit(x, target, sample_weight=weight)
                booster = model.booster_; booster.save_model(str(path))
                probe = x[np.linspace(0, len(x)-1, 17).astype(int)]
                assert np.array_equal(booster.predict(probe, num_threads=THREADS), lgb.Booster(model_file=str(path)).predict(probe, num_threads=THREADS))
                del model
            arm_models.append(booster)
            print(json.dumps({'phase': 'fit', 'fold': fold, 'arm': arm, 'seed': seed, 'seconds': time.time()-start}), flush=True)
        models[arm] = arm_models; timings[arm] = time.time()-t0
        if arm == 'augmented':
            del x
        gc.collect()
    del xbase, xlearned; gc.collect()
    pred = {arm: np.zeros((len(held), 1000), dtype='f8') for arm in models}
    lookup = np.full(10000, -1, dtype='i4'); lookup[held] = np.arange(len(held))
    parity = []
    for inner in (0, 1):
        identity = np.load(scratch / f'held{inner}.npz'); ii = identity['ids']; tt = identity['ticks']
        assert np.array_equal(np.unique(ii), held) and np.all(tt % INFER_STRIDE == 0)
        neural = np.load(scratch / f'held{inner}_x.npy', mmap_mode='r')
        for lo in range(0, len(held), 100):
            sids = held[lo:lo+100]
            a = int(np.searchsorted(ii, sids[0])); b = int(np.searchsorted(ii, sids[-1], side='right'))
            base = np.concatenate([np.load(manifest[s]['path'], mmap_mode='r')[::INFER_STRIDE, :422] for s in sids])
            assert len(base) == b-a
            for arm in models:
                if arm == 'control' and inner == 1:
                    continue
                x = base if arm == 'control' else np.concatenate([base, neural[a:b]], axis=1)
                pp = np.mean([m.predict(x, num_threads=THREADS) for m in models[arm]], axis=0)
                if lo == 0:
                    scalar = np.mean([[m.predict(row[None], num_threads=1)[0] for row in x[:7]] for m in models[arm]], axis=0)
                    error = float(np.max(np.abs(scalar-pp[:7]))); assert error == 0
                    parity.append({'arm': arm, 'inner': inner, 'batch_scalar_max_error': error})
                for delta in range(INFER_STRIDE):
                    alive = tt[a:b]+delta < d['lengths'][ii[a:b]]
                    pred[arm][lookup[ii[a:b][alive]], tt[a:b][alive]+delta] += (1. if arm == 'control' else .5)*pp[alive]
        del neural
    yy = np.asarray(d['labels'][held]); incumbent = np.asarray(d['baseline'][held])
    scores = {arm: metric(yy, p) for arm, p in pred.items()}
    blends = {arm: metric(yy, .9*incumbent + .1*p) for arm, p in pred.items()}
    for arm, p in pred.items():
        assert np.isfinite(p).all() and ((p[yy >= 0] > 0) & (p[yy >= 0] < 1)).all()
        np.save(out / f'{arm}_probability.npy', p)
    np.save(out / 'held_ids.npy', held)
    result = {'fold': fold, 'scores': scores, 'primary_delta': scores['augmented']-scores['control'],
              'incumbent': metric(yy, incumbent), 'fixed_10pct_blends': blends,
              'augmented_blend_delta_vs_incumbent': blends['augmented']-metric(yy, incumbent),
              'training_rows': len(ids), 'training_series': 8000, 'held_series': len(held),
              'fit_seconds': timings, 'total_seconds': time.time()-start, 'head_batch_scalar_checks': parity,
              'model_hashes': {p.name: sha(p) for p in out.glob('*_seed*.txt')},
              'prediction_hashes': {p.name: sha(p) for p in out.glob('*_probability.npy')},
              'official_score': False, 'development_reused': True, 'neural_retrained': fold > 1}
    write(out / 'result.json', result); print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--fold', type=int, required=True); a = p.parse_args(); fit(a.fold)
