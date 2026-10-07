"""Focused checks of arbitrary tick joins, fold isolation and causal features."""
from integration_common import *
import torch

def verify():
    ensure_protocol(); network, features = predictor_modules(); torch.set_num_threads(2)
    d = arrays(); manifest = read(MANIFEST)
    rowids = np.load(WORK / 'sample_ids.npy'); ticks = np.load(WORK / 'sample_ticks.npy')
    base = np.load(WORK / 'sampled_base.npy', mmap_mode='r')
    assert np.all(np.diff(rowids.astype('i8')*1000+ticks) > 0)
    checks = []
    for fold in (0, 1):
        for inner in (0, 1):
            ck = torch.load(predictor_path(fold, inner), map_location='cpu', weights_only=False)
            fitids = np.array(ck['fit_ids']); ids = np.arange(10000)
            expected = ids[(ids % 5 != fold) & ((ids // 5) % 2 == inner)]
            assert np.array_equal(fitids, expected)
            m = network.Predictor().eval(); m.load_state_dict(ck['model'])
            sid = fold; h = int(d['hist_lengths'][sid]); n = int(d['lengths'][sid]); cal = min(512, h//2-64)
            v = np.array(d['values'][sid, h-cal-64:h+n]); r = np.array(d['innovations'][sid, h-cal-64:h+n]); static = np.array(d['static'][sid])
            bx = torch.from_numpy(network.inputs(v[:-1], r[:-1])[None]); ss = torch.from_numpy(static[None])
            stop = min(n-1, 31); end = cal+64+stop
            with torch.no_grad():
                allq = m(bx, ss); prefix = m(bx[:, :, :end], ss)
                poisoned = bx.clone(); poisoned[:, :, end:] += 100
                future = m(poisoned, ss)
            assert torch.equal(allq[:, :, :end], future[:, :, :end])
            prefix_error = float((allq[:, :, :end]-prefix).abs().max()); assert prefix_error < 2e-6
            qq = allq[0, :, 63:63+cal+n].T.numpy()
            allx, width = features.feature_matrix(v[64:], r[64:], qq, cal, np.arange(n), static)
            sampled, _ = features.feature_matrix(v[64:], r[64:], qq, cal, log_steps(n), static)
            assert width == 101 and np.array_equal(sampled, allx[log_steps(n)])
            short, _ = features.feature_matrix(v[64:64+cal+stop+1], r[64:64+cal+stop+1], qq[:cal+stop+1], cal, np.arange(stop+1), static)
            assert np.array_equal(short, allx[:stop+1])
            picked = rowids == sid
            assert np.array_equal(base[picked], np.load(manifest[sid]['path'], mmap_mode='r')[ticks[picked], :422])
            checks.append({'fold': fold, 'inner': inner, 'fit_series': len(fitids), 'prefix_error': prefix_error, 'tick_join_exact': True, 'feature_prefix_exact': True})
    y = np.array([[0, 0, 1, -1], [0, 1, 1, 1], [0, 0, 0, 0]], dtype='i1')
    p = np.array([[.1, .2, .9, 0], [.2, .8, .8, .7], [.3, .1, .1, .1]])
    assert metric(y, p) == 1.
    write(OUT / 'implementation_checks.json', {'passed': True, 'predictor_checks': checks, 'metric_ragged_perfect_case': True, 'official_sdk_check': False})
    print('Tick joins, causal prefixes, feature columns and training isolation verified.', flush=True)

if __name__ == '__main__':
    verify()
