"""Re-extract 144 neural features at the exact comparison ticks."""
from integration_common import *
import argparse
import time
import torch

def extract(fold, inner):
    ensure_protocol(); torch.set_num_threads(2)
    network, features = predictor_modules()
    d = arrays(); ids = np.arange(10000)
    train_ids = ids[(ids % 5 != fold) & ((ids // 5) % 2 != inner)]
    held_ids = ids[ids % 5 == fold]
    checkpoint = predictor_path(fold, inner)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    expected_fit = ids[(ids % 5 != fold) & ((ids // 5) % 2 == inner)]
    assert np.array_equal(ck['fit_ids'], expected_fit) and ck['step'] == 800
    assert not np.intersect1d(ck['fit_ids'], np.r_[train_ids, held_ids]).size
    dest = WORK / f'fold{fold}'; dest.mkdir(exist_ok=True)
    done = OUT / f'fold{fold}/extraction{inner}.json'
    if done.exists() and all((dest / f'{split}{inner}.npz').exists() for split in ('train', 'held')):
        return
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = network.Predictor().to(device); model.load_state_dict(ck['model']); model.eval()
    start = time.time(); reports = {}
    for split, sids in [('train', train_ids), ('held', held_ids)]:
        ticks = [log_steps(int(d['lengths'][s])) if split == 'train'
                 else np.arange(0, int(d['lengths'][s]), INFER_STRIDE) for s in sids]
        counts = np.array([len(t) for t in ticks]); offsets = np.r_[0, counts.cumsum()]
        mat = np.lib.format.open_memmap(dest / f'{split}{inner}_x.npy', mode='w+', dtype='f4', shape=(int(offsets[-1]), 144))
        for lo in range(0, len(sids), 16):
            batch = sids[lo:lo+16]; seq = []; meta = []
            for sid in batch:
                h = int(d['hist_lengths'][sid]); n = int(d['lengths'][sid]); cal = min(512, h//2-64)
                v = np.array(d['values'][sid, h-cal-64:h+n]); r = np.array(d['innovations'][sid, h-cal-64:h+n])
                seq.append(network.inputs(v[:-1], r[:-1])); meta.append((v[64:], r[64:], cal))
            bx = np.zeros((len(batch), 4, max(s.shape[1] for s in seq)), dtype='f4')
            for i, seq_i in enumerate(seq):
                bx[i, :, :seq_i.shape[1]] = seq_i
            with torch.no_grad():
                q = model(torch.from_numpy(bx).to(device), torch.from_numpy(np.array(d['static'][batch])).to(device)).cpu().numpy()
            for j, sid in enumerate(batch):
                v, r, cal = meta[j]; qq = q[j, :, 63:63+len(r)].T
                x, width = features.feature_matrix(v, r, qq, cal, ticks[lo+j], d['static'][sid])
                assert width == 101 and x.shape == (len(ticks[lo+j]), 245) and np.isfinite(x).all()
                mat[offsets[lo+j]:offsets[lo+j+1]] = x[:, 101:]
        mat.flush()
        identities = dest / f'{split}{inner}.npz'
        np.savez(identities, ids=np.repeat(sids, counts).astype('i4'), ticks=np.concatenate(ticks).astype('i4'))
        reports[split] = {'series': len(sids), 'rows': int(offsets[-1]), 'columns': 144,
                          'features_sha256': sha(dest / f'{split}{inner}_x.npy'), 'identity_sha256': sha(identities)}
        del mat
    reports.update({'fold': fold, 'inner': inner, 'device': device, 'checkpoint_sha256': sha(checkpoint),
                    'seconds': time.time()-start, 'fit_and_feature_ids_disjoint': True})
    write(done, reports); print(json.dumps({'phase': 'extract', 'fold': fold, 'inner': inner, 'seconds': time.time()-start}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--fold', type=int, required=True); p.add_argument('--inner', type=int, required=True)
    a = p.parse_args(); extract(a.fold, a.inner)
