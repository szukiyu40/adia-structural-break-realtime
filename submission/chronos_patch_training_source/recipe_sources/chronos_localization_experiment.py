"""Train matched heads on frozen fold encoders, then evaluate entire paths."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import math
import time
import numpy as np
import torch
from scipy.special import expit
from data import Data, Sampler, OUT, WORK, INCUMBENT, integrate
from network import Encoder, PatchHead, raw_features, objective
from util import write, sha, metric, state_sha
from protocol import prepare, STEPS, SEED, STRIDE, CONDITIONS


def reference_cache(fold, data, encoder, device):
    directory = WORK/f'fold{fold}'; directory.mkdir(exist_ok=True, parents=True)
    dest = directory/'reference.npy'; metadata = directory/'reference.json'
    expected = {'encoder_state_sha256': state_sha(encoder), 'protocol_sha256': sha(OUT/'protocol.json')}
    if dest.exists():
        existing = json.loads(metadata.read_text())
        assert all(existing[k] == v for k, v in expected.items())
        assert existing['file_sha256'] == sha(dest)
        return np.load(dest, mmap_mode='r')
    value = np.empty((10000, 2, 768), dtype='f4')
    start = time.monotonic()
    for lo in range(0, 10000, 64):
        ids = np.arange(lo, min(lo+64, 10000))
        h = torch.from_numpy(data.history(ids)).to(device)
        value[ids] = encoder.reference(h).cpu().numpy()
        if lo % 1024 == 0:
            print({'fold': fold, 'phase': 'normal_reference_cache', 'done': int(ids[-1])+1,
                   'seconds': time.monotonic()-start}, flush=True)
    assert np.isfinite(value).all()
    np.save(dest.with_suffix('.tmp.npy'), value); dest.with_suffix('.tmp.npy').replace(dest)
    write(metadata, {**expected, 'file_sha256': sha(dest)})
    return np.load(dest, mmap_mode='r')


def fit(fold, data, encoder, reference, device):
    dest = OUT/f'fold{fold}'; dest.mkdir(exist_ok=True, parents=True)
    fit_ids = np.flatnonzero(np.arange(10000) % 5 != fold)
    held = np.flatnonzero(np.arange(10000) % 5 == fold)
    np.savez(dest/'split.npz', train=fit_ids, held=held)
    torch.manual_seed(SEED+fold)
    point = PatchHead().to(device); heads = [point, copy.deepcopy(point)]
    assert state_sha(heads[0]) == state_sha(heads[1])
    initial_head_sha = state_sha(point)
    initial_encoder_sha = state_sha(encoder)
    optimizers = [torch.optim.AdamW(h.parameters(), lr=.001, weight_decay=.001) for h in heads]
    sampler = Sampler(data, fit_ids, SEED+fold)
    start = time.monotonic(); history = []; schedule = []; begin = 0
    checkpoint = dest/'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['protocol_sha256'] == sha(OUT/'protocol.json')
        for head, optimizer, hs, opt in zip(heads, optimizers, state['heads'], state['optimizers']):
            head.load_state_dict(hs); optimizer.load_state_dict(opt)
        begin, history, schedule = state['step'], state['history'], state['schedule']
        sampler.rng.bit_generator.state = state['numpy_rng_state']
    for h in heads: h.train()
    for step in range(begin+1, STEPS+1):
        ids, ticks, prior = sampler.batch()
        assert np.all(ids % 5 != fold)
        schedule.append([int(ticks[0]), *ids.tolist()])
        h = torch.from_numpy(data.history(ids)).to(device)
        c = torch.from_numpy(data.current(ids, ticks)).to(device)
        tt = torch.from_numpy(ticks).to(device)
        ref = torch.from_numpy(np.array(reference[ids], copy=True)).to(device)
        z = encoder.patches(c); raw = raw_features(h, c, tt)
        targets, valid = data.targets(ids, ticks)
        targets, valid = torch.from_numpy(targets).to(device), torch.from_numpy(valid).to(device)
        multiplier = .1+.9*(1+math.cos(math.pi*(step-1)/STEPS))/2
        records = {}
        for condition, head, optimizer in zip(CONDITIONS, heads, optimizers):
            torch.manual_seed(SEED+100000+fold*10000+step)
            optimizer.param_groups[0]['lr'] = .001*multiplier
            optimizer.zero_grad(set_to_none=True)
            loss = objective(head(ref, z, raw), targets, valid, prior, condition == 'localization')
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            assert torch.isfinite(norm)
            optimizer.step()
            records[condition] = {'loss': float(loss.detach().cpu()), 'gradient_norm': float(norm.cpu())}
        if step % 50 == 0 or step == 1:
            record = {'fold': fold, 'step': step, 'seconds': time.monotonic()-start, **records}
            history.append(record); write(dest/'learning.json', history); print(record, flush=True)
        if step % 400 == 0 or step == STEPS:
            state = {'heads': [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in heads],
                     'optimizers': [opt.state_dict() for opt in optimizers], 'step': step, 'history': history,
                     'schedule': schedule, 'numpy_rng_state': sampler.rng.bit_generator.state,
                     'protocol_sha256': sha(OUT/'protocol.json'), 'fold': fold, 'conditions': CONDITIONS}
            torch.save(state, dest/'checkpoint.tmp.pt'); (dest/'checkpoint.tmp.pt').replace(checkpoint)
    assert state_sha(encoder) == initial_encoder_sha
    np.save(dest/'training_schedule.npy', np.array(schedule, dtype='i4'))
    for condition, head in zip(CONDITIONS, heads):
        folder = dest/condition; folder.mkdir(exist_ok=True)
        torch.save({'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    'protocol_sha256': sha(OUT/'protocol.json'), 'fold': fold, 'condition': condition}, folder/'head.pt')
    write(dest/'training.json', {'fold': fold, 'steps_per_condition': STEPS, 'conditions': CONDITIONS,
          'train_series': len(fit_ids), 'held_series': len(held), 'encoder_unchanged': True,
          'encoder_sha256': initial_encoder_sha, 'identical_head_initial_state_sha256': initial_head_sha,
          'shared_training_schedule_sha256': sha(dest/'training_schedule.npy'),
          'same_per_step_dropout_seeds': True, 'wall_seconds_this_process': time.monotonic()-start,
          'checkpoint_sha256': sha(checkpoint), 'device': device})
    return [h.eval() for h in heads]


@torch.inference_mode()
def evaluate(fold, data, encoder, reference, heads, device):
    held = np.flatnonzero(np.arange(10000) % 5 == fold)
    rows = np.concatenate([np.full((int(data.lengths[sid])+STRIDE-1)//STRIDE, i, dtype='i4') for i, sid in enumerate(held)])
    ticks = np.concatenate([np.arange(0, int(data.lengths[sid]), STRIDE, dtype='i8') for sid in held])
    logits = np.empty((2, len(rows)), dtype='f8'); start = time.monotonic()
    for lo in range(0, len(rows), 64):
        rr, tt = rows[lo:lo+64], ticks[lo:lo+64]; ids = held[rr]
        history = torch.from_numpy(data.history(ids)).to(device)
        current = torch.from_numpy(data.current(ids, tt)).to(device)
        t = torch.from_numpy(tt).to(device)
        ref = torch.from_numpy(np.array(reference[ids], copy=True)).to(device)
        z = encoder.patches(current); raw = raw_features(history, current, t)
        for i, head in enumerate(heads):
            logits[i, lo:lo+len(rr)] = head(ref, z, raw)[:, -1].cpu().numpy()
        if lo % 4096 == 0:
            print({'fold': fold, 'phase': 'held_inference', 'done': min(lo+64, len(rows)),
                   'total': len(rows), 'seconds': time.monotonic()-start}, flush=True)
    assert np.isfinite(logits).all()
    baseline = np.load(INCUMBENT, mmap_mode='r')[held]; y = data.labels[held]
    base_score = metric(y, baseline)
    for condition, raw_scores in zip(CONDITIONS, logits):
        probability = np.zeros((len(held), 1000), dtype='f8'); cursor = 0
        for row, sid in enumerate(held):
            n = int(data.lengths[sid]); count = (n+STRIDE-1)//STRIDE
            probability[row, :n] = np.repeat(expit(raw_scores[cursor:cursor+count]), STRIDE)[:n]
            cursor += count
        assert cursor == len(rows)
        integrated = integrate(fold, probability)
        standalone, replaced = metric(y, probability), metric(y, integrated)
        error = max(abs(replaced-metric(y, integrated, True)), abs(standalone-metric(y, probability, True)))
        assert error < 1e-12
        folder = OUT/f'fold{fold}'/condition
        np.save(folder/'held_ids.npy', held); np.save(folder/'probability.npy', probability)
        np.save(folder/'integrated.npy', integrated)
        result = {'fold': fold, 'condition': condition, 'incumbent': base_score,
                  'standalone': standalone, 'integrated': replaced, 'integrated_delta': replaced-base_score,
                  'held_series': len(held), 'full_path_points': int(data.lengths[held].sum()),
                  'encoder_updates': len(rows), 'independent_metric_max_error': error,
                  'wall_seconds_shared_evaluation': time.monotonic()-start,
                  'development_reused': True, 'official_score': False}
        write(folder/'result.json', result); print(json.dumps(result), flush=True)


def run(fold, device):
    prepare(); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    data = Data(); encoder = Encoder(fold, device)
    reference = reference_cache(fold, data, encoder, device)
    heads = fit(fold, data, encoder, reference, device)
    evaluate(fold, data, encoder, reference, heads, device)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--fold', required=True, type=int, choices=[0, 1])
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = p.parse_args(); run(args.fold, args.device)
