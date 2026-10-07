"""Matched head continuation and last-two-block adaptation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import math
import time
import numpy as np
import torch
from scipy.special import expit
from common import OUT, ORIGINAL, Data, Sampler, STEPS, SEED, STRIDE, CONDITIONS, INCUMBENT
from common import raw_features, objective, integrate, write, sha, state_sha, metric
from adaptation import SharedPrefix, Tail, retained_head, reference
from freeze import prepare


def cpu_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def fit(fold, data, prefix):
    directory = OUT/f'fold{fold}'; directory.mkdir(exist_ok=True)
    fit_ids = np.flatnonzero(np.arange(10000)%5 != fold)
    held = np.flatnonzero(np.arange(10000)%5 == fold)
    np.savez(directory/'split.npz', train=fit_ids, held=held)
    tails = [Tail(prefix, False), Tail(prefix, True)]
    heads = [retained_head(fold, 'mps') for _ in CONDITIONS]
    initial_prefix = state_sha(prefix); initial_tail = state_sha(tails[0]); initial_head = state_sha(heads[0])
    assert initial_tail == state_sha(tails[1]) and initial_head == state_sha(heads[1])
    optimizers = [torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=.001) for h in heads]
    optimizers[1].add_param_group({'params': list(tails[1].parameters()), 'lr': 1e-5, 'weight_decay': .01})
    sampler = Sampler(data, fit_ids, SEED+fold)
    history = []; schedule = []; begin = 0; checkpoint = directory/'checkpoint.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == sha(OUT/'protocol.json') and saved['fold'] == fold
        for head, tail, opt, hs, ts, os in zip(heads, tails, optimizers, saved['heads'], saved['tails'], saved['optimizers']):
            head.load_state_dict(hs); tail.load_state_dict(ts); opt.load_state_dict(os)
        begin, history, schedule = saved['step'], saved['history'], saved['schedule']
        sampler.rng.bit_generator.state = saved['numpy_rng_state']
        del saved
    for h in heads: h.train()
    start = time.monotonic()
    for step in range(begin+1, STEPS+1):
        ids, ticks, prior = sampler.batch()
        assert np.all(ids%5 != fold)
        schedule.append([int(ticks[0]), *ids.tolist()])
        h = torch.from_numpy(data.history(ids)).to('mps')
        c = torch.from_numpy(data.current(ids, ticks)).to('mps')
        z, mask = prefix(torch.cat([h, c]))
        raw = raw_features(h, c, torch.from_numpy(ticks).to('mps'))
        target, valid = data.targets(ids, ticks)
        target, valid = torch.from_numpy(target).to('mps'), torch.from_numpy(valid).to('mps')
        multiplier = .1+.9*(1+math.cos(math.pi*(step-1)/STEPS))/2
        records = {}
        for i, (condition, head, tail, opt) in enumerate(zip(CONDITIONS, heads, tails, optimizers)):
            torch.manual_seed(SEED+100000+fold*10000+step)
            opt.param_groups[0]['lr'] = 3e-4*multiplier
            if i: opt.param_groups[1]['lr'] = 1e-5*multiplier
            opt.zero_grad(set_to_none=True)
            patches = tail(z, mask, prefix.model.encoder)
            ref = reference(patches[:16], h)
            loss = objective(head(ref, patches[16:], raw), target, valid, prior, True)
            assert torch.isfinite(loss)
            loss.backward()
            head_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            assert torch.isfinite(head_norm)
            tail_norm = torch.nn.utils.clip_grad_norm_(tail.parameters(), 1.) if i else torch.tensor(0.)
            assert torch.isfinite(tail_norm)
            opt.step()
            records[condition] = dict(loss=float(loss.detach().cpu()), head_gradient_norm=float(head_norm.cpu()), tail_gradient_norm=float(tail_norm.cpu()))
        if step == 1 or step%50 == 0:
            record = dict(fold=fold, step=step, seconds=time.monotonic()-start, **records)
            history.append(record); write(directory/'learning.json', history); print(record, flush=True)
        if step%250 == 0 or step == STEPS:
            saved = dict(heads=[cpu_state(m) for m in heads], tails=[cpu_state(m) for m in tails],
                         optimizers=[o.state_dict() for o in optimizers], step=step, history=history, schedule=schedule,
                         numpy_rng_state=sampler.rng.bit_generator.state, protocol_sha256=sha(OUT/'protocol.json'), fold=fold)
            torch.save(saved, directory/'checkpoint.tmp.pt'); (directory/'checkpoint.tmp.pt').replace(checkpoint)
            del saved
    assert state_sha(prefix) == initial_prefix
    assert state_sha(tails[0]) == initial_tail and state_sha(tails[1]) != initial_tail
    np.save(directory/'training_schedule.npy', np.array(schedule, dtype='i4'))
    for condition, head, tail in zip(CONDITIONS, heads, tails):
        destination = directory/condition; destination.mkdir(exist_ok=True)
        torch.save(dict(head=cpu_state(head), tail=cpu_state(tail), fold=fold, condition=condition,
                        protocol_sha256=sha(OUT/'protocol.json')), destination/'model.pt')
    write(directory/'training.json', dict(fold=fold, steps_per_condition=STEPS,
          initial_prefix_sha256=initial_prefix, initial_tail_sha256=initial_tail, initial_head_sha256=initial_head,
          frozen_prefix_unchanged=True, control_tail_unchanged=True, tuned_tail_changed=True,
          final_tail_sha256=[state_sha(t) for t in tails], final_head_sha256=[state_sha(h) for h in heads],
          shared_schedule_sha256=sha(directory/'training_schedule.npy'), same_per_step_dropout_seeds=True,
          checkpoint_sha256=sha(checkpoint), wall_seconds_this_process=time.monotonic()-start))
    return [h.eval() for h in heads], tails


@torch.inference_mode()
def evaluate(fold, data, prefix, heads, tails):
    held = np.flatnonzero(np.arange(10000)%5 == fold)
    refs = np.empty((2, len(held), 2, 768), dtype='f4')
    for lo in range(0, len(held), 64):
        history = torch.from_numpy(data.history(held[lo:lo+64])).to('mps')
        z, mask = prefix(history)
        for i, tail in enumerate(tails):
            refs[i, lo:lo+len(history)] = reference(tail(z, mask, prefix.model.encoder), history).cpu().numpy()
    assert np.isfinite(refs).all()
    # The frozen control normal cache must agree with the original cache.
    old_ref = np.load(ORIGINAL.parents[1]/'work/chronos_localization'/f'fold{fold}/reference.npy', mmap_mode='r')[held]
    normal_error = float(np.max(np.abs(refs[0]-old_ref)))
    assert normal_error < 2e-4, normal_error
    rows = np.concatenate([np.full((int(data.lengths[sid])+STRIDE-1)//STRIDE, i, dtype='i4') for i, sid in enumerate(held)])
    ticks = np.concatenate([np.arange(0, int(data.lengths[sid]), STRIDE, dtype='i8') for sid in held])
    logits = np.empty((2, len(rows)), dtype='f8'); start = time.monotonic()
    for lo in range(0, len(rows), 64):
        rr, tt = rows[lo:lo+64], ticks[lo:lo+64]; ids = held[rr]
        history = torch.from_numpy(data.history(ids)).to('mps')
        current = torch.from_numpy(data.current(ids, tt)).to('mps')
        z, mask = prefix(current)
        raw = raw_features(history, current, torch.from_numpy(tt).to('mps'))
        for i, (head, tail) in enumerate(zip(heads, tails)):
            ref = torch.from_numpy(np.array(refs[i, rr], copy=True)).to('mps')
            logits[i, lo:lo+len(rr)] = head(ref, tail(z, mask, prefix.model.encoder), raw)[:, -1].cpu().numpy()
        if lo%4096 == 0:
            print(dict(fold=fold, phase='held_inference', done=min(lo+64, len(rows)), total=len(rows), seconds=time.monotonic()-start), flush=True)
    assert np.isfinite(logits).all()
    np.savez(OUT/f'fold{fold}/held_logits.npz', rows=rows, ticks=ticks, logits=logits)
    y = data.labels[held]; incumbent = metric(y, np.load(INCUMBENT, mmap_mode='r')[held])
    retained = metric(y, np.load(ORIGINAL/f'fold{fold}/localization/integrated.npy'))
    for condition, scores in zip(CONDITIONS, logits):
        probability = np.zeros((len(held), 1000), dtype='f8'); cursor = 0
        for row, sid in enumerate(held):
            n = int(data.lengths[sid]); count = (n+STRIDE-1)//STRIDE
            probability[row, :n] = np.repeat(expit(scores[cursor:cursor+count]), STRIDE)[:n]
            cursor += count
        assert cursor == len(rows)
        integrated = integrate(fold, probability)
        standalone, score = metric(y, probability), metric(y, integrated)
        err = max(abs(standalone-metric(y, probability, True)), abs(score-metric(y, integrated, True)))
        assert err < 1e-12
        folder = OUT/f'fold{fold}/{condition}'
        np.save(folder/'held_ids.npy', held); np.save(folder/'probability.npy', probability); np.save(folder/'integrated.npy', integrated)
        result = dict(fold=fold, condition=condition, standalone=standalone, integrated=score, incumbent=incumbent,
                      retained_localization=retained, delta_incumbent=score-incumbent, delta_retained=score-retained,
                      held_series=len(held), full_path_points=int(data.lengths[held].sum()),
                      frozen_normal_cache_max_error=normal_error, independent_metric_max_error=err,
                      development_reused=True, official_score=False)
        write(folder/'result.json', result); print(result, flush=True)


def run(fold):
    prepare(); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    data = Data(); prefix = SharedPrefix(fold, 'mps')
    heads, tails = fit(fold, data, prefix)
    evaluate(fold, data, prefix, heads, tails)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--fold', type=int, required=True, choices=[0, 1])
    run(p.parse_args().fold)
