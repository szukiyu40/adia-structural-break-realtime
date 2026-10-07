"""Bounded parity and series-independence checks against supplied artifacts.

Invoke as python -m serving.validate from the directory containing this package.
Only caller-supplied paths are accessed. This is not the official SDK runner.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .adapter import Resources, State, file_sha256


def check(args):
    import pyarrow.parquet as pq
    started = time.perf_counter()
    resources = Resources(args.models)
    frame = pq.read_table(args.raw, columns=['id', 'time', 'period', 'value'],
                          filters=[('id', 'in', args.ids)]).to_pandas(ignore_metadata=True)
    streams = {}
    for sid in args.ids:
        group = frame[frame.id == sid]
        if not len(group) or not np.all(np.diff(group.time) == 1):
            raise ValueError(f'Missing or unordered series: {sid}')
        streams[sid] = (group.loc[group.period == 1, 'value'].to_numpy('f8'),
                        group.loc[group.period == 2, 'value'].to_numpy('f8'))
    manifest = json.loads(Path(args.feature_manifest).read_text())
    prepared = Path(args.prepared)
    values = np.load(prepared / 'values.npy', mmap_mode='r')
    innovations = np.load(prepared / 'innovations.npy', mmap_mode='r')
    expected = Path(args.expected)
    held_ids = np.load(expected / 'held_ids.npy')
    reference = {'augmented': np.load(expected / 'augmented_probability.npy', mmap_mode='r')}
    if resources.control_heads:
        reference['control'] = np.load(expected / 'control_probability.npy', mmap_mode='r')
    rows, independent = [], {}
    for sid, (history, online) in streams.items():
        t0 = time.perf_counter()
        state = State(history, resources)
        init_seconds = time.perf_counter()-t0
        features = np.load(manifest[sid]['path'], mmap_mode='r')
        assert file_sha256(manifest[sid]['path']) == manifest[sid]['sha256']
        generated = []
        t0 = time.perf_counter()
        for tick, value in enumerate(online):
            generated.append(state.update_pair(value))
            assert np.array_equal(state.last_base_features, features[tick, :422])
            assert state.last_normalized_value == values[sid, len(history)+tick]
            assert state.last_innovation == innovations[sid, len(history)+tick]
        elapsed = time.perf_counter()-t0
        actual = np.array(generated, dtype=np.float64)
        independent[sid] = actual
        position = np.searchsorted(held_ids, sid)
        assert position < len(held_ids) and held_ids[position] == sid
        errors = {}
        for arm, column in [('control', 0), ('augmented', 1)]:
            if arm in reference:
                error = float(np.max(np.abs(actual[:, column]-reference[arm][position, :len(online)])))
                assert error < 1e-8, (sid, arm, error)
                errors[arm] = error
                assert np.array_equal(actual[:, column], np.repeat(actual[::4, column], 4)[:len(actual)])
        rows.append({'id': sid, 'history_points': len(history), 'online_points': len(online),
                     'init_seconds': init_seconds, 'online_seconds': elapsed,
                     'milliseconds_per_observation': 1000*elapsed/len(online),
                     'saved_probability_max_errors': errors, 'base422_exact': True,
                     'ar_values_and_innovations_exact': True, 'held_cadence_exact': True})
    # Reuse one resource object while three independent states are interleaved
    # in reverse order. A different subset and fresh resource are checked below.
    states = {sid: State(streams[sid][0], resources) for sid in reversed(args.ids)}
    interleaved = {sid: [] for sid in args.ids}
    for tick in range(max(len(streams[sid][1]) for sid in args.ids)):
        for sid in reversed(args.ids):
            if tick < len(streams[sid][1]):
                interleaved[sid].append(states[sid].update_pair(streams[sid][1][tick]))
    for sid in args.ids:
        assert np.array_equal(np.array(interleaved[sid], dtype=np.float64), independent[sid], equal_nan=True)
    prefix_results = []
    fresh = Resources(args.models, expected_hashes=resources.model_hashes)
    for sid in args.ids[::2]:
        history, online = streams[sid]
        count = min(33, len(online))
        for label, loader in [('shared_resource', resources), ('fresh_resource', fresh)]:
            state = State(history, loader)
            probability = np.array([state.update_pair(value) for value in online[:count]], dtype=np.float64)
            assert np.array_equal(probability, independent[sid][:count], equal_nan=True)
            prefix_results.append({'id': sid, 'resource': label, 'prefix_points': count,
                                   'prefix_repeat_subset_order_exact': True})
    report = {'status': 'passed', 'scope': 'Bounded CPU serving adapter validation on supplied saved-fold models.',
              'series': rows, 'interleaved_reverse_order_exact': True,
              'prefix_repeat_and_subset_checks': prefix_results, 'resources': resources.describe(),
              'input_hashes': {'feature_manifest': file_sha256(args.feature_manifest),
                               'held_ids': file_sha256(expected / 'held_ids.npy'),
                               **{f'{arm}_probability': file_sha256(expected / f'{arm}_probability.npy')
                                  for arm in reference}},
              'official_sdk_called': False, 'official_submission': False,
              'full_competition_runtime_certification': False,
              'incumbent_or_blend_executed': False, 'wall_seconds': time.perf_counter()-started}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'series': len(rows),
                      'online_points': sum(row['online_points'] for row in rows),
                      'largest_saved_probability_error': max(max(row['saved_probability_max_errors'].values()) for row in rows),
                      'interleaving_repeat_prefix_subset_exact': True,
                      'wall_seconds': report['wall_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--feature-manifest', required=True)
    parser.add_argument('--expected', required=True)
    parser.add_argument('--ids', nargs='+', type=int, default=[0, 5, 10])
    parser.add_argument('--report', required=True)
    check(parser.parse_args())
