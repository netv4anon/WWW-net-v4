"""Measure input-unit reuse consequences in the original NET-V4 snapshots.

No fitting, target-law replacement, new samples, or external data acquisition.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FALLBACKS = ['LastValue'] + ['STID_NoCalendar_seed' + str(i) for i in (1, 2, 3)]
KEYS = ('center', 'all_nine_slots')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csvout(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def keyed(row):
    return np.ascontiguousarray(row).tobytes()


def statistics(prediction, truth, selection):
    n = int(selection.sum())
    if not n:
        return {'queries': 0, 'points': 0, 'mae': None, 'rmse': None}
    residual = prediction[selection] - truth[selection]
    return {'queries': n, 'points': n * 12,
            'mae': float(np.abs(residual).mean()),
            'rmse': float(np.sqrt(np.square(residual).mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path,
                        default=HERE.parent / 'submission_completion/netv4_review_candidate_v0_3')
    parser.add_argument('--out', type=Path, default=HERE / 'core_reuse_results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': digest(HERE / 'CORE_REUSE_PROTOCOL_ZH.md'),
              'runner_sha256': digest(Path(__file__)), 'sources': {},
              'new_fits': 0, 'new_target_laws': 0, 'new_samples': 0,
              'development_exposed': True, 'source_identity_established': False}
    summaries, metric_rows, witnesses = [], [], []
    for variant in ('packaged', 'working'):
        folder = args.base / 'data' / variant
        source_paths = [folder / (name + '.npz') for name in ('history', 'targets', 'splits', 'analysis_metadata')]
        frozen['sources'][variant] = {path.name: digest(path) for path in source_paths}
        with np.load(folder / 'history.npz', allow_pickle=False) as z:
            x = z['history']
        with np.load(folder / 'splits.npz', allow_pickle=False) as z:
            train, test = z['train'], z['test']
        assert x.shape == (33076, 12, 9) and x.dtype == np.float32
        assert np.isfinite(x).all() and not set(train).intersection(test)
        # Only training futures enter construction of lookup predictions.
        with np.load(folder / 'targets.npz', allow_pickle=False) as z:
            train_future = z['future'][train, :, 0].astype(np.float64)
        reference = {'LastValue': np.repeat(x[test, -1:, 0], 12, axis=1).astype(np.float64)}
        reference_hashes = {}
        for name in FALLBACKS[1:]:
            path = args.base / 'runs' / variant / name / 'predictions.npz'
            reference_hashes[name] = digest(path)
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['test_indices'], test)
                reference[name] = z['prediction'].astype(np.float64)
        frozen['sources'][variant]['saved_references'] = reference_hashes
        variant_payload = {}
        for key_name in KEYS:
            feature = x[:, :, 0] if key_name == 'center' else x
            library = defaultdict(list)
            for train_position, index in enumerate(train):
                library[keyed(feature[index])].append(train_position)
            matches = [library.get(keyed(feature[index]), []) for index in test]
            hit = np.array([bool(m) for m in matches], dtype=bool)
            retrieval = np.zeros((len(test), 12), dtype=np.float64)
            conflict = np.zeros(len(test), dtype=bool)
            for position in np.flatnonzero(hit):
                members = matches[position]
                future = train_future[members]
                retrieval[position] = future.mean(axis=0)
                conflict[position] = not np.all(future == future[0])
            predictions, seals = {}, {}
            for name in FALLBACKS:
                prediction = reference[name].copy()
                prediction[hit] = retrieval[hit]
                assert np.isfinite(prediction).all()
                assert np.array_equal(prediction[~hit], reference[name][~hit])
                path = args.out / ('prediction_' + variant + '_' + key_name + '_' + name + '.npz')
                np.savez_compressed(path, prediction=prediction, test_indices=test, hit=hit)
                seals[name] = {'path': path.name, 'sha256': digest(path)}
                predictions[name] = prediction
            variant_payload[key_name] = (matches, hit, conflict, predictions, seals)
        # Bind all rule outputs before extracting test futures for scoring.
        dump(args.out / ('PREDICTION_SEAL_' + variant + '.json'),
             {'data': frozen['sources'][variant],
              'predictions': {name: payload[4] for name, payload in variant_payload.items()},
              'query_targets_used_for_lookup': False})
        with np.load(folder / 'targets.npz', allow_pickle=False) as z:
            truth = z['future'][test, :, 0].astype(np.float64)
            event = z['is_event'][test].astype(bool)
        with np.load(folder / 'analysis_metadata.npz', allow_pickle=False) as z:
            source = z['source'][test]
            segment = z['segment_id']
        varying = np.ptp(truth, axis=1) > 0
        for key_name, payload in variant_payload.items():
            matches, hit, conflict, predictions, seals = payload
            exact_future = np.zeros(len(test), dtype=bool)
            for position in np.flatnonzero(hit):
                exact_future[position] = bool(np.any(np.all(train_future[matches[position]] == truth[position], axis=1)))
                members = train[matches[position]]
                witnesses.append({'variant': variant, 'input_key': key_name,
                                  'test_position': int(position), 'test_index': int(test[position]),
                                  'train_indices': '|'.join(map(str, members)),
                                  'matching_train_count': len(members),
                                  'conflicting_train_futures': bool(conflict[position]),
                                  'any_exact_train_future': bool(exact_future[position]),
                                  'constant_query_history': bool(np.ptp(x[test[position], :, 0]) == 0),
                                  'constant_query_target': bool(not varying[position]),
                                  'same_recorded_center_id_exists': bool(np.any(segment[members] == segment[test[position]])),
                                  'query_source': str(source[position]),
                                  'physical_source_identity_established': False})
            # A second, numerical tuple-key implementation checks the history matches.
            tuple_library = defaultdict(list)
            feature = x[:, :, 0] if key_name == 'center' else x
            for index in train:
                tuple_library[tuple(feature[index].ravel().tolist())].append(int(index))
            for position, index in enumerate(test):
                independent = tuple_library.get(tuple(feature[index].ravel().tolist()), [])
                assert independent == train[matches[position]].tolist(), (variant, key_name, int(index))
            if key_name == 'center':
                assert int(hit.sum()) == (102 if variant == 'working' else 0)
                assert int(exact_future.sum()) == (74 if variant == 'working' else 0)
            selections = {'all': np.ones(len(test), dtype=bool), 'hit': hit, 'miss': ~hit,
                          'event': event, 'control': ~event,
                          'varying_target': varying, 'constant_target': ~varying,
                          'hit_varying_target': hit & varying, 'hit_constant_target': hit & ~varying,
                          'hit_unique_train_future': hit & ~conflict,
                          'hit_conflicting_train_future': hit & conflict,
                          'hit_exact_train_future': exact_future,
                          'hit_no_exact_train_future': hit & ~exact_future}
            selections.update({'source:' + str(s): source == s for s in np.unique(source)})
            consequence = {}
            for name in FALLBACKS:
                prediction = predictions[name]
                all_base = statistics(reference[name], truth, selections['all'])
                all_lookup = statistics(prediction, truth, selections['all'])
                delta = all_lookup['mae'] - all_base['mae']
                hit_sum_difference = float((np.abs(prediction[hit] - truth[hit]) -
                                            np.abs(reference[name][hit] - truth[hit])).sum())
                assert np.isclose(delta, hit_sum_difference / truth.size, rtol=0, atol=1e-10)
                for group, selection in selections.items():
                    before = statistics(reference[name], truth, selection)
                    after = statistics(prediction, truth, selection)
                    metric_rows.append({'variant': variant, 'input_key': key_name, 'fallback': name,
                                        'population': group, 'queries': before['queries'], 'points': before['points'],
                                        'baseline_mae': before['mae'], 'lookup_mae': after['mae'],
                                        'baseline_rmse': before['rmse'], 'lookup_rmse': after['rmse'],
                                        'mae_change': None if not before['queries'] else after['mae'] - before['mae']})
                varied_before = statistics(reference[name], truth, selections['hit_varying_target'])
                varied_after = statistics(prediction, truth, selections['hit_varying_target'])
                consequence[name] = {'baseline_mae': all_base['mae'], 'lookup_mae': all_lookup['mae'],
                                     'mae_gain_percent': None if all_base['mae'] == 0 else -100 * delta / all_base['mae'],
                                     'hit_varying_baseline_mae': varied_before['mae'],
                                     'hit_varying_lookup_mae': varied_after['mae'],
                                     'all_miss_predictions_identical': True,
                                     'loss_change_entirely_on_hits': True}
            summaries.append({'variant': variant, 'input_key': key_name, 'train_rows': len(train),
                              'test_rows': len(test), 'hit_rows': int(hit.sum()),
                              'hit_rows_with_exact_train_future': int(exact_future.sum()),
                              'hit_conflicting_future_rows': int(conflict.sum()),
                              'hit_varying_target_rows': int((hit & varying).sum()),
                              'hit_event_rows': int((hit & event).sum()), 'consequences': consequence})
    csvout(args.out / 'metrics.csv', metric_rows)
    csvout(args.out / 'witnesses.csv', witnesses)
    main_result = next(r for r in summaries if r['variant'] == 'working' and r['input_key'] == 'center')
    primary = main_result['consequences']['LastValue']
    gate = (primary['lookup_mae'] < primary['baseline_mae'] and
            primary['hit_varying_lookup_mae'] is not None and
            primary['hit_varying_lookup_mae'] < primary['hit_varying_baseline_mae'])
    dump(args.out / 'FROZEN_INPUTS.json', frozen)
    dump(args.out / 'SUMMARY.json', {'primary_gate_passed': gate, 'results': summaries,
                                   'new_fits': 0, 'new_rule_output_files': len(summaries) * len(FALLBACKS),
                                   'old_collision_counts_reverified_not_claimed_as_new': True,
                                   'new_scientific_endpoint': 'Lookup consequences on original NET-V4',
                                   'independent_tuple_matching_verified': True,
                                   'physical_source_identity_established': False,
                                   'published_model_inflation_established': False,
                                   'chronological_repair_established': False})
    print(json.dumps({'primary_gate_passed': gate, 'core_results': summaries}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
