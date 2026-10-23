"""Bound fixed-forecast losses on a declared segment between two frozen targets."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TOL = 1e-9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def root_locations(x, y):
    roots = [float(x[i]) for i in np.flatnonzero(np.abs(y) <= TOL)]
    for i in np.flatnonzero(y[:-1] * y[1:] < 0):
        roots.append(float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
    return sorted(set(roots))


def loss_curve(prediction, a, d, knots):
    prediction, a, d = map(lambda value: np.asarray(value, dtype=np.float64).ravel(),
                           (prediction, a, d))
    n = len(a)
    start = float(np.abs(prediction - a).mean())
    diff = a - prediction
    slope = float(np.where(diff == 0, np.abs(d), d * np.sign(diff)).sum() / n)
    nonzero = d != 0
    crossings = (prediction[nonzero] - a[nonzero]) / d[nonzero]
    use = (crossings > 0) & (crossings < 1)
    jumps = np.zeros(len(knots), dtype=np.float64)
    np.add.at(jumps, np.searchsorted(knots, crossings[use]), 2 * np.abs(d[nonzero][use]) / n)
    interval_slope = slope + np.cumsum(jumps)
    return np.r_[start, start + np.cumsum(interval_slope[:-1] * np.diff(knots))]


def envelope(p_b, p_f, background, old, replacement):
    p_b, p_f, background, old, replacement = [np.asarray(v, dtype=np.float64).ravel()
                                             for v in (p_b, p_f, background, old, replacement)]
    assert all(v.shape == old.shape and np.isfinite(v).all() for v in
               (p_b, p_f, background, replacement))
    d = replacement - old
    nonzero = d != 0
    candidates = [np.array([0., 1.])]
    for p in (p_b, p_f):
        c = (p[nonzero] - old[nonzero]) / d[nonzero]
        candidates.append(c[(c > 0) & (c < 1)])
    knots = np.unique(np.concatenate(candidates))
    bf = loss_curve(p_b, old, d, knots)
    ff = loss_curve(p_f, old, d, knots)
    bb = float(np.abs(p_b - background).mean())
    curves = {'L_BF': bf, 'L_FF': ff, 'A': bf - ff, 'D': ff - bb, 'T': bf - bb}
    # Independent direct-loss arithmetic at endpoints, extrema and fixed knots.
    check_indices = {0, len(knots) - 1}
    for values in curves.values():
        check_indices.update((int(np.argmin(values)), int(np.argmax(values))))
    check_indices.update(np.linspace(0, len(knots) - 1, min(len(knots), 257), dtype=int).tolist())
    maximum_error = 0.
    for i in sorted(check_indices):
        target = old + knots[i] * d
        direct_bf = float(np.abs(p_b - target).mean())
        direct_ff = float(np.abs(p_f - target).mean())
        direct = {'L_BF': direct_bf, 'L_FF': direct_ff, 'A': direct_bf - direct_ff,
                  'D': direct_ff - bb, 'T': direct_bf - bb}
        maximum_error = max(maximum_error, *(abs(curves[k][i] - value) for k, value in direct.items()))
    assert maximum_error < 1e-9, maximum_error
    for alpha in (.125, .375, .625, .875):
        target = old + alpha * d
        assert abs(float(np.interp(alpha, knots, bf)) - np.abs(p_b - target).mean()) < 1e-9
        assert abs(float(np.interp(alpha, knots, ff)) - np.abs(p_f - target).mean()) < 1e-9
    record = {'L_BB': bb, 'breakpoints_including_endpoints': len(knots),
              'direct_verification_points': len(check_indices) + 4,
              'direct_verification_max_absolute_error': maximum_error}
    for name, values in curves.items():
        lo, hi = int(np.argmin(values)), int(np.argmax(values))
        record.update({name + '_min': float(values[lo]), name + '_max': float(values[hi]),
                       name + '_argmin_alpha': float(knots[lo]), name + '_argmax_alpha': float(knots[hi]),
                       name + '_alpha0': float(values[0]), name + '_alpha1': float(values[-1])})
    a = curves['A']
    record['A_status'] = ('uniformly_adverse' if a.max() < -TOL else
                          'uniformly_helpful' if a.min() > TOL else
                          'zero_within_tolerance' if np.abs(a).max() <= TOL else 'sign_or_zero_sensitive')
    record['same_endpoint_sign_misses_internal_A_reversal'] = bool(
        (a[0] < -TOL and a[-1] < -TOL and a.max() > TOL) or
        (a[0] > TOL and a[-1] > TOL and a.min() < -TOL))
    record['A_roots_alpha'] = root_locations(knots, a)
    record['D_roots_alpha'] = root_locations(knots, curves['D'])
    return record, {'alpha': knots, **curves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', type=Path,
                        default=HERE.parent / 'submission_ready_candidate_2026-10-08/resource/components/netv4')
    parser.add_argument('--predictions', type=Path, help='Optional keyed reader predictions; otherwise replay all saved configurations')
    parser.add_argument('--out', type=Path, default=HERE / 'target_envelope_results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.component.resolve()))
    from evaluate import align_prediction, load_task
    # Known finite example: endpoint A=-1/3, but midpoint A=7/3.
    toy, _ = envelope(np.array([0., 10., 0.]), np.array([4., 6., 1.]),
                      np.zeros(3), np.zeros(3), np.array([10., 10., 0.]))
    assert toy['same_endpoint_sign_misses_internal_A_reversal']
    assert abs(toy['A_max'] - 7 / 3) < TOL
    task = load_task(args.component / 'task')
    with np.load(args.component / 'task/recovery_targets.npz', allow_pickle=False) as z:
        positions = {int(k): i for i, k in enumerate(z['context_id'])}
        replacement = z['target'][[positions[int(c)] for c in task['contexts']]].astype(np.float64)
    models = (json.loads((args.component / 'MODELS.json').read_text(encoding='utf-8'))
              if args.predictions is None else
              [dict(id='submitted', variant='user', model='submitted', seed=None,
                    file=str(args.predictions.resolve()), sha256=sha(args.predictions))])
    configurations, seals = [], {}
    for model in models:
        path = args.component / model['file'] if args.predictions is None else args.predictions
        assert sha(path) == model['sha256'], model['id']
        pred = align_prediction(task, path)
        configurations.append((model['id'], pred[0], pred[3], model))
        seals[model['id']] = model['sha256']
    if args.predictions is None:
        for variant in ('packaged', 'working'):
            selected = [(pb, pf) for _, pb, pf, m in configurations
                        if m['variant'] == variant and m['model'] == 'STID_NoCalendar']
            assert len(selected) == 3
            # Repeated targets preserve the mean of the three losses, not ensemble prediction loss.
            configurations.append((variant + '__STID_loss_mean',
                                   np.stack([p[0] for p in selected]), np.stack([p[1] for p in selected]),
                                   dict(variant=variant, model='STID_loss_mean', seed=None)))
    scopes = {'all': np.ones(len(task['contexts']), dtype=bool),
              'test_origin': task['cohort'] == 'test_origin',
              'future_starting': task['timing'] == 'future_only'}
    rows, records = [], []
    old = task['target'][3]
    background = task['target'][0]
    for name, pb, pf, model in configurations:
        for scope, selection in scopes.items():
            if pb.ndim == 3:
                p_b, p_f = pb[:, selection], pf[:, selection]
                bg, y0, y1 = [np.broadcast_to(v[selection], p_b.shape) for v in (background, old, replacement)]
            else:
                p_b, p_f = pb[selection], pf[selection]
                bg, y0, y1 = [v[selection] for v in (background, old, replacement)]
            result, curve = envelope(p_b, p_f, bg, y0, y1)
            if scope == 'future_starting' and args.predictions is None:
                assert result['A_status'] == 'zero_within_tolerance', (name, result['A_status'])
            record = {'id': name, 'variant': model['variant'], 'model': model['model'],
                      'seed': model.get('seed'), 'cohort': scope, 'contexts': int(selection.sum()), **result}
            records.append(record)
            rows.append({k: v for k, v in record.items() if not k.endswith('_alpha')})
            np.savez_compressed(args.out / ('curve_' + name + '_' + scope + '.npz'), **curve)
    with (args.out / 'envelopes.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [r for r in records if r['cohort'] == 'all']
    reversals = [r['id'] for r in primary if r['same_endpoint_sign_misses_internal_A_reversal']]
    source_files = [args.component / 'task' / (n + '.npz') for n in ('inputs', 'evaluation', 'recovery_targets')]
    summary = {'family': 'Float64 convex segment between two frozen clipped full-arm targets',
               'alpha_interval': [0., 1.], 'sign_tolerance': TOL,
               'new_model_fits': 0, 'new_forecast_executions': 0,
               'new_target_family_analyzed': True, 'old_endpoint_objects_unchanged': True,
               'new_testing_theory_claimed': False, 'physical_target_truth_established': False,
               'protocol_sha256': sha(HERE / 'TARGET_ENVELOPE_PROTOCOL_ZH.md'),
               'runner_sha256': sha(Path(__file__)), 'source_sha256': {p.name: sha(p) for p in source_files},
               'prediction_sha256': seals, 'saved_configurations': len(models),
               'loss_mean_aggregates': len(configurations) - len(models),
               'primary_endpoint_reversal_hypothesis_supported': bool(reversals),
               'same_endpoint_sign_misses_internal_A_reversal': reversals,
               'known_finite_internal_reversal_check_passed': True,
               'all_future_starting_zero_response_checks_passed': all(
                   r['A_status'] == 'zero_within_tolerance' for r in records if r['cohort'] == 'future_starting'),
               'submitted_forecast_determinism_certified': False,
               'records': records}
    dump(args.out / 'SUMMARY.json', summary)
    print(json.dumps({'endpoint_reversal_hypothesis_supported': bool(reversals),
                      'missed_reversals': reversals,
                      'primary_rows': [{k: r[k] for k in ('id', 'A_alpha0', 'A_alpha1', 'A_min', 'A_max',
                                                         'A_status', 'D_min', 'D_max', 'D_roots_alpha')}
                                       for r in primary]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
