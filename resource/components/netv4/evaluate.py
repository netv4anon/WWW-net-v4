"""Evaluate keyed point forecasts on fixed paired NET-V4 histories and targets.

Python >=3.8, NumPy >=1.20. No fitting, model-name branches, or source-project imports.
This file scores constructed contrasts; it does not estimate physical incident effects.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ('B', 'P', 'R', 'F')
COHORTS = ('all', 'train_origin', 'validation_origin', 'test_origin')
TIMINGS = ('all', 'historical_only', 'crossing_cut', 'future_only')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('No scoring rows; the declared population is empty')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def identifiers(values, label):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'US':
        raise ValueError(label + ' must be a one-dimensional string array')
    keys = values.astype(str).tolist()
    if any(not key.strip() for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(label + ' contains empty or duplicate IDs')
    return keys


def numeric(values, shape, label):
    values = np.asarray(values)
    if values.shape != shape or values.dtype.kind not in 'fiu':
        raise ValueError('%s: expected real numeric shape %s, got %s' % (label, shape, values.shape))
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(label + ' contains nonfinite values')
    return values


def load_task(folder):
    folder = Path(folder)
    x = load_npz(folder / 'inputs.npz')
    ev = load_npz(folder / 'evaluation.npz')
    expected = identifiers(ev['sample_id'], 'evaluation sample_id')
    input_ids = identifiers(x['sample_id'], 'input sample_id')
    if set(input_ids) != set(expected):
        raise ValueError('Input/evaluation ID sets differ')
    history = numeric(x['history'], (len(input_ids), 12), 'history')
    target = numeric(ev['target'], (len(expected), 12), 'target')
    raw_context = ev['context_id']
    if raw_context.shape != (len(expected),) or raw_context.dtype.kind not in 'iu':
        raise ValueError('context_id must be an integer vector')
    contexts = np.unique(raw_context)
    if len(contexts) * 4 != len(expected):
        raise ValueError('Every context must have four arms')
    if ev['arm'].shape != (len(expected),) or set(ev['arm'].tolist()) != set(ARMS):
        raise ValueError('Arms must be B, P, R, F')
    if ev['pulse'].shape != target.shape or ev['pulse'].dtype != np.bool_:
        raise ValueError('pulse must be a boolean target-shaped mask')
    if ev['phase'].shape != target.shape or set(ev['phase'].ravel()) - {'pre', 'pulse', 'post'}:
        raise ValueError('phase must use pre, pulse, post')
    for key, allowed in [('cohort', COHORTS[1:]), ('timing', TIMINGS[1:])]:
        if ev[key].shape != (len(expected),) or set(ev[key]) - set(allowed):
            raise ValueError('Invalid ' + key)
    positions = {}
    for row, (context, arm) in enumerate(zip(raw_context.tolist(), ev['arm'].tolist())):
        key = (context, arm)
        if key in positions:
            raise ValueError('Duplicate context/arm pair')
        positions[key] = row
    try:
        grid = np.array([[positions[(int(c), arm)] for c in contexts] for arm in ARMS])
    except KeyError as exc:
        raise ValueError('Missing context/arm pair') from exc
    input_position = {key: i for i, key in enumerate(input_ids)}
    aligned_x = history[[input_position[key] for key in expected]][grid]
    for key in ['pulse', 'phase', 'cohort', 'timing']:
        value = ev[key][grid]
        if not np.array_equal(value, np.broadcast_to(value[0], value.shape)):
            raise ValueError('Evaluation context metadata differs between arms: ' + key)
    if not np.array_equal(ev['pulse'], ev['phase'] == 'pulse'):
        raise ValueError('Pulse mask and phase labels disagree')
    return {'ids': expected, 'grid': grid, 'history': aligned_x, 'target': target[grid],
            'contexts': contexts, **{key: ev[key][grid][0] for key in ['pulse', 'phase', 'cohort', 'timing']}}


def align_prediction(task, path):
    saved = load_npz(path)
    keys = identifiers(saved['sample_id'], 'prediction sample_id')
    expected = set(task['ids'])
    if set(keys) != expected:
        raise ValueError('Prediction ID mismatch: %d missing, %d unexpected' %
                         (len(expected - set(keys)), len(set(keys) - expected)))
    values = numeric(saved['prediction'], (len(keys), task['target'].shape[-1]), 'prediction')
    position = {key: i for i, key in enumerate(keys)}
    return values[[position[key] for key in task['ids']]][task['grid']]


def score(task, prediction, prediction_mode, atol=1e-6, rtol=1e-7):
    if prediction_mode not in ('deterministic', 'single_realization_stochastic'):
        raise ValueError('Declare deterministic or single_realization_stochastic predictions')
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    y = task['target']
    prediction = numeric(prediction, y.shape, 'aligned prediction')
    populations = {'all_steps': np.ones(y.shape[1:], dtype=bool), 'marked_steps': task['pulse'],
                   'pre_pulse_steps': task['phase'] == 'pre', 'post_pulse_steps': task['phase'] == 'post'}
    cross, decomposition, counts = [], [], []
    for population, mask in populations.items():
        for cohort in COHORTS:
            for timing in TIMINGS:
                selected = np.ones(y.shape[1], dtype=bool)
                if cohort != 'all':
                    selected &= task['cohort'] == cohort
                if timing != 'all':
                    selected &= task['timing'] == timing
                chosen = mask & selected[:, None]
                n = int(chosen.sum())
                header = {'population': population, 'cohort': cohort, 'timing_group': timing,
                          'contexts': int(chosen.any(1).sum()), 'points': n}
                counts.append({**header, 'status': 'scored' if n else 'empty'})
                if not n:
                    continue
                losses = np.empty((4, 4))
                for hi, ha in enumerate(ARMS):
                    for ti, ta in enumerate(ARMS):
                        error = (prediction[hi] - y[ti])[chosen]
                        absolute = np.abs(error)
                        denominator = float(np.abs(y[ti][chosen]).sum())
                        losses[hi, ti] = float(absolute.mean())
                        cross.append({**header, 'history_arm': ha, 'target_arm': ta,
                                      'mae': float(absolute.mean()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
                                      'wmape': 100 * float(absolute.sum()) / denominator if denominator else None})
                bb, bf, ff, fb, pp = losses[0, 0], losses[0, 3], losses[3, 3], losses[3, 0], losses[1, 1]
                decomposition.append({**header, 'background_mae': float(bb), 'full_mae': float(ff),
                    'total_D': float(ff - bb), 'target_T': float(bf - bb), 'history_gain_A': float(bf - ff),
                    'reverse_history_H': float(fb - bb), 'reverse_target_Q': float(ff - fb),
                    'recovery_delta_F_P': float(ff - pp)})
    equal = (task['history'][0] == task['history'][3]).all(axis=1)
    difference = np.abs(prediction[0, equal] - prediction[3, equal])
    tolerance = atol + rtol * np.maximum(np.abs(prediction[0, equal]), np.abs(prediction[3, equal]))
    exceeds = int((difference > tolerance).sum())
    status = ('not_tested_single_stochastic_realization' if prediction_mode != 'deterministic' else
              'no_equal_histories' if not equal.any() else
              'inspect_numerics_state_or_extra_inputs' if exceeds else 'consistent_with_declared_determinism')
    check = {'prediction_mode': prediction_mode, 'identical_history_contexts': int(equal.sum()),
             'max_B_F_prediction_difference': float(difference.max()) if difference.size else None,
             'points_exceeding_tolerance': exceeds, 'atol': atol, 'rtol': rtol, 'status': status,
             'interpretation': 'A discrepancy is not proof of target leakage; inspect state and numerics. '
                               'A stochastic predictor needs replicated controlled draws.'}
    return {'cross_losses': cross, 'decompositions': decomposition, 'population_counts': counts,
            'equal_history': check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prediction-mode', required=True, choices=['deterministic', 'single_realization_stochastic'])
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=1e-7)
    args = parser.parse_args()
    task = load_task(args.task)
    result = score(task, align_prediction(task, args.predictions), args.prediction_mode, args.atol, args.rtol)
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ['cross_losses', 'decompositions', 'population_counts']:
        write_csv(args.out / (name + '.csv'), result[name])
    dump(args.out / 'equal_history.json', result['equal_history'])
    print(json.dumps({'status': 'scored', 'contexts': len(task['contexts']),
                      'decompositions': len(result['decompositions'])}))


if __name__ == '__main__':
    main()
