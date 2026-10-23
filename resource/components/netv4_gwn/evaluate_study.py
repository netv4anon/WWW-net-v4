# -*- coding: utf-8 -*-
"""Evaluate all sealed models, retain unfavorable groups and independently check losses."""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from model_adapter import build_model, point_forecast

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''): h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def readcsv(path):
    with path.open(encoding='utf-8', newline='') as stream: return list(csv.DictReader(stream))

def csvout(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def align(path, expected):
    with np.load(path, allow_pickle=False) as z:
        ids, pred = z['sample_id'], z['prediction']
    pos = {k: i for i, k in enumerate(ids.tolist())}
    assert len(pos) == len(ids) and set(pos) == set(expected.tolist())
    return pred[[pos[k] for k in expected]].astype(np.float64)

def main():
    seal = json.loads((ROOT / 'FORECAST_SEAL.json').read_text(encoding='utf-8'))
    assert len(seal['models']) == 6
    assert seal['protocol_sha256'] == sha(ROOT / 'PROTOCOL_ZH.md')
    for m in seal['models']:
        assert sha(ROOT / m['prediction_file']) == m['prediction_sha256']
        assert sha(ROOT / m['weight_file']) == m['weight_sha256']
    out = ROOT / 'evaluation'
    out.mkdir(exist_ok=False)
    dump(out / 'EVALUATION_START.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'forecast_seal_sha256': sha(ROOT / 'FORECAST_SEAL.json'), 'all_six_forecasts_sealed_before_target_decoding': True})
    with np.load(ROOT / 'task/evaluation.npz', allow_pickle=False) as z: task = {k: z[k] for k in z.files}
    n = len(task['timing'])
    assert n == 11424
    old = task['target'].reshape(6, n, 12, 9)[:, :, :, 0].astype(np.float64)
    targets = {'legacy': old[3], 'residual': task['residual_full_target'][:, :, 0].astype(np.float64)}
    ids = task['sample_id']
    with np.load(ROOT / 'task/inputs.npz', allow_pickle=False) as z:
        ipos = {k: i for i, k in enumerate(z['sample_id'].tolist())}
        histories = z['history'][[ipos[k] for k in ids]].reshape(6, n, 12, 9)
    all_responses, all_accuracy, all_profiles, all_controls, original_test = [], [], [], [], []
    direct_checks, errors, functional = 0, [], []
    for m in seal['models']:
        destination = out / m['id']
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'score.py'),
            '--predictions', str(ROOT / m['prediction_file']), '--out', str(destination)],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout.decode('utf-8', errors='replace')
        rows = readcsv(destination / 'responses.csv')
        for r in rows:
            r.update({'model': m['id'], 'variant': m['variant'], 'kind': 'GraphWaveNet', 'seed': m['seed']})
        all_responses.extend(rows)
        for name, collection in (('accuracy.csv', all_accuracy), ('profile_response.csv', all_profiles), ('structural_controls.csv', all_controls)):
            records = readcsv(destination / name)
            for r in records: r['model'] = m['id']
            collection.extend(records)
        pred = align(ROOT / m['prediction_file'], ids).reshape(6, n, 12)
        for law, target in targets.items():
            for scope in ('all', 'cutoff_spanning'):
                take = np.ones(n, bool) if scope == 'all' else task['timing'] == scope
                losses = {}
                for key, view, truth in (('L_BB', 0, old[0]), ('L_BF', 0, target), ('L_CF', 4, target), ('L_NF', 5, target), ('L_FF', 3, target)):
                    per_context = np.abs(pred[view, take] - truth[take]).sum(axis=1) / 12
                    losses[key] = float(per_context.sum() / int(take.sum()))
                losses.update({'A': losses['L_BF'] - losses['L_FF'], 'T': losses['L_BF'] - losses['L_BB'],
                    'D': losses['L_FF'] - losses['L_BB'], 'neighbor_given_center': losses['L_CF'] - losses['L_FF'],
                    'neighbor_given_background_center': losses['L_BF'] - losses['L_NF']})
                scored = next(r for r in rows if r['target_law'] == law and r['scope'] == scope and r['phase'] == 'all_points')
                for key, value in losses.items():
                    error = abs(value - float(scored[key]))
                    assert error < 1e-10, (m['id'], law, scope, key, error)
                    direct_checks += 1
                    errors.append(error)
        future = np.isin(task['timing'], ['future_early', 'future_late'])
        future_diff = float(np.max(np.abs(pred[:, future] - pred[0, future])))
        assert future_diff <= 1e-3
        directory = ROOT / 'runs' / m['id']
        scaler = json.loads((directory / 'scaler.json').read_text(encoding='utf-8'))
        model = build_model('cpu')
        model.load_state_dict(torch.load(directory / 'best_weights.pt', map_location='cpu', weights_only=True))
        model.eval()
        spanning = np.flatnonzero(task['timing'] == 'cutoff_spanning')
        subset = spanning[np.linspace(0, len(spanning) - 1, 32, dtype=int)]
        inp = torch.tensor(histories[3, subset], dtype=torch.float32, requires_grad=True)
        center = (point_forecast(model, (inp - scaler['mean']) / scaler['std'])[:, :, 0] * scaler['std'] + scaler['mean']).clamp_min(0)
        center.sum().backward()
        cross_grad = float(inp.grad[:, :, 2:].abs().max())
        center_fixed_diff = float(np.abs(pred[3, spanning] - pred[4, spanning]).max())
        background_fixed_diff = float(np.abs(pred[0, spanning] - pred[5, spanning]).max())
        functional.append({'model': m['id'], 'F_vs_C_max_forecast_difference': center_fixed_diff,
            'B_vs_N_max_forecast_difference': background_fixed_diff, 'other_input_gradient_max': cross_grad,
            'gradient_sample_count': 32, 'future_equal_history_max_difference': future_diff,
            'functional_joint_input_demonstrated': center_fixed_diff > 1e-3 and cross_grad > 0,
            'gradient_selection': '32 evenly spaced spanning contexts; not selected on error or response'})
        with np.load(ROOT / 'data' / m['variant'] / 'targets.npz', allow_pickle=False) as z: truth = z['future']
        with np.load(directory / 'original_test_predictions.npz', allow_pickle=False) as z:
            test = z['test_indices']
            raw_pred = z['prediction'].astype(np.float64)
        error = raw_pred - truth[test, :, 0].astype(np.float64)
        original_test.append({'model': m['id'], 'variant': m['variant'], 'seed': m['seed'], 'test_contexts': len(test),
            'center_mae': float(np.abs(error).mean()), 'center_rmse': float(np.sqrt(np.square(error).mean()))})
        print(json.dumps({'evaluated': m['id'], 'functional_cross_input': functional[-1]['functional_joint_input_demonstrated']}), flush=True)
    csvout(out / 'responses.csv', all_responses)
    csvout(out / 'accuracy.csv', all_accuracy)
    csvout(out / 'profile_response.csv', all_profiles)
    csvout(out / 'structural_controls.csv', all_controls)
    csvout(out / 'original_test_scores.csv', original_test)
    csvout(out / 'functional_input_checks.csv', functional)
    means, seed_primary, comparisons = [], [], []
    for variant in ('packaged', 'working'):
        for law in ('legacy', 'residual'):
            selected = [r for r in all_responses if r['variant'] == variant and r['target_law'] == law and r['scope'] == 'cutoff_spanning' and r['phase'] == 'all_points']
            assert len(selected) == 3
            for r in selected:
                seed_primary.append({**r, 'conditional_error_increase_percent': 100 * (float(r['L_FF']) / float(r['L_CF']) - 1),
                    'helpful_total_harmful_other': float(r['A']) > 0 and float(r['neighbor_given_center']) < 0})
            fields = ['L_BB', 'L_BF', 'L_CF', 'L_NF', 'L_FF', 'A', 'T', 'D', 'neighbor_given_center', 'neighbor_given_background_center']
            avg = {key: float(np.mean([float(r[key]) for r in selected])) for key in fields}
            profile_means = []
            for group in range(102):
                values = [float(r['neighbor_given_center']) for r in all_profiles if r['model'].startswith(variant + '__') and r['target_law'] == law and int(r['profile_index']) == group]
                assert len(values) == 3
                profile_means.append(float(np.mean(values)))
            avg.update({'variant': variant, 'target_law': law, 'kind': 'GraphWaveNet seed-loss mean', 'seeds': 3,
                'helpful_total_harmful_other': avg['A'] > 0 and avg['neighbor_given_center'] < 0,
                'conditional_error_increase_percent': 100 * (avg['L_FF'] / avg['L_CF'] - 1),
                'helpful_profiles': sum(v > 0 for v in profile_means), 'harmful_profiles': sum(v < 0 for v in profile_means),
                'seed_conditional_response_min': min(float(r['neighbor_given_center']) for r in selected),
                'seed_conditional_response_max': max(float(r['neighbor_given_center']) for r in selected),
                'seeds_with_helpful_total_harmful_other': sum(float(r['A']) > 0 and float(r['neighbor_given_center']) < 0 for r in selected)})
            means.append(avg)
            for scope in ('all', 'history_completed', 'cutoff_spanning', 'future_early', 'future_late'):
                group = [r for r in all_responses if r['variant'] == variant and r['target_law'] == law and r['scope'] == scope and r['phase'] == 'all_points']
                comparisons.append({'variant': variant, 'target_law': law, 'scope': scope,
                    'A_mean': float(np.mean([float(r['A']) for r in group])),
                    'L_FF_mean': float(np.mean([float(r['L_FF']) for r in group])),
                    'neighbor_given_center_mean': float(np.mean([float(r['neighbor_given_center']) for r in group])),
                    'A_seed_min': min(float(r['A']) for r in group), 'A_seed_max': max(float(r['A']) for r in group)})
    csvout(out / 'primary_seed_rows.csv', seed_primary)
    csvout(out / 'primary_seed_loss_means.csv', means)
    csvout(out / 'timing_seed_loss_means.csv', comparisons)
    technical = seal['technical_layout_forecasts'][0]
    technical_out = out / 'technical_contiguous_seed1'
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'score.py'),
        '--predictions', str(ROOT / technical['prediction_file']), '--out', str(technical_out)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout.decode('utf-8', errors='replace')
    technical_primary = [r for r in readcsv(technical_out / 'responses.csv') if r['scope'] == 'cutoff_spanning' and r['phase'] == 'all_points']
    report = {'new_architectures': 1, 'completed_primary_fits': 6, 'completed_new_fits_total': 7,
        'technical_superseded_completed_fits': 1, 'technical_aborted_fit_attempts': 1,
        'total_fit_attempts_started': 8, 'new_primary_frozen_mv_forecast_configurations': 6,
        'new_frozen_mv_forecast_configurations_total': 7,
        'superseded_layout_primary_rows_retained_separately': technical_primary,
        'new_generated_scenarios': 0, 'new_observed_records': 0, 'inherited_mv_configurations': 13,
        'all_six_forecasts_sealed_before_evaluation': True,
        'direct_loss_values_checked': direct_checks, 'direct_loss_max_error': max(errors),
        'functional_input_checks': functional, 'primary_seed_loss_means': means,
        'timing_seed_loss_means': comparisons, 'original_test_scores': original_test,
        'all_seeds_and_unfavorable_results_retained': True,
        'independent_real_incidents_or_new_algorithm_claimed': False,
        'acceptance_probability_established': False, 'grade_upgrade_automatically_inferred': False}
    dump(out / 'FINDINGS.json', report)
    ledger = json.loads((ROOT / 'SCIENTIFIC_LEDGER.json').read_text(encoding='utf-8'))
    assert ledger['fits_completed'] == 6 and ledger['fit_attempts_started'] == 8
    ledger.update({'status': 'evaluated_awaiting_reader_replay_and_paper_update', 'evaluation_fits': 6,
        'direct_loss_values_checked': direct_checks})
    dump(ROOT / 'SCIENTIFIC_LEDGER.json', ledger)
    print(json.dumps({'primary_means': means, 'direct_loss_values_checked': direct_checks, 'direct_max_error': max(errors)}), flush=True)

if __name__ == '__main__': main()
