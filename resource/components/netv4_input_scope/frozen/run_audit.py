# -*- coding: utf-8 -*-
"""One history-only proximity rule, then a sealed subset of existing forecasts."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESOURCE = ROOT.parent / 'NETV4_WWW_final_reading_20261010/resource'
STAGING = ROOT.parent / 'nv4_gwn_training_staging_validation_20261010'
MV = RESOURCE / 'components/netv4_mv/task'
GWN = RESOURCE / 'components/netv4_gwn'
ARMS = ('B', 'C', 'F')
ARM_INDEX = (0, 4, 3)
VARIANTS = ('packaged', 'working')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def writecsv(path, rows):
    assert rows
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def aligned(order_ids, ids, values):
    assert len(np.unique(ids)) == len(ids)
    order = np.argsort(ids)
    pos = np.searchsorted(ids[order], order_ids)
    assert np.all(pos < len(ids)) and np.array_equal(ids[order][pos], order_ids)
    return values[order[pos]]

def nearest(reference, query, device):
    """Exact exhaustive search in float64, with direct distances for selected neighbours."""
    ref = torch.as_tensor(reference, dtype=torch.float64, device=device)
    ref_sq = (ref * ref).sum(1)
    distances = np.empty(len(query), dtype=np.float64)
    indices = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), 512):
        end = min(start + 512, len(query))
        q = torch.as_tensor(query[start:end], dtype=torch.float64, device=device)
        sq = (q * q).sum(1)[:, None] + ref_sq[None, :] - 2 * q.mm(ref.t())
        idx = sq.argmin(1).cpu().numpy()
        indices[start:end] = idx
        # Avoid cancellation in near-identical histories.
        delta = query[start:end] - reference[idx]
        distances[start:end] = np.sqrt(np.mean(delta * delta, axis=1))
    del ref, ref_sq
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return distances, indices

def main():
    assert not (ROOT / 'FROZEN_PLAN.json').exists(), 'No rerun or policy search.'
    start_time = time.time()
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = [ROOT / 'PROTOCOL_ZH.md', Path(__file__), MV / 'inputs.npz', MV / 'evaluation.npz',
             GWN / 'PINS.json', GWN / 'evaluation/primary_seed_rows.csv']
    for variant in VARIANTS:
        paths += [STAGING / ('data/' + variant + '/' + name) for name in ('history.npz', 'splits.npz')]
        paths += [GWN / ('runs/' + variant + '__GraphWaveNet_seed' + str(seed) + '/mv_predictions.npz')
                  for seed in (1, 2, 3)]
    dump(ROOT / 'FROZEN_PLAN.json', {
        'date': '2026-10-11', 'policy': 'PROTOCOL_ZH.md',
        'posthoc_fixed_rule_not_independent_preregistration': True,
        'representation': '108 ordered values from the model 12x9 history; training-global mean/std',
        'distance': 'training-history nearest-neighbour standardized RMS; exhaustive float64 search',
        'threshold': 'validation-to-training 0.95 quantile, linear interpolation',
        'included_arms': list(ARMS), 'max_arm_distance_not_above_threshold': True,
        'coverage_gate': {'minimum_primary_scenario_fraction': .5,
                          'minimum_profiles_with_14_of_28_fraction': .8},
        'new_fits': 0, 'new_inference': 0, 'new_data': 0,
        'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths]
    })
    print(json.dumps({'stage': 'plan_frozen', 'device': str(device), 'policy_search': False}), flush=True)
    pins = json.loads((GWN / 'PINS.json').read_text(encoding='utf-8'))
    pin_map = {r['path']: r['sha256'] for r in pins['files']}
    # Read history and context metadata, not future losses, to fix masks.
    with np.load(MV / 'inputs.npz', allow_pickle=False) as z:
        input_ids, history = z['sample_id'].copy(), z['history'].copy()
    with np.load(MV / 'evaluation.npz', allow_pickle=False) as z:
        evaluation_ids = z['sample_id'].copy()
        context = z['context_id'].copy()
        arm = z['arm'].copy()
        profile = z['profile_index'].copy()
        timing = z['timing'].copy()
    n = len(profile)
    assert n == 11424 and len(np.unique(profile)) == 102
    assert np.array_equal(context, np.tile(np.arange(n), 6))
    assert np.array_equal(arm, np.repeat(np.arange(6), n))
    ordered_history = aligned(evaluation_ids, input_ids, history).reshape(6, n, 108).astype(np.float64)
    queries = ordered_history[list(ARM_INDEX)].reshape(-1, 108)
    primary = timing == 'cutoff_spanning'
    assert primary.sum() == 2856
    coverage_rows, profile_rows, distance_summaries, gates, masks = [], [], {}, {}, {}
    numeric_checks = []
    for variant in VARIANTS:
        for name in ('history.npz', 'splits.npz'):
            relative = 'data/' + variant + '/' + name
            assert sha(STAGING / relative) == pin_map[relative], relative
        with np.load(STAGING / ('data/' + variant + '/history.npz'), allow_pickle=False) as z:
            original = z['history'].astype(np.float64)
        with np.load(STAGING / ('data/' + variant + '/splits.npz'), allow_pickle=False) as z:
            train_index, val_index = z['train'].copy(), z['validation'].copy()
        training = original[train_index].reshape(-1, 108)
        mean, scale = float(training.mean()), float(training.std(ddof=0))
        assert np.isfinite(scale) and scale > 0
        reference = (training - mean) / scale
        val = (original[val_index].reshape(-1, 108) - mean) / scale
        val_distance, val_neighbour = nearest(reference, val, device)
        threshold = float(np.quantile(val_distance, .95, method='linear'))
        print(json.dumps({'stage': 'threshold_fixed_from_validation', 'variant': variant,
                          'threshold': threshold, 'training_rows': len(train_index),
                          'validation_rows': len(val_index)}), flush=True)
        q = (queries - mean) / scale
        distance, neighbour = nearest(reference, q, device)
        distance = distance.reshape(3, n)
        nearest_index = train_index[neighbour.reshape(3, n)]
        included = (distance <= threshold).all(axis=0)
        masks[variant] = included
        np.savez_compressed(ROOT / (variant + '_INPUT_SELECTION.npz'),
            arm=np.array(ARMS), distance=distance, nearest_training_index=nearest_index,
            included=included, profile_index=profile, timing=timing,
            validation_distance=val_distance, validation_nearest_training_index=train_index[val_neighbour])
        # Independent CPU brute-force checks on evenly spaced query positions, not favourable outcomes.
        max_error = 0.
        for index in np.linspace(0, len(q) - 1, 16, dtype=int):
            direct = np.sqrt(((reference - q[index]) ** 2).mean(axis=1)).min()
            max_error = max(max_error, abs(float(direct) - float(distance.reshape(-1)[index])))
        assert max_error <= 1e-9
        numeric_checks.append({'variant': variant, 'cpu_bruteforce_queries': 16,
                               'max_distance_discrepancy': max_error})
        distance_summaries[variant] = {
            'training_mean': mean, 'training_std_population': scale, 'threshold_q95': threshold,
            'validation_distance_quantiles': {str(x): float(np.quantile(val_distance, x)) for x in (0, .5, .95, 1)},
            'primary_arm_distance_quantiles': {a: {str(x): float(np.quantile(distance[i, primary], x))
                                                      for x in (0, .5, .95, 1)} for i, a in enumerate(ARMS)}
        }
        for tag in ('history_completed', 'cutoff_spanning', 'future_early', 'future_late'):
            group = timing == tag
            sizes = []
            for p in range(102):
                group_p = group & (profile == p)
                total, kept = int(group_p.sum()), int((group_p & included).sum())
                assert total == 28
                sizes.append(kept)
                profile_rows.append({'variant': variant, 'timing': tag, 'profile_index': p,
                                     'total': total, 'included': kept, 'excluded': total - kept})
            qualified = int((np.array(sizes) >= 14).sum())
            row = {'variant': variant, 'timing': tag, 'total': int(group.sum()),
                   'included': int((group & included).sum()), 'excluded': int((group & ~included).sum()),
                   'included_fraction': float(included[group].mean()),
                   'profiles_with_any_included': int((np.array(sizes) > 0).sum()),
                   'profiles_with_14_or_more': qualified,
                   'profile_fraction_with_14_or_more': qualified / 102}
            coverage_rows.append(row)
            if tag == 'cutoff_spanning':
                gates[variant] = {'scenario_coverage_pass': row['included_fraction'] >= .5,
                                 'profile_coverage_pass': qualified / 102 >= .8,
                                 'coverage_pass': row['included_fraction'] >= .5 and qualified / 102 >= .8}
        print(json.dumps({'stage': 'input_selection_sealed_for_variant', 'variant': variant,
                          'primary_included': int((primary & included).sum()),
                          'primary_total': int(primary.sum()), 'coverage_gate': gates[variant]}), flush=True)
    writecsv(ROOT / 'COVERAGE.csv', coverage_rows)
    writecsv(ROOT / 'PROFILE_COVERAGE.csv', profile_rows)
    dump(ROOT / 'DISTANCE_SUMMARIES.json', distance_summaries)
    dump(ROOT / 'NUMERIC_DISTANCE_CHECKS.json', numeric_checks)
    dump(ROOT / 'SELECTION_SEAL.json', {
        'stage': 'before_saved_forecast_subset_scoring',
        'outcomes_used_to_select_inputs': False,
        'files': [{'path': p.name, 'sha256': sha(p)} for p in sorted(ROOT.glob('*_INPUT_SELECTION.npz'))],
        'coverage_gates': gates
    })

    # Only now read saved forecasts and targets for the fixed populations.
    with np.load(MV / 'evaluation.npz', allow_pickle=False) as z:
        targets = z['target'][:, :, 0].reshape(6, n, 12).astype(np.float64)
        residual = z['residual_full_target'][:, :, 0].astype(np.float64)
    with (GWN / 'evaluation/primary_seed_rows.csv').open(encoding='utf-8', newline='') as stream:
        frozen_rows = list(csv.DictReader(stream))
    rows, means, profile_scores, full_checks = [], [], [], []
    saved = {}
    for variant in VARIANTS:
        for seed in (1, 2, 3):
            prediction_path = GWN / ('runs/' + variant + '__GraphWaveNet_seed' + str(seed) + '/mv_predictions.npz')
            with np.load(prediction_path, allow_pickle=False) as z:
                prediction = aligned(evaluation_ids, z['sample_id'], z['prediction']).reshape(6, n, 12).astype(np.float64)
            for law, full_target in (('legacy', targets[3]), ('residual', residual)):
                losses = {
                    'L_BB': np.abs(prediction[0] - targets[0]).mean(1),
                    'L_BF': np.abs(prediction[0] - full_target).mean(1),
                    'L_CF': np.abs(prediction[4] - full_target).mean(1),
                    'L_NF': np.abs(prediction[5] - full_target).mean(1),
                    'L_FF': np.abs(prediction[3] - full_target).mean(1)
                }
                saved[(variant, seed, law)] = losses
                frozen = [r for r in frozen_rows if r['variant'] == variant and int(r['seed']) == seed
                          and r['target_law'] == law and r['scope'] == 'cutoff_spanning' and r['phase'] == 'all_points']
                assert len(frozen) == 1
                discrepancy = max(abs(float(losses[k][primary].mean()) - float(frozen[0][k])) for k in losses)
                assert discrepancy <= 1e-9, (variant, seed, law, discrepancy)
                full_checks.append({'variant': variant, 'seed': seed, 'target_law': law,
                                    'max_full_score_discrepancy': discrepancy})
                for name, mask in (('full', primary), ('included', primary & masks[variant]),
                                   ('excluded', primary & ~masks[variant])):
                    row = {'variant': variant, 'seed': seed, 'target_law': law, 'population': name,
                           'contexts': int(mask.sum())}
                    for k in losses:
                        row[k] = float(losses[k][mask].mean()) if mask.any() else None
                    row['A'] = row['L_BF'] - row['L_FF'] if mask.any() else None
                    row['other_response'] = row['L_CF'] - row['L_FF'] if mask.any() else None
                    row['conditional_error_increase_percent'] = ((row['L_FF'] - row['L_CF']) / row['L_CF'] * 100
                                                                 if mask.any() and row['L_CF'] else None)
                    row['helpful_total_harmful_other'] = bool(row['A'] > 0 and row['other_response'] < 0) if mask.any() else None
                    rows.append(row)
        for law in ('legacy', 'residual'):
            loss_mean = {k: np.mean([saved[(variant, seed, law)][k] for seed in (1, 2, 3)], axis=0)
                         for k in ('L_BB', 'L_BF', 'L_CF', 'L_NF', 'L_FF')}
            for name, mask in (('full', primary), ('included', primary & masks[variant]),
                               ('excluded', primary & ~masks[variant])):
                row = {'variant': variant, 'target_law': law, 'population': name, 'contexts': int(mask.sum())}
                for k, values in loss_mean.items():
                    row[k] = float(values[mask].mean()) if mask.any() else None
                row['A'] = row['L_BF'] - row['L_FF'] if mask.any() else None
                row['other_response'] = row['L_CF'] - row['L_FF'] if mask.any() else None
                row['conditional_error_increase_percent'] = ((row['L_FF'] - row['L_CF']) / row['L_CF'] * 100
                                                             if mask.any() and row['L_CF'] else None)
                group_A, group_other = [], []
                for p in range(102):
                    p_mask = mask & (profile == p)
                    a = float((loss_mean['L_BF'][p_mask] - loss_mean['L_FF'][p_mask]).mean()) if p_mask.any() else None
                    other = float((loss_mean['L_CF'][p_mask] - loss_mean['L_FF'][p_mask]).mean()) if p_mask.any() else None
                    profile_scores.append({'variant': variant, 'target_law': law, 'population': name,
                                           'profile_index': p, 'contexts': int(p_mask.sum()), 'A': a, 'other_response': other})
                    if p_mask.any():
                        group_A.append(a)
                        group_other.append(other)
                row['represented_profiles'] = len(group_A)
                row['profile_equal_weight_A'] = float(np.mean(group_A)) if group_A else None
                row['profile_equal_weight_other_response'] = float(np.mean(group_other)) if group_other else None
                row['helpful_other_profiles'] = int((np.array(group_other) > 0).sum())
                row['harmful_other_profiles'] = int((np.array(group_other) < 0).sum())
                means.append(row)
    writecsv(ROOT / 'PRIMARY_SEED_SCORES.csv', rows)
    writecsv(ROOT / 'PRIMARY_LOSS_MEANS.csv', means)
    writecsv(ROOT / 'PRIMARY_PROFILE_LOSS_MEANS.csv', profile_scores)
    dump(ROOT / 'FULL_SCORE_VERIFICATION.json', full_checks)
    decisions = {}
    for variant in VARIANTS:
        included_rows = [r for r in rows if r['variant'] == variant and r['population'] == 'included']
        signs = bool(all(r['helpful_total_harmful_other'] is True for r in included_rows))
        decisions[variant] = {**gates[variant], 'all_three_seeds_both_laws_keep_disagreement': signs,
                              'branch': ('near_reference_evidence_supported' if signs else 'direction_not_preserved')
                                        if gates[variant]['coverage_pass'] else 'coverage_insufficient'}
    dump(ROOT / 'SUMMARY.json', {
        'date': '2026-10-11', 'question': 'MV history proximity and preserved response disagreement',
        'elapsed_seconds': time.time() - start_time, 'device': str(device),
        'decisions': decisions, 'primary_loss_means': [r for r in means if r['population'] == 'included'],
        'new_model_fits': 0, 'new_model_inference': 0, 'new_forecast_configurations': 0,
        'new_generated_scenarios': 0, 'new_observed_records': 0,
        'new_history_proximity_analysis': 1, 'new_saved_forecast_subset_scoring_run': 1,
        'same_distribution_proved': False, 'physical_realism_validated': False,
        'old_raw_source_lineage_recovered': False, 'frozen_paper_modified': False,
        'policy_or_threshold_search': False
    })
    print(json.dumps({'stage': 'complete', 'decisions': decisions,
                      'saved_forecast_subset_scoring_run': 1, 'new_fits': 0,
                      'elapsed_seconds': time.time() - start_time}), flush=True)

if __name__ == '__main__':
    main()
