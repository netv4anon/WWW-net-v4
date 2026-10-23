"""Recompute the two paper examples from delivered, already saved evidence.

No model fitting or inference. Release losses and masks are sealed intermediate
evidence; this command does not re-audit the unbundled upstream source.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.absolute().resolve()
    require(out != ROOT and ROOT not in out.parents, 'Output must be outside the resource')
    require(not out.exists(), 'Choose a new output directory')
    manifest = read(ROOT/'MANIFEST.json')
    for row in manifest['files']:
        path = (ROOT/row['path']).resolve()
        require(ROOT in path.parents, 'Manifest path outside resource')
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], row['path'])

    paired = ROOT/'components/netv4'
    with np.load(paired/'task/evaluation.npz', allow_pickle=False) as z:
        ids, context, arm = z['sample_id'], z['context_id'], z['arm']
        targets = z['target'].astype(np.float64)
    bg_rows = np.flatnonzero(arm == 'B')
    full_by_context = {int(context[i]): i for i in np.flatnonzero(arm == 'F')}
    full_rows = np.array([full_by_context[int(context[i])] for i in bg_rows])
    bg_ids = context[bg_rows]
    require(len(bg_rows) == 256 and len(set(bg_ids.tolist())) == 256, 'Context contract')
    with np.load(paired/'task/recovery_targets.npz', allow_pickle=False) as z:
        residual_index = {int(k): i for i,k in enumerate(z['context_id'])}
        residual = z['target'][[residual_index[int(k)] for k in bg_ids]].astype(np.float64)
    capsule = ROOT/'evidence/worked_example_replay'
    contexts = read(capsule/'contexts.json')
    require({int(c['pair_id']) for c in contexts} == set(bg_ids.tolist()), 'Parameter IDs')
    with np.load(capsule/'construction_full_precision.npz', allow_pickle=False) as z:
        background, pulse, saved_alternative = z['background'], z['pulse_unclipped'], z['alternative']
    reconstructed = []
    for c in contexts:
        k = int(c['pair_id'])
        e = int(c['ev_start']) + int(c['ev_duration'])
        require(e >= 12, 'Target-only replacement requires unchanged histories')
        tau = max(2., float(c['response_sec'])/900.)
        v = background[k]
        alt = pulse[k].copy()
        for t in range(e, 24):
            alt[t] = v[t] + (pulse[k,e-1]-v[e-1])*np.exp(-(t-e)/tau)
        alt = np.clip(alt, 3., 1.3*v.max())
        require(np.allclose(alt, saved_alternative[k], rtol=0, atol=1e-11), 'Residual construction')
        reconstructed.append(alt[12:].astype(np.float32))
    parameter_rows = {int(c['pair_id']): i for i,c in enumerate(contexts)}
    reconstructed = np.stack(reconstructed)[[parameter_rows[int(k)] for k in bg_ids]]
    require(np.array_equal(reconstructed.astype(np.float64), residual), 'Exported residual targets')

    wanted_ids = ids.tolist()
    model_rows = []
    prediction_bindings = {}
    expected = read(ROOT/'worked_examples/USE_CASE_EVIDENCE.json')
    expected_rows = expected['response']['all_frozen_model_rows']
    for model in read(paired/'MODELS.json'):
        path = paired/model['file']
        prediction_bindings[model['id']] = digest(path)
        require(prediction_bindings[model['id']] == model['sha256'], 'Saved prediction digest')
        with np.load(path, allow_pickle=False) as z:
            index = {k:i for i,k in enumerate(z['sample_id'].tolist())}
            require(set(index) == set(wanted_ids), 'Forecast IDs')
            predictions = z['prediction'][[index[k] for k in wanted_ids]].astype(np.float64)
        pb, pf = predictions[bg_rows], predictions[full_rows]
        yb = targets[bg_rows]
        for law,yf in [('legacy',targets[full_rows]),('residual_recovery',residual)]:
            values = {'L_BB': float(np.abs(pb-yb).mean()), 'L_BF': float(np.abs(pb-yf).mean()),
                      'L_FF': float(np.abs(pf-yf).mean()), 'L_FB': float(np.abs(pf-yb).mean())}
            values.update(D=values['L_FF']-values['L_BB'], T=values['L_BF']-values['L_BB'],
                          A=values['L_BF']-values['L_FF'])
            meta = {k:model[k] for k in ['variant','run','model','seed']}
            reference = [q for q in expected_rows if q['law']==law and all(q[k]==v for k,v in meta.items())]
            require(len(reference)==1, 'One frozen primary row per configuration/law')
            require(all(abs(v-reference[0][k]) < 1e-10 for k,v in values.items()), 'Primary score agreement')
            model_rows.append(dict(meta, law=law, contexts=256, points=3072, **values))
    means = []
    for law in ['legacy','residual_recovery']:
        seeds = [q for q in model_rows if q['variant']=='working' and q['model']=='STID_NoCalendar' and q['law']==law]
        require({q['seed'] for q in seeds} == {1,2,3}, 'All mixed STID seeds')
        means.append({'law':law, 'seeds':3,
                      **{k:float(np.mean([q[k] for q in seeds])) for k in ['L_BB','L_BF','L_FF','D','T','A']},
                      'individual_seed_A':[q['A'] for q in seeds]})

    seal = read(capsule/'PREDICTION_SEAL.json')
    require(digest(capsule/'PREDICTIONS.npz') == seal['prediction_sha256'], 'Release prediction seal')
    with np.load(capsule/'ROW_METRICS.npz', allow_pickle=False) as z:
        row_ids, hit = z['query_rows'], z['history_hit']
        last_mae, lookup_mae = z['LastValue_mae'], z['ExactHistory_mae']
    with np.load(capsule/'PREDICTIONS.npz', allow_pickle=False) as z:
        require(np.array_equal(z['query_rows'], row_ids), 'Release query ordering')
        require(np.array_equal(z['LastValue'][~hit],z['ExactHistory'][~hit]), 'Miss predictions identical')
    test = row_ids >= 4746
    groups = {}
    for name, mask in [('all',test),('hit',test & hit),('miss',test & ~hit)]:
        base, prediction = float(last_mae[mask].mean()), float(lookup_mae[mask].mean())
        groups[name] = {'rows':int(mask.sum()), 'lastvalue_mae':base, 'lookup_mae':prediction,
                        'gain_percent':100*(base-prediction)/base}
        reference = expected['reuse']['fixed_test_groups'][name]
        require(all(abs(groups[name][k]-reference[k]) < 1e-10 for k in reference), 'Release saved-loss agreement')
    require(groups['hit']['rows']==93 and groups['miss']['rows']==746, 'Release group sizes')

    station = ROOT/'components/station_exclusion/expected'
    with np.load(station/'STATION_TRAIN_MASKS.npz', allow_pickle=False) as z:
        retained = z['keep'].sum(axis=1)
        configs = z['configs'].tolist()
    with np.load(station/'QUERY_MEMBERSHIP.npz', allow_pickle=False) as z:
        require(z['configs'].tolist()==configs, 'Station configuration order')
        after_hits = (z['after'] > 0).sum(axis=(1,2))
        flow_hits = (z['flow_after'] > 0).sum(axis=(1,2))
    support = {name:{'empty_stations':int((retained[i]==0).sum()),
                       'minimum_retained_rows':int(retained[i].min()),
                       'declared_key_hits_after':int(after_hits[i]),
                       'flow_only_hits_after':int(flow_hits[i])} for i,name in enumerate(configs)}
    require(support['Original__flow']['empty_stations']==51 and
            support['TargetSupport__flow']['empty_stations']==51, 'Empty support')
    require(support['TargetSupport__flow_history_calendar']['minimum_retained_rows']==1679 and
            support['TargetSupport__flow_history_calendar']['flow_only_hits_after']==75167, 'Calendar support tradeoff')
    historical = read(capsule/'FINDINGS_VERIFICATION.json')
    station_score = read(ROOT/'components/station_lookup/EXPECTED_RESULTS.json')['scores']['TargetSupport__flow']['all']
    require(abs(station_score['mae_gain_percent']-4.97225961064638)<1e-10, 'Archived local lookup result')

    summary = {'saved_prediction_configurations':11,'primary_response_rows':22,
               'prediction_sha256':prediction_bindings,'mixed_STID_three_seed_mean':means,
               'residual_targets_reconstructed_from_preserved_parameters':256,
               'release_saved_loss_groups':groups,'release_saved_forecast_seal_verified':True,
               'release_record_uniqueness':{'historical_verification_report':historical,
                   'raw_source_reaudited_by_this_command':False},
               'station_support_from_saved_masks':support,
               'station_lookup_aggregate_from_archived_verified_results':station_score,
               'full_release_audit_requires_external_source':True,
               'new_fits':0,'new_inference':0,'new_source_records':0,'new_scientific_configurations':0,
               'scope':'Saved-forecast scoring, target reconstruction and saved-support arithmetic; not independent physical validation or a new upstream audit'}
    out.mkdir(parents=True)
    (out/'EXAMPLE_RESULTS.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'RESPONSE_ALL_CONFIGURATIONS.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(model_rows[0]))
        writer.writeheader();writer.writerows(model_rows)
    print(json.dumps({'passed':True,'response_configurations':11,'response_rows':22,
                      'residual_targets_reconstructed':256,'test_rows':839,'test_history_hits':93,
                      'test_lookup_gain_percent':groups['all']['gain_percent'],
                      'flow_exclusion_empty_stations':51,'raw_source_audit_executed':False}))

if __name__ == '__main__':
    main()
