"""Verify concrete count, loss and configuration questions against frozen inputs and forecasts; no fitting."""
import collections
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import mmap
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究')
RESOURCE = ARCHIVE / 'workspace/NETV4_WWW_input_scope_20261011/resource'
PROGRAM = ARCHIVE / 'workspace/special veilcle predict/traffic_two_paper_program_2026-09-15/www_netv4_revision_2026-10-01'
OUT = ROOT / 'verification'

def fs(p):
    s = str(Path(p).absolute())
    return Path('\\\\?\\' + s) if os.name == 'nt' and not s.startswith('\\\\?\\') else Path(s)

def read(p):
    return json.loads(fs(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with fs(p).open('rb') as stream:
        for b in iter(lambda: stream.read(4194304), b''):
            h.update(b)
    return h.hexdigest()

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def verify_reuse():
    capsule = RESOURCE / 'components/release_audit'
    spec = read(capsule / 'SOURCE.json')
    source = PROGRAM / 'incident_split_feasibility_2026-10-03/sources/contra_complete.npy'
    assert fs(source).stat().st_size == spec['bytes'] and sha(source) == spec['sha256']
    with gzip.open(fs(capsule / 'ARRAY_OFFSETS.json.gz'), 'rb') as stream:
        offsets_raw = stream.read()
    assert hashlib.sha256(offsets_raw).hexdigest() == spec['offset_manifest_sha256']
    offsets = {(r['row'], r['role']):r for r in json.loads(offsets_raw.decode('utf-8-sig'))}
    with np.load(fs(RESOURCE / 'evidence/fixed_query_target_support/TRAIN_ROWS.npz'), allow_pickle=False) as z:
        masks = {k: z[k].copy() for k in ('Original', 'WholeTraffic', 'TargetSupport')}
    assert np.array_equal(masks['Original'], np.arange(3909))
    membership = {k:set(v.tolist()) for k,v in masks.items()}
    hits = {k:{'history':set(), 'target':set()} for k in masks}
    witnesses = {}
    qkeys, fullqueries = collections.defaultdict(list), collections.defaultdict(list)
    raw_checks = 0
    kept_target, kept_whole = [], []
    with fs(source).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        def raw(i, role):
            nonlocal raw_checks
            a = offsets[i, role]
            b = mm[a['offset']:a['offset']+a['bytes']]
            assert len(b) == 12*496*3*4 and hashlib.sha256(b).hexdigest() == a['sha256']
            raw_checks += 1
            return b
        for i in range(4746,5585):
            x, y = raw(i,'x_data'), raw(i,'y_data')
            fullqueries[x+y].append(i-4746)
            for step in range(12):
                qkeys[y[step*5952:(step+1)*5952]].append((i-4746)*12+step)
        for i in range(3909):
            x, y = raw(i,'x_data'), raw(i,'y_data')
            has_target_match = False
            if x+y not in fullqueries:
                kept_whole.append(i)
            for role, data in (('history',x), ('target',y)):
                for step in range(12):
                    b = data[step*5952:(step+1)*5952]
                    positions = qkeys.get(b, [])
                    if not positions:
                        continue
                    has_target_match = True
                    for condition in masks:
                        if i not in membership[condition]:
                            continue
                        hits[condition][role].update(positions)
                        for position in positions:
                            key = condition, role, position
                            if key not in witnesses:
                                witnesses[key] = {'training_row':i, 'training_role':role,
                                    'training_step':step, 'state_sha256':hashlib.sha256(b).hexdigest(),
                                    'training_state_byte_offset':offsets[i,'x_data' if role=='history' else 'y_data']['offset']+step*5952}
            if not has_target_match:
                kept_target.append(i)
            if i % 1000 == 0:
                emit({'stage':'raw_byte_matching', 'training_rows_checked':i+1})
    assert np.array_equal(kept_whole,masks['WholeTraffic'])
    assert np.array_equal(kept_target,masks['TargetSupport'])
    counts, query_lists, rows = {}, {}, []
    for condition, values in hits.items():
        x, y = values['history'], values['target']
        scopes = {'training_history':x, 'training_target':y, 'training_history_or_target':x|y}
        counts[condition] = {'training_rows':len(masks[condition])}
        query_lists[condition] = {}
        for scope, positions in scopes.items():
            queries = sorted({p//12 for p in positions})
            query_lists[condition][scope] = {'local_test_query_ids':queries,
                'global_test_query_rows':[i+4746 for i in queries], 'test_target_positions':sorted(positions)}
            counts[condition][scope] = {'queries':len(queries), 'positions':len(positions)}
        for role in ('history','target'):
            for p in sorted(values[role]):
                rows.append({'condition':condition, 'test_local_query':p//12,
                    'test_global_row':4746+p//12, 'test_step':p%12,
                    **witnesses[condition,role,p]})
    expected = {'Original':(3909,559,658,6816), 'WholeTraffic':(3812,541,642,6538), 'TargetSupport':(2665,0,0,0)}
    for k,(n,qt,qb,pb) in expected.items():
        assert counts[k]['training_rows']==n
        assert counts[k]['training_target']['queries']==qt
        assert counts[k]['training_history_or_target']=={'queries':qb,'positions':pb}
    monotonic = {}
    for scope in ('training_history','training_target','training_history_or_target'):
        original = set(query_lists['Original'][scope]['local_test_query_ids'])
        whole = set(query_lists['WholeTraffic'][scope]['local_test_query_ids'])
        support = set(query_lists['TargetSupport'][scope]['local_test_query_ids'])
        assert support <= whole <= original
        monotonic[scope] = [len(original),len(whole),len(support)]
    write_csv('MATCH_WITNESSES.csv', rows)
    dump('MATCH_QUERY_IDS.json', query_lists)
    result = {'status':'PASS', 'source_sha256':spec['sha256'], 'source_version':spec,
        'test_interval_global_half_open':[4746,5585], 'train_interval_global_half_open':[0,3909],
        'shape_per_array':[12,496,3], 'dtype':'little-endian float32',
        'fields':'all 496 flow values and both calendar channels; exact complete-state bytes',
        'calendar_included':True, 'rounding_or_tolerance':None,
        'counting_unit':'query: any of its 12 targets matches; position: each test horizon counted once',
        'direction':'test target -> retained training history and/or target, as specified',
        'counts':counts, 'fixed_scope_query_monotonicity':monotonic,
        'reconstructed_masks_equal_frozen_masks':True, 'source_arrays_digest_checked':raw_checks,
        'verification_is_new_independent_cohort':False}
    dump('REUSE_SCOPE_VERIFIED.json',result)
    emit({'stage':'reuse_verified','counts':counts})
    return result

def verify_losses():
    mv = RESOURCE / 'components/netv4_mv'
    models = read(mv/'predictions/MODELS.json')
    model = next(r for r in models if r['id']=='working__RidgeLocal9')
    prediction_path = mv / model['file']
    assert sha(prediction_path)==model['sha256']
    with np.load(fs(mv/'task/evaluation.npz'),allow_pickle=False) as z:
        ids=z['sample_id']; n=len(z['timing']); use=z['timing']=='cutoff_spanning'
        y=z['target'].reshape(6,n,12,9)[3,:,:,0].astype(np.float64)
        profiles=z['profile_index'].copy(); backgrounds=z['background_id'].copy()
    with np.load(fs(prediction_path),allow_pickle=False) as z:
        index={v:i for i,v in enumerate(z['sample_id'].tolist())}
        assert len(index)==len(ids) and set(index)==set(ids.tolist())
        prediction=z['prediction'][[index[v] for v in ids.tolist()]].astype(np.float64).reshape(6,n,12)
    cf=np.abs(prediction[4,use]-y[use]).mean(axis=1)
    ff=np.abs(prediction[3,use]-y[use]).mean(axis=1)
    cfm,ffm=float(cf.mean()),float(ff.mean())
    gain=cfm-ffm
    with fs(mv/'scores/responses.csv').open(encoding='utf-8',newline='') as stream:
        row=next(r for r in csv.DictReader(stream) if r['model']==model['id'] and r['target_law']=='legacy'
                 and r['scope']=='cutoff_spanning' and r['phase']=='all_points')
    assert abs(cfm-float(row['L_CF']))<1e-12 and abs(ffm-float(row['L_FF']))<1e-12
    assert abs(gain-float(row['neighbor_given_center']))<1e-12
    assert len(cf)==2856 and format(gain,'.4f')=='-0.0823'
    write_csv('RIDGE_UNROUNDED_PER_CONTEXT.csv',[
        {'context_id':int(i), 'profile_index':int(profiles[i]), 'background_id':int(backgrounds[i]),
         'L_CF':float(a), 'L_FF':float(b), 'conditional_response':float(a-b)}
        for i,a,b in zip(np.flatnonzero(use),cf,ff)])
    observed=RESOURCE/'components/observed'
    with np.load(fs(observed/'saved_predictions.npz'),allow_pickle=False) as z:
        names=z['models'].tolist(); pids=z['ids'].tolist(); forecasts=z['predictions'].astype(np.float64)
    with np.load(fs(observed/'data/targets.npz'),allow_pickle=False) as z:
        wanted=z['ids'].tolist(); truth=z['y'].astype(np.float64)
    lookup={v:i for i,v in enumerate(pids)}
    assert len(lookup)==len(wanted) and set(lookup)==set(wanted)
    forecasts=forecasts[:,[lookup[v] for v in wanted]]
    losses=np.abs(forecasts-truth[None,:,:]).mean(axis=2)
    with fs(observed/'data/episodes.csv').open(encoding='utf-8',newline='') as stream:
        episodes=list(csv.DictReader(stream))
    pairs=read(observed/'data/pairs.json')
    obs_rows=[]
    for seed in (1,2,3):
        modelname='STID_seed%d'%seed
        err=losses[names.index(modelname)]
        for j,pair in enumerate(pairs):
            case=episodes[pair['case_episode_row']]
            controls=[episodes[i] for i in pair['control_episode_rows']]
            cp=float(err[int(case['pre_prediction_row'])]); ca=float(err[int(case['post_prediction_row'])])
            bp=float(np.mean([err[int(c['pre_prediction_row'])] for c in controls]))
            ba=float(np.mean([err[int(c['post_prediction_row'])] for c in controls]))
            obs_rows.append({'seed':seed,'case_index':j,'station_id':pair['station_id'],
                'controls':len(controls),'case_pre':cp,'control_pre':bp,'case_post':ca,'control_post':ba,
                'delta_pre':cp-bp,'delta_post':ca-ba,'change':(ca-cp)-(ba-bp)})
    pre=float(np.mean([r['delta_pre'] for r in obs_rows]))
    post=float(np.mean([r['delta_post'] for r in obs_rows]))
    change=post-pre
    old=next(r for r in read(observed/'RESULTS.json')['aggregate'] if r['model']=='STID_seed_mean')
    assert all(abs(value-old[field])<1e-12 for field,value in [('delta_pre',pre),('delta_post',post),('change',change)])
    assert len(pairs)==268 and format(change,'.3f')=='3.905'
    write_csv('OBSERVED_UNROUNDED_PER_CASE_SEED.csv',obs_rows)
    result={'status':'PASS', 'ridge':{'model':model['id'],'target_law':'legacy','cohort':'cutoff_spanning',
        'contexts':2856,'horizons_per_context':12,'weighting':'uniform context and horizon',
        'L_CF':cfm,'L_FF':ffm,'difference':gain,'printed_difference':format(gain,'.4f'),
        'prediction_sha256':sha(prediction_path),'evaluation_sha256':sha(mv/'task/evaluation.npz')},
        'observed':{'cases':268,'seeds':[1,2,3],
        'control_memberships':sum(len(p['control_episode_rows']) for p in pairs),
        'weighting':'equal cases; controls averaged within case; equal seed loss weights',
        'delta_pre':pre,'delta_post':post,'difference':change,'printed_difference':format(change,'.3f'),
        'predictions_sha256':sha(observed/'saved_predictions.npz'),'targets_sha256':sha(observed/'data/targets.npz'),
        'pairs_sha256':sha(observed/'data/pairs.json')},
        'new_predictions_or_training':0,'new_independent_confirmatory_evidence':False}
    dump('UNROUNDED_LOSSES_VERIFIED.json',result)
    emit({'stage':'unrounded_losses_verified', **result})
    return result

def verify_configuration():
    gwn=RESOURCE/'components/netv4_gwn'
    import torch
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('verified_gwn_adapter',fs(gwn/'model_adapter.py'))
    adapter=importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    model=adapter.build_model('cpu')
    parameter_count=sum(v.numel() for v in model.parameters())
    assert model.receptive_field==13 and parameter_count==272640
    expected_shape={k:tuple(v.shape) for k,v in model.state_dict().items()}
    rows=[]
    archive=RESOURCE/'archives/netv4_review_candidate_v0_3.zip'
    pins=read(gwn/'PINS.json')
    assert sha(archive)==pins['legacy_archive_sha256']
    hashes={r['path']:r['sha256'] for r in pins['files']}
    for variant in ('packaged','working'):
        with zipfile.ZipFile(fs(archive)) as bundle:
            hk,sk='data/%s/history.npz'%variant,'data/%s/splits.npz'%variant
            hb,sb=bundle.read(hk),bundle.read(sk)
            assert hashlib.sha256(hb).hexdigest()==hashes[hk]
            assert hashlib.sha256(sb).hexdigest()==hashes[sk]
        with np.load(io.BytesIO(hb),allow_pickle=False) as z:
            history=z['history'].astype(np.float32)
        with np.load(io.BytesIO(sb),allow_pickle=False) as z:
            train=z['train']
        mean,std=float(history[train].mean()),float(history[train].std())+1e-6
        for seed in (1,2,3):
            directory=gwn/('runs/%s__GraphWaveNet_seed%d'%(variant,seed))
            record=read(directory/'TRAINING_RESULT.json'); logs=read(directory/'epoch_log.json')
            scaler=read(directory/'scaler.json')
            assert scaler=={'mean':mean,'std':std}
            assert len(logs)==30 and [r['epoch'] for r in logs]==list(range(1,31))
            best=min(logs,key=lambda r:r['validation_center_mae'])
            assert best['epoch']==record['best_epoch'] and best['validation_center_mae']==record['best_validation_center_mae']
            assert record['parameters']==parameter_count and record['training_test_targets_nan']
            assert sha(directory/'best_weights.pt')==record['weight_sha256']
            assert sha(directory/'mv_predictions.npz')==record['prediction_sha256']
            weights=torch.load(fs(directory/'best_weights.pt'),map_location='cpu',weights_only=True)
            assert {k:tuple(v.shape) for k,v in weights.items()}==expected_shape
            rows.append({'variant':variant,'seed':seed,'epochs_in_log':len(logs),
                'best_epoch':record['best_epoch'],'parameters':parameter_count,
                'weight_sha256':record['weight_sha256'],'forecast_sha256':record['prediction_sha256'],
                'scaler':scaler,'log_sha256':sha(directory/'epoch_log.json')})
    mv=RESOURCE/'components/netv4_mv'
    with np.load(fs(mv/'task/catalog.npz'),allow_pickle=False) as z:
        names=z['profile_id']; profiles=z['profile']; stored_neighbors=z['neighbor_indices']
    means=profiles.mean(axis=(1,2))
    features=np.concatenate([(profiles/means[:,None,None]).reshape(102,-1),np.log(means)[:,None]],axis=1)
    distance=np.mean((features[:,None,:]-features[None,:,:])**2,axis=2)
    neighbors=np.stack([np.array([j for j in np.argsort(distance[i],kind='stable') if j!=i])[:7] for i in range(102)])
    assert names.tolist()==sorted(names.tolist()) and np.array_equal(neighbors,stored_neighbors)
    with np.load(fs(mv/'task/evaluation.npz'),allow_pickle=False) as z:
        profile_index=z['profile_index']; assignments=z['profile_assignment']
        expected=np.stack([np.r_[i,i,neighbors[i]] for i in profile_index])
        assert np.array_equal(assignments,expected)
    runner=fs(gwn/'run_study.py').read_text(encoding='utf-8')
    for literal in ('lr=0.001, weight_decay=0.0001','range(30)','len(order), 256',
                    'clip_grad_norm_(model.parameters(), 5.)','torch.abs(output - targets[take]).mean()',
                    'clamp_min(0)','if metric < best:'):
        assert literal in runner, literal
    result={'status':'PASS','upstream_commit':read(gwn/'PINS.json')['upstream_commit'],
        'architecture':{'slots':9,'input_channels':1,'forecast_steps':12,
        'widths_residual_dilation_skip_end':[32,32,256,512],'blocks':4,'layers_per_block':2,
        'kernel_size':2,'dilation_schedule':[1,2]*4,'receptive_field':13,'left_padding_for_12_step_input':1,
        'dropout':0.3,'adaptive_only':True,'adaptive_embedding_rank':10,'graph_convolution_order':2,
        'physical_graph_input':False,'parameters':parameter_count},
        'optimization':{'optimizer':'Adam','learning_rate':0.001,'weight_decay':0.0001,
        'epochs':30,'batch_size':256,'gradient_clip_norm':5,'objective':'mean L1 across all nine slots and 12 horizons',
        'checkpoint_selection':'minimum nonnegative inverse-scaled center validation MAE, strict improvement',
        'normalization':'snapshot-specific global training-history mean/std + 1e-6',
        'prediction_clipping':'inverse transform, then clamp minimum zero','primary_storage':'FP32 channels_last'},
        'runs':rows,'model_adapter_sha256':sha(gwn/'model_adapter.py'),'training_runner_sha256':sha(gwn/'run_study.py'),
        'profile_selection':{'profiles':102,'weekly_profile_shape':list(profiles.shape[1:]),
        'curve_normalization':'divide by own weekly mean','volume_feature':'natural logarithm of own weekly mean',
        'feature_shape':list(features.shape),'distance':'mean squared coordinate difference across concatenated features',
        'feature_weights':'equal coordinate weights, without separate curve/volume reweighting',
        'sort':'stable ascending distance after lexicographic profile-ID sorting; exclude self; first seven',
        'assignment_matches_all_11424_scenarios':True,'center_copy_slot':1},
        'new_fits':0,'new_inference':0,'independent_confirmation':False}
    dump('GWN_AND_CONSTRUCTION_VERIFIED.json',result)
    emit({'stage':'configuration_verified','runs':len(rows),'receptive_field':13,'profile_shape':list(profiles.shape)})
    return result

def main():
    if '--configuration-only' in sys.argv:
        assert read(OUT/'REUSE_SCOPE_VERIFIED.json')['status']=='PASS'
        assert read(OUT/'UNROUNDED_LOSSES_VERIFIED.json')['status']=='PASS'
        dump('CONFIGURATION_RETRY.json',{'first_attempt':'stopped at missing unpacked training directory; no scientific data modified',
            'correction':'read the delivered digest-bound ZIP directly, without extracting into frozen resources',
            'corrected_script_sha256':sha(Path(__file__))})
    else:
        OUT.mkdir(exist_ok=False)
        pro=ROOT.parent/'123'
        dump('INPUT_BINDING.json',{'incoming':[{ 'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(pro.iterdir()) if p.is_file()],
            'resource_manifest_sha256':sha(RESOURCE/'MANIFEST.json'),'script_sha256':sha(Path(__file__)),
            'purpose':'existing evidence verification and response; no fresh confirmatory cohort'})
        verify_reuse()
        verify_losses()
    verify_configuration()
    dump('ROUND_VERIFICATION.json',{'status':'PASS','reuse_matching':'PASS','unrounded_loss_recomputation':'PASS',
        'actual_GWN_config_and_logs':'PASS','actual_profile_assignments':'PASS',
        'new_fits':0,'new_forecast_inference':0,'new_observed_records':0,
        'independent_confirmation_completed':False})

if __name__=='__main__':
    main()
