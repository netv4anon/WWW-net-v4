"""Portable saved-forecast replay. --arrays also rebuilds the H condition.

Optional --infer uses a fresh output directory, never replacing frozen results.
"""
import argparse,csv,hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--arrays',action='store_true');parser.add_argument('--infer',choices=['cpu','cuda']);parser.add_argument('--out',type=Path);args=parser.parse_args()
manifest=read(ROOT/'RESOURCE_MANIFEST.json')
for r in manifest['files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
freeze=read(ROOT/'admission/FOLLOWUP_DESIGN_FREEZE.json')
assert sha(ROOT/'admission/FOLLOWUP_DESIGN_FREEZE.json')==(ROOT/'admission/FOLLOWUP_DESIGN_SHA256.txt').read_text().strip()
for r in freeze['files']:
    p=ROOT/r['path']
    if p.is_file():assert sha(p)==r['sha256'],r['path']
    else:assert r['path'] in read(ROOT/'DISTRIBUTION_OMISSIONS.json')
with np.load(ROOT/'paired/original/catalog.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
with np.load(ROOT/'paired/hour_uniform/catalog.npz',allow_pickle=False) as z:uniform={k:z[k].copy() for k in z.files}
expected_h=np.repeat(old['profile'].reshape(len(old['profile']),7,24,4).mean(axis=3),4,axis=2)
assert np.array_equal(expected_h,uniform['profile'])
assert np.array_equal(old['neighbor_indices'],uniform['neighbor_indices'])
assert np.array_equal(old['profile_id'],uniform['profile_id'])
assert np.array_equal(old['profile'].reshape(len(old['profile']),7,24,4).sum(axis=3),uniform['profile'].reshape(len(uniform['profile']),7,24,4).sum(axis=3))
elements=0
if args.arrays:
    p=ROOT/'portable/paired_construct.py';spec=importlib.util.spec_from_file_location('replay_H_constructor',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    objects,description=module.generate(ROOT/'paired/hour_uniform/profile_catalog.npz',ROOT/'paired/original/catalog.npz')
    for name,values in objects.items():
        with np.load(ROOT/'paired/hour_uniform'/(name+'.npz'),allow_pickle=False) as z:
            assert set(values)==set(z.files)
            for key,value in values.items():assert np.array_equal(value,z[key]),(name,key);elements+=value.size

# Independently aggregate point responses, using the aligned saved C/F forecasts.
with np.load(ROOT/'analysis/aligned_targets.npz',allow_pickle=False) as z:points={k:z[k].copy() for k in z.files}
use=points['timing']=='cutoff_spanning';decomposition=read(ROOT/'analysis/SUMMARY.json');maxdiff=0.;decomposition_checks=0
provided=list(csv.DictReader((ROOT/'analysis/PRIMARY_RECOMPUTED.csv').open(encoding='utf-8-sig')))
for file in sorted((ROOT/'analysis/saved_points').glob('*.npz')):
    with np.load(file,allow_pickle=False) as z:pc=z['center_history_prediction'];pf=z['full_history_prediction']
    for law in ('legacy','residual'):
        value=float((np.abs(pc[use]-points[law][use])-np.abs(pf[use]-points[law][use])).mean())
        expected=next(r for r in decomposition['primary_rows'] if r['model']==file.stem and r['law']==law)
        delta=abs(value-expected['A_N_given_C']);assert delta<1e-12;maxdiff=max(maxdiff,delta)
        supplied=next(r for r in provided if r['model']==file.stem and r['law']==law)
        assert abs(value-float(supplied['A_N_given_C']))<1e-11
        decomposition_checks+=1

tasks={}
for condition in ('original','hour_uniform'):
    with np.load(ROOT/'paired'/condition/'evaluation.npz',allow_pickle=False) as z:tasks[condition]={k:z[k].copy() for k in z.files}
for key in ('profile_assignment','timing','severity','calendar_coordinate','profile_index','background_id','start','end'):
    assert np.array_equal(tasks['original'][key],tasks['hour_uniform'][key]),key
seal=read(ROOT/'paired/predictions/PREDICTION_SEAL.json');summary=read(ROOT/'paired/scores/SUMMARY.json')
n=len(tasks['original']['timing']);use=tasks['original']['timing']=='cutoff_spanning';crossed_checks=0;signchanges=0
for model in sorted(set(r['model'] for r in seal['forecasts'])):
    predictions={}
    for condition in ('original','hour_uniform'):
        row=next(r for r in seal['forecasts'] if r['model']==model and r['condition']==condition)
        assert sha(ROOT/row['file'])==row['sha256']
        assert sha(ROOT/'models/runs'/model/'best_weights.pt')==row['weight_sha256']
        with np.load(ROOT/row['file'],allow_pickle=False) as z:
            lookup={k:i for i,k in enumerate(z['sample_id'].tolist())}
            predictions[condition]=z['prediction'][[lookup[k] for k in tasks[condition]['sample_id'].tolist()]].astype(np.float64).reshape(6,n,12)
    for law in ('legacy','residual'):
        expected=next(r for r in summary['primary_rows'] if r['model']==model and r['law']==law)
        values={}
        for b,bc in enumerate(('original','hour_uniform')):
            task=tasks[bc];y=(task['target'].reshape(6,n,12,9)[3,:,:,0] if law=='legacy' else task['residual_full_target'][:,:,0]).astype(np.float64)
            for a,ac in enumerate(('original','hour_uniform')):
                p=predictions[ac];value=float(np.abs(p[4,use]-y[use]).mean()-np.abs(p[3,use]-y[use]).mean());values[(a,b)]=value
                diff=abs(value-expected['R%d%d'%(a,b)]);assert diff<1e-11;maxdiff=max(maxdiff,diff);crossed_checks+=1
        assert values[(0,0)]<0 and values[(1,1)]<0
        signchanges+=int(np.sign(values[(0,0)])!=np.sign(values[(1,1)]))
result={'status':'PASS','manifest_files_checked':len(manifest['files']),'frozen_design_files_checked':len(freeze['files']),
    'Basel_existing_primary_responses_recomputed':decomposition_checks,'pair_crossed_primary_responses_recomputed':crossed_checks,
    'maximum_scoring_difference':maxdiff,'H_task_elements_regenerated_exactly':elements,
    'float32_hourly_sum_check_and_fixed_neighbors_checked':True,'float64_hourly_sum_max_error':float(np.abs(old['profile'].astype(np.float64).reshape(102,7,24,4).sum(axis=3)-uniform['profile'].astype(np.float64).reshape(102,7,24,4).sum(axis=3)).max()),'old_profile_pair_raw_sign_changes':signchanges,
    'Basel_confirmation_result_unchanged':'7 retained signs, 5 reversals; all-twelve conjunction failed','new_inference':0}
if args.infer:
    assert args.out is not None,'Use --out NEW_DIRECTORY'
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    sys.path.insert(0,str(ROOT/'models'))
    import reader_adapter
    for variant in ('packaged','working'):
        for seed in (1,2,3):
            for condition in ('original','hour_uniform'):
                with np.load(ROOT/'paired'/condition/'inputs.npz',allow_pickle=False) as z:ids=z['sample_id'].copy();history=z['history'].copy()
                pred=reader_adapter.predict(history,variant,seed,args.infer)
                np.savez_compressed(args.out/('%s__GraphWaveNet_seed%d__%s.npz'%(variant,seed,condition)),sample_id=ids,prediction=pred)
            reader_adapter._CACHE.clear()
    result['new_inference']=12
    result['optional_replay_note']='New backend outputs remain separate; no frozen forecasts are overwritten or selected.'
print(json.dumps(result),flush=True)
