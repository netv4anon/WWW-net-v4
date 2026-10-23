"""Offline replay of admission, arrays and saved-forecast primary losses.

Default mode generates no neural forecasts. --infer cpu/cuda optionally runs
the included six actual weights, saving outputs in a separate directory.
"""
import argparse,contextlib,hashlib,importlib.util,io,json,os,shutil,sys,tempfile
from pathlib import Path
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

parser=argparse.ArgumentParser();parser.add_argument('--infer',choices=['cpu','cuda']);parser.add_argument('--out',type=Path)
args=parser.parse_args()
manifest=read(ROOT/'RESOURCE_MANIFEST.json')
for r in manifest['files']:
    assert sha(ROOT/r['path'])==r['sha256'],r['path']
freeze=read(ROOT/'admission/CONFIRMATION_FREEZE.json')
assert sha(ROOT/'admission/CONFIRMATION_FREEZE.json')==(ROOT/'admission/CONFIRMATION_FREEZE_SHA256.txt').read_text().strip()
for r in freeze['campaign_files']:
    p=ROOT/r['path']
    if p.is_file():assert sha(p)==r['sha256'],r['path']
    else:assert r['path'] in read(ROOT/'DISTRIBUTION_OMISSIONS.json')
for r in freeze['pinned_external_code']:
    relative=r['path'].replace('\\','/').split('/components/netv4_gwn/',1)[1]
    assert sha(ROOT/'models'/relative)==r['sha256'],relative
with tempfile.TemporaryDirectory(prefix='netv4-admission-') as temp:
    scratch=Path(temp)
    shutil.copytree(ROOT/'raw',scratch/'raw');(scratch/'admission').mkdir()
    for name in ('SOURCE_INTAKE_FREEZE.json','SOURCE_INTAKE_SHA256.txt'):
        shutil.copy2(ROOT/'admission'/name,scratch/'admission'/name)
    source=(ROOT/'scripts/build_profiles.py').read_text(encoding='utf-8')
    source=source.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path(REPLAY_ROOT)')
    old_resource="RESOURCE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_WWW_input_scope_20261011/resource')"
    assert source.count(old_resource)==1
    source=source.replace(old_resource,'RESOURCE=Path(REPLAY_REFERENCE)')
    namespace={'__file__':str(ROOT/'scripts/build_profiles.py'),'__name__':'admission_replay',
               'REPLAY_ROOT':str(scratch),'REPLAY_REFERENCE':str(ROOT/'legacy_reference')}
    with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,'admission_replay','exec'),namespace)
    actual=read(scratch/'profiles/ADMISSION_RESULT.json');saved=read(ROOT/'profiles/ADMISSION_RESULT.json')
    for field in ('status','eligible_profiles','source_channels','excluded_channels','failure_reason_counts','eligible_manifest'):
        assert actual[field]==saved[field],field
    with np.load(scratch/'profiles/catalog.npz',allow_pickle=False) as a,np.load(ROOT/'profiles/catalog.npz',allow_pickle=False) as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)

constructor=load(ROOT/'portable/construct.py','portable_confirmation_constructor')
objects,description=constructor.generate(ROOT/'profiles/catalog.npz')
elements=0
for name,fields in objects.items():
    with np.load(ROOT/'task'/(name+'.npz'),allow_pickle=False) as actual:
        assert set(actual.files)==set(fields)
        for key,value in fields.items():
            assert np.array_equal(actual[key],value),(name,key)
            elements+=value.size
task=objects['evaluation'];n=len(task['timing']);use=task['timing']=='cutoff_spanning'
target=task['target'].reshape(6,n,12,9)[3,:,:,0].astype(np.float64)
residual=task['residual_full_target'][:,:,0].astype(np.float64)
summary=read(ROOT/'scores/SUMMARY.json');seal=read(ROOT/'predictions/PREDICTION_SEAL.json')
comparisons=0;maxdiff=0.;pattern_passes=0;pattern_total=0
for item in seal['models']:
    path=ROOT/Path(item['file'].replace('\\','/'))
    assert sha(path)==item['sha256']
    with np.load(path,allow_pickle=False) as z:
        lookup={v:i for i,v in enumerate(z['sample_id'].tolist())}
        p=z['prediction'][[lookup[v] for v in task['sample_id'].tolist()]].astype(np.float64).reshape(6,n,12)
    for law,y in [('legacy',target),('residual',residual)]:
        loss=np.abs(p[:,use]-y[None,use]).mean(axis=(1,2))
        expected=next(r for r in summary['primary_rows'] if r['model']==item['id'] and r['target_law']==law)
        for value,key in [(loss[0]-loss[3],'A'),(loss[4]-loss[3],'A_N_given_C')]:
            diff=abs(float(value)-expected[key]);maxdiff=max(maxdiff,diff);assert diff<1e-12;comparisons+=1
        pattern_total+=1
        pattern_passes+=int(loss[0]-loss[3]>1e-3 and loss[4]-loss[3]<-1e-3)
    directory=ROOT/'models/runs'/item['id']
    assert sha(directory/'best_weights.pt')==item['weight_sha256']
result={'status':'PASS','source_admission_replayed':True,'task_elements_exact':elements,
        'primary_effects_recalculated':comparisons,'max_primary_effect_difference':maxdiff,
        'actual_weights_present_and_hash_matched':len(seal['models']),
        'primary_pattern_passes':pattern_passes,'primary_pattern_comparisons':pattern_total,
        'new_model_inference':0,'all_12_confirmation_passed':pattern_passes==pattern_total}

if args.infer:
    assert args.out is not None,'Supply a new --out directory'
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    sys.path.insert(0,str(ROOT/'models'))
    adapter=load(ROOT/'models/reader_adapter.py','portable_GWN_reader')
    with np.load(ROOT/'task/inputs.npz',allow_pickle=False) as z:
        ids=z['sample_id'].copy();history=z['history'].copy()
    for item in seal['models']:
        pred=adapter.predict(history,item['variant'],item['seed'],args.infer)
        np.savez_compressed(args.out/(item['id']+'.npz'),sample_id=ids,prediction=pred)
        adapter._CACHE.clear()
    result['new_model_inference']=6
    result['replay_note']='This optional replay uses the requested backend. It does not change the frozen original outputs or select a preferred effect sign.'
print(json.dumps(result),flush=True)
