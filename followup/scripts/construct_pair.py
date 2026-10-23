"""Paired representation sensitivity: preserve neighbors, draws and metadata."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
RESOURCE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_WWW_input_scope_20261011/resource')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dump=lambda p,o:p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=(ROOT/'portable/construct.py').read_text(encoding='utf-8')
assert source.count('def generate(profile_catalog=None):')==1
source=source.replace('def generate(profile_catalog=None):','def generate(profile_catalog=None, fixed_neighbor_path=None, legacy_ids=False):')
marker='    n=len(names)*7*4*4'
source=source.replace(marker,"    if fixed_neighbor_path is not None:\n        with np.load(fixed_neighbor_path,allow_pickle=False) as fixed:\n            assert np.array_equal(fixed['profile_id'],np.array(names))\n            neighbors=fixed['neighbor_indices'].copy()\n        assert neighbors.shape==(len(names),7)\n"+marker)
source=source.replace("'NETV4-MV|' if profile_catalog is None else","'NETV4-MV|' if profile_catalog is None or legacy_ids else")
adapted=ROOT/'portable/paired_construct.py';adapted.write_text(source,encoding='utf-8')
spec=importlib.util.spec_from_file_location('paired_constructor',adapted)
constructor=importlib.util.module_from_spec(spec);spec.loader.exec_module(constructor)
old=ROOT/'paired/original';old.mkdir(exist_ok=False)
for name in ('inputs.npz','evaluation.npz','catalog.npz','DESCRIPTION.json'):
    (old/name).write_bytes((RESOURCE/'components/netv4_mv/task'/name).read_bytes())
with np.load(old/'catalog.npz',allow_pickle=False) as z:catalog={k:z[k].copy() for k in z.files}
p=catalog['profile'];h=np.repeat(p.reshape(len(p),7,24,4).mean(axis=3),4,axis=2)
assert np.allclose(p.reshape(len(p),7,24,4).sum(axis=3),h.reshape(len(p),7,24,4).sum(axis=3),atol=1e-10,rtol=0)
assert np.array_equal(h,np.repeat(h.reshape(len(h),7,24,4).mean(axis=3),4,axis=2))
new=ROOT/'paired/hour_uniform';new.mkdir(exist_ok=False)
np.savez_compressed(new/'profile_catalog.npz',profile_id=catalog['profile_id'],profile=h,neighbor_indices=catalog['neighbor_indices'])
# Same source code and explicit old namespace must regenerate the complete old task.
objects,description=constructor.generate(old/'catalog.npz',old/'catalog.npz',legacy_ids=True)
elements=0
for name,values in objects.items():
    with np.load(old/(name+'.npz'),allow_pickle=False) as z:
        assert set(values)==set(z.files)
        for key,value in values.items():assert np.array_equal(value,z[key]),(name,key);elements+=value.size
hour_objects,hour_description=constructor.generate(new/'profile_catalog.npz',old/'catalog.npz')
for name,values in hour_objects.items():np.savez_compressed(new/(name+'.npz'),**values)
for key in ('profile_index','background_id','timing','start','end','future_pulse_mask','calendar_coordinate','profile_assignment','severity','arm','context_id'):
    assert np.array_equal(objects['evaluation'][key],hour_objects['evaluation'][key]),key
assert np.array_equal(objects['catalog']['neighbor_indices'],hour_objects['catalog']['neighbor_indices'])
report={'status':'PASS','original_task_elements_exact':elements,'profiles':len(p),'profile_dtype':str(p.dtype),
    'hourly_total_max_abs_difference':float(np.max(np.abs(p.reshape(len(p),7,24,4).sum(axis=3)-h.reshape(len(p),7,24,4).sum(axis=3)))),
    'changed_profiles':int(np.any(p!=h,axis=(1,2)).sum()),'mean_abs_profile_change':float(np.abs(p-h).mean()),
    'neighbors_fixed_to_original':True,'same_random_number_streams':True,'all_scenario_metadata_exact':True,
    'original':description,'hour_uniform':hour_description,'only_new_synthetic_scenarios':hour_description['contexts'],
    'new_observed_records':0,'original_profiles_are_processed_not_raw_quarter_hour_validation':True,
    'adapted_constructor_sha256':sha(adapted)}
dump(ROOT/'admission/PAIR_CONSTRUCTION_CHECK.json',report)
print(json.dumps(report),flush=True)
