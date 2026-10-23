"""Use the pinned constructor, changing only the profile catalog and ID namespace."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
RESOURCE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_WWW_input_scope_20261011/resource')
MV=RESOURCE/'components/netv4_mv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def load_constructor():
    source=(MV/'construct.py').read_text(encoding='utf-8')
    before="""    contexts=json.loads((ROOT/'pins/contexts.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'pins/profiles.npz',allow_pickle=False) as z:original=z['profiles'].copy()
    groups={}
    for row,context in enumerate(contexts):
        name=context['profile_id']
        if name in groups:assert np.array_equal(original[groups[name]],original[row])
        else:groups[name]=row
    names=sorted(groups);profiles=np.stack([original[groups[name]] for name in names])"""
    assert source.count(before)==1
    after="""    if profile_catalog is None:
"""+'\n'.join('    '+line for line in before.splitlines())+"""
    else:
        with np.load(profile_catalog,allow_pickle=False) as z:
            names=z['profile_id'].tolist();profiles=z['profile'].copy()
        assert names==sorted(names) and len(names)>=8
        assert profiles.shape==(len(names),7,96)
        assert np.isfinite(profiles).all() and (profiles.mean(axis=(1,2))>0).all()
"""
    source=source.replace('def generate():','def generate(profile_catalog=None):').replace(before,after)
    # A distinct namespace avoids presenting new data under old sample identities.
    source=source.replace("('NETV4-MV|'+str(SEED)","(('NETV4-MV|' if profile_catalog is None else 'NETV4-confirmation|'+hashlib.sha256(Path(profile_catalog).read_bytes()).hexdigest()+'|')+str(SEED)")
    namespace={'__name__':'confirmation_constructor','__file__':str(MV/'construct.py')}
    exec(compile(source,str(MV/'construct.py'), 'exec'),namespace)
    namespace['ROOT']=MV
    return namespace,source

def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--regression',action='store_true');args=parser.parse_args()
    namespace,source=load_constructor()
    if args.regression:
        objects,description=namespace['generate']()
        compared=0
        for name,fields in objects.items():
            with np.load(MV/'task'/(name+'.npz'),allow_pickle=False) as old:
                assert set(old.files)==set(fields)
                for key,values in fields.items():
                    assert np.array_equal(old[key],values),(name,key)
                    compared+=values.size
        result={'status':'PASS','all_original_array_elements_exact':True,'elements':compared,
                'original_profiles':description['profiles'],'original_constructor_sha256':sha(MV/'construct.py'),
                'adapter_script_sha256':sha(Path(__file__)),'adapted_constructor_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'model_predictions_generated':0,'new_scoring':0}
        write(ROOT/'admission/CONSTRUCTOR_REGRESSION.json',result)
        print(json.dumps(result),flush=True)
        return
    catalog=ROOT/'profiles/catalog.npz'
    assert json.loads((ROOT/'admission/CONSTRUCTOR_REGRESSION.json').read_text())['status']=='PASS'
    objects,description=namespace['generate'](catalog)
    directory=ROOT/'task';directory.mkdir(exist_ok=False)
    for name,fields in objects.items():
        np.savez_compressed(directory/(name+'.npz'),**fields)
    description.update({'source':'Basel100006','new_observed_traffic_records':0,
        'new_source_hourly_records':14448,'source_catalog_sha256':sha(catalog),
        'new_generated_12_to_12_rows':description['input_rows'],'existing_checkpoint_training':False,
        'hour_to_quarter_uniform_allocation_assumed':True})
    write(directory/'DESCRIPTION.json',description)
    write(directory/'BINDING.json',{'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(directory.glob('*.npz'))],
        'original_constructor_sha256':sha(MV/'construct.py'),'adapter_script_sha256':sha(Path(__file__)),
        'adapted_constructor_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'catalog_sha256':sha(catalog),'source_intake_sha256':sha(ROOT/'admission/SOURCE_INTAKE_FREEZE.json')})
    print(json.dumps(description),flush=True)

if __name__=='__main__':main()
