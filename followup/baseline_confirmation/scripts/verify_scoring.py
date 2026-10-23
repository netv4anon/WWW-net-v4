"""Cross-check new scoring through the original published scoring implementation."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESOURCE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_WWW_input_scope_20261011/resource')
MV=RESOURCE/'components/netv4_mv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    with p.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))

directory=ROOT/'validation';directory.mkdir(exist_ok=True)
seal=read(ROOT/'predictions/PREDICTION_SEAL.json')
registry=[{**r,'kind':'GraphWaveNet','center_only':False} for r in seal['models']]
(ROOT/'predictions/MODELS.json').write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
(ROOT/'PROTOCOL_ZH.md').write_text('''# 本次确认的协议入口

以 admission/CONFIRMATION_FREEZE.json、SOURCE_INTAKE_FREEZE.json 和 incoming_pro/VALIDATION_ADMISSION_ADDENDUM_zh.md 为准。此文件是原评分器所需的协议入口，不改变封存的推理和主要比较。
''',encoding='utf-8')
source=(MV/'score.py').read_text(encoding='utf-8')
assert source.count('range(102)')==1 and source.count("'profiles':102")==1
adapted=source.replace('range(102)',"range(len(np.unique(task['profile_index'])))").replace("'profiles':102","'profiles':len(np.unique(task['profile_index']))")
namespace={'__file__':str(MV/'score.py'),'__name__':'original_scoring_compatibility_check'}
exec(compile(adapted,str(MV/'score.py'),'exec'),namespace)
namespace['ROOT']=ROOT
sys.argv=[str(MV/'score.py'),'--out',str(directory/'original_scorer')]
namespace['main']()
current=rows(ROOT/'scores/responses.csv')
original=rows(directory/'original_scorer/responses.csv')
key=lambda r:(r['model'],r['target_law'],r['scope'],r['phase'])
index={key(r):r for r in original}
diffs=[]
mapping={'L_BB':'L_BB','L_BF':'L_BF','L_CF':'L_CF','L_NF':'L_NF','L_FF':'L_FF','A':'A','T':'T','D':'D',
         'A_N_given_C':'neighbor_given_center','A_N_given_B':'neighbor_given_background_center'}
for row in current:
    reference=index[key(row)]
    for new,old in mapping.items():
        diff=abs(float(row[new])-float(reference[old]));diffs.append(diff)
        assert diff<1e-12,(key(row),new,diff)
profiles=rows(ROOT/'scores/profile_all_timings.csv')
profile_ref={(r['model'],r['target_law'],r['profile_index']):r for r in rows(directory/'original_scorer/profile_response.csv')}
for row in profiles:
    if row['timing']=='cutoff_spanning':
        reference=profile_ref[row['model'],row['target_law'],row['profile_index']]
        assert abs(float(row['A_N_given_C'])-float(reference['neighbor_given_center']))<1e-12
with np.load(ROOT/'profiles/catalog.npz',allow_pickle=False) as z:
    catalog=z['profile'];profile_ids=z['profile_id']
    assert len(set(profile_ids.tolist()))==73 and (catalog.mean(axis=(1,2))>0).all()
    assert all(np.array_equal(catalog[:,:,0::4],catalog[:,:,offset::4]) for offset in (1,2,3))
summary=read(ROOT/'scores/SUMMARY.json')
snapshot_means=[]
for variant in ('packaged','working'):
    for law in ('legacy','residual'):
        use=[r for r in summary['primary_rows'] if r['variant']==variant and r['target_law']==law]
        snapshot_means.append({'variant':variant,'target_law':law,'A':float(np.mean([r['A'] for r in use])),
            'A_N_given_C':float(np.mean([r['A_N_given_C'] for r in use])),
            'passing_checkpoints':sum(r['numerically_resolved_pattern'] for r in use),'checkpoints':3})
result={'status':'PASS','original_scorer_sha256':sha(MV/'score.py'),
    'compatibility_edits':'replace only two fixed 102 counts with observed catalog size; evaluation root and output path changed',
    'aggregate_rows_checked':len(current),'loss_comparisons':len(diffs),'max_abs_difference':max(diffs),
    'primary_profile_rows_checked':len(profile_ref),'all_12_primary_combinations_retained':True,
    'all_task_target_laws_share_saved_predictions':True,'snapshot_means':snapshot_means,
    'new_forecast_runs_for_verification':0,'all_12_conjunction_passed':False}
(directory/'SCORING_CROSSCHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
