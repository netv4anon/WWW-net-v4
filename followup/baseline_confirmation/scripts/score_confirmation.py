"""Score both target laws once, preserving all timing/profile heterogeneity."""
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

freeze=read(ROOT/'admission/CONFIRMATION_FREEZE.json')
assert sha(ROOT/'admission/CONFIRMATION_FREEZE.json')==(ROOT/'admission/CONFIRMATION_FREEZE_SHA256.txt').read_text().strip()
for item in freeze['campaign_files']:
    assert sha(ROOT/item['path'])==item['sha256'],item['path']
seal=read(ROOT/'predictions/PREDICTION_SEAL.json')
assert seal['all_six_saved_before_effect_scoring'] and len(seal['models'])==6
for item in seal['models']:assert sha(ROOT/item['file'])==item['sha256']
with np.load(ROOT/'task/evaluation.npz',allow_pickle=False) as z:task={k:z[k].copy() for k in z.files}
with np.load(ROOT/'task/catalog.npz',allow_pickle=False) as z:names=z['profile_id'].tolist()
n=len(task['timing']); old=task['target'].reshape(6,n,12,9)[:,:,:,0].astype(np.float64)
residual=task['residual_full_target'][:,:,0].astype(np.float64)
scopes={'all':np.ones(n,dtype=bool)}
for name in ('history_completed','cutoff_spanning','future_early','future_late'):scopes[name]=task['timing']==name
phases={'all_points':np.ones((n,12),dtype=bool),'pulse':task['future_pulse_mask'],
        'pre_pulse':np.arange(12,24)[None,:]<task['start'][:,None],
        'post_pulse':np.arange(12,24)[None,:]>=task['end'][:,None]}
responses=[];profiles=[];structural=[];context_rows=[]

def metrics(losses,bb):
    bf,cf,nf,ff=map(float,(losses[0],losses[4],losses[5],losses[3]))
    return {'L_BB':float(bb),'L_BF':bf,'L_CF':cf,'L_NF':nf,'L_FF':ff,
            'A':bf-ff,'T':bf-bb,'D':ff-bb,'A_N_given_C':cf-ff,'A_N_given_B':bf-nf,
            'conditional_error_increase_pct':100*(ff-cf)/cf if cf else None}

for model in seal['models']:
    with np.load(ROOT/model['file'],allow_pickle=False) as z:
        assert len(set(z['sample_id'].tolist()))==len(task['sample_id'])
        lookup={v:i for i,v in enumerate(z['sample_id'].tolist())}
        assert set(lookup)==set(task['sample_id'].tolist())
        pred=z['prediction'][[lookup[v] for v in task['sample_id'].tolist()]].astype(np.float64).reshape(6,n,12)
    use_future=np.isin(task['timing'],['future_early','future_late'])
    maxdiff=float(np.abs(pred[:,use_future]-pred[0,use_future]).max())
    structural.append({'model':model['id'],'identical_future_history_prediction_max_abs_diff':maxdiff,
                       'within_frozen_1e_3_tolerance':maxdiff<=.001})
    for law,target in [('legacy',old[3]),('residual',residual)]:
        errors=np.abs(pred-target[None,:,:]);bg=np.abs(pred[0]-old[0])
        for scope,selection in scopes.items():
            for phase,mask in phases.items():
                use=mask&selection[:,None]
                if not use.any():continue
                responses.append({'model':model['id'],'variant':model['variant'],'seed':model['seed'],
                    'target_law':law,'scope':scope,'phase':phase,'contexts':int(selection.sum()),'points':int(use.sum()),
                    **metrics(errors[:,use].mean(axis=1),float(bg[use].mean()))})
        per_context=errors.mean(axis=2);bg_context=bg.mean(axis=1)
        for timing in ('history_completed','cutoff_spanning','future_early','future_late'):
            for index,name in enumerate(names):
                selection=(task['profile_index']==index)&(task['timing']==timing)
                profiles.append({'model':model['id'],'variant':model['variant'],'seed':model['seed'],
                    'target_law':law,'timing':timing,'profile_id':name,'profile_index':index,'contexts':int(selection.sum()),
                    **metrics(per_context[:,selection].mean(axis=1),float(bg_context[selection].mean()))})
        for index in range(n):
            context_rows.append({'model':model['id'],'target_law':law,'context_id':index,
                'profile_index':int(task['profile_index'][index]),'background_id':int(task['background_id'][index]),
                'timing':str(task['timing'][index]),**metrics(per_context[:,index],float(bg_context[index]))})

primary=[r for r in responses if r['scope']=='cutoff_spanning' and r['phase']=='all_points']
assert len(primary)==12
epsilon=freeze['numerics']['effect_zero_tolerance_mae']
for row in primary:
    row['raw_sign_pattern']=row['A']>0 and row['A_N_given_C']<0
    row['numerically_resolved_pattern']=row['A']>epsilon and row['A_N_given_C']<-epsilon
    row['near_zero_A']=abs(row['A'])<=epsilon
    row['near_zero_A_N_given_C']=abs(row['A_N_given_C'])<=epsilon
# Add flag columns to all aggregate rows, with explicit blanks outside the primary cohort.
for row in responses:
    for key in ('raw_sign_pattern','numerically_resolved_pattern','near_zero_A','near_zero_A_N_given_C'):
        row.setdefault(key,None)
out=ROOT/'scores';out.mkdir(exist_ok=False)
write_csv(out/'responses.csv',responses);write_csv(out/'primary_12.csv',primary)
write_csv(out/'profile_all_timings.csv',profiles);write_csv(out/'context_losses_all.csv',context_rows)
write_csv(out/'structural_checks.csv',structural)
summary={'scored_utc':datetime.now(timezone.utc).isoformat(),'source':'Basel100006','profiles':len(names),
         'contexts':n,'frozen_models':6,'target_laws':2,'new_fits':0,'primary_rows':primary,
         'primary_raw_sign_passes':sum(r['raw_sign_pattern'] for r in primary),
         'primary_numerically_resolved_passes':sum(r['numerically_resolved_pattern'] for r in primary),
         'all_12_numerically_resolved':all(r['numerically_resolved_pattern'] for r in primary),
         'future_history_structural_checks':structural,
         'all_models_laws_timings_and_profiles_reported':True,
         'profile_reweighting':'each profile has 28 backgrounds per timing, so profile and scenario weighted means agree',
         'uncertainty':'finite cohort effects; no independent-incident confidence interval',
         'source_independence_scope':'newly acquired source absent from searchable retained project artifacts',
         'hourly_uniform_quarter_allocation':True,'physical_incident_validation':False,
         'freeze_sha256':sha(ROOT/'admission/CONFIRMATION_FREEZE.json'),
         'prediction_seal_sha256':sha(ROOT/'predictions/PREDICTION_SEAL.json')}
(out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True),flush=True)
