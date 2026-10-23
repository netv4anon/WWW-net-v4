import hashlib,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;data=p/'probe'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
seal=json.loads((data/'PREDICTION_SEAL.json').read_text())
assert sha(data/'PREDICTIONS.npz')==seal['prediction_file_sha256']
x=np.load(data/'INPUTS.npz',allow_pickle=False);y=np.load(data/'TARGETS.npz',allow_pickle=False);pred=np.load(data/'PREDICTIONS.npz',allow_pickle=False)
n=len(x['history']);mask=np.zeros((n,24),dtype=bool)
for i,(s,d) in enumerate(zip(x['event_start'],x['event_duration'])):mask[i,s:s+d]=True
scores=[];by={}
for target in ['reference','pointwise']:
    for name in pred.files:
        errors=np.abs(pred[name]-y[target]);per_roi=np.array([errors[i,mask[i]].mean() for i in range(n)])
        row={'target':target,'predictor':name,'instances':n,'full_mae':float(errors.mean()),'event_mae_case_equal':float(per_roi.mean()),'event_mae_pooled':float(errors[mask].mean()),'per_instance_full_mae':errors.mean(axis=1).tolist(),'per_instance_event_mae':per_roi.tolist()}
        scores.append(row);by[(target,name)]=row
contrasts=[]
for base in ['last_value','last_day','mean_daily_profile']:
    item={'base':base}
    for measure in ['full_mae','event_mae_case_equal','event_mae_pooled']:
        a=by[('reference',base+'__anchor')][measure]-by[('reference',base+'__pointwise')][measure]
        s=by[('pointwise',base+'__anchor')][measure]-by[('pointwise',base+'__pointwise')][measure]
        item[measure]={'reference_delta_anchor_minus_pointwise':a,'alternative_delta_anchor_minus_pointwise':s,'sign_changes':bool(a*s<0 and min(abs(a),abs(s))>1e-12)}
    contrasts.append(item)
variation=[float(np.ptp(y['background'][i,mask[i]])) for i in range(n)]
over=y['reference']>y['background']
changed=y['reference']!=y['pointwise']
struct={'instances':n,'unique_history_windows':len(set(a.tobytes() for a in x['history'])),'event_positions':int(mask.sum()),'multi_hour_instances':int((x['event_duration']>1).sum()),'varying_background_event_instances':sum(v>0 for v in variation),'reference_event_varying_instances':sum(float(np.ptp(y['reference'][i,mask[i]]))>0 for i in range(n)),'pointwise_event_varying_instances':sum(float(np.ptp(y['pointwise'][i,mask[i]]))>0 for i in range(n)),'target_difference_event_positions':int(changed[mask].sum()),'max_target_difference':float(np.abs(y['reference']-y['pointwise']).max()),'event_target_mae_difference_case_equal':float(np.mean([np.abs(y['reference'][i,mask[i]]-y['pointwise'][i,mask[i]]).mean() for i in range(n)])),'reference_exceeds_background_event_positions':int(over[mask].sum()),'reference_exceeds_background_instances':int(np.any(over&mask,axis=1).sum()),'unchanged_outside_event':bool(np.array_equal(y['reference'][~mask],y['pointwise'][~mask]))}
result={'scope':'Constrained one-source reference-operation probe; not CiK published-benchmark reproduction','structure':struct,'scores':scores,'within_base_contrasts':contrasts,'prespecified_utility_gate_passed':any(z['event_mae_case_equal']['sign_changes'] for z in contrasts),'declared_predictor_cells':9,'distinct_predictor_configurations':seal['distinct_prediction_arrays'],'trained_models':0,'physical_truth_or_published_ranking_claim':False}
(data/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(data/'CODE_AND_INPUT_SEAL.json').write_text(json.dumps({'files':{n:sha(p/n) for n in ['TARGET_PROBE_PROTOCOL.md','SCORING_CLARIFICATION.md','prepare_probe.py','construct_probe.py','predict_probe.py','score_probe.py','probe/INPUTS.npz','probe/TARGETS.npz','probe/PREDICTIONS.npz']}},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'structure':struct,'scores':[{k:v for k,v in z.items() if not k.startswith('per_instance')} for z in scores],'contrasts':contrasts,'gate_passed':result['prespecified_utility_gate_passed']},indent=2))
