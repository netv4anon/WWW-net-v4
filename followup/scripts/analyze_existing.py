"""Post-confirmation descriptive decomposition; no new forecasts or data."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PREV=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_profile_confirmation_20261015')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def save(name,rows):
    if not rows:return
    with (ROOT/'analysis'/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

with np.load(PREV/'task/evaluation.npz',allow_pickle=False) as z:task={k:z[k].copy() for k in z.files}
with np.load(PREV/'task/catalog.npz',allow_pickle=False) as z:catalog={k:z[k].copy() for k in z.files}
n=len(task['timing']);profile=task['profile_index'];assignment=task['profile_assignment']
manifest=read(PREV/'profiles/ADMISSION_RESULT.json')['eligible_manifest']
site_lookup={r['profile_id']:r['sitecode'] for r in manifest}
site_of_profile=np.array([site_lookup[p] for p in catalog['profile_id']]);sites=site_of_profile[profile]
ys={'legacy':task['target'].reshape(6,n,12,9)[3,:,:,0].astype(np.float64),
    'residual':task['residual_full_target'][:,:,0].astype(np.float64)}
out=ROOT/'analysis';(out/'saved_points').mkdir(exist_ok=False)
np.savez_compressed(out/'aligned_targets.npz',**ys,profile_index=profile,profile_id=catalog['profile_id'],
    site_of_profile=site_of_profile,profile_assignment=assignment,timing=task['timing'],calendar_coordinate=task['calendar_coordinate'])
primary=[];profiles=[];directions=[];horizons=[];omissions=[];target_changes=[];checks=[]
for model in read(PREV/'predictions/PREDICTION_SEAL.json')['models']:
    with np.load(PREV/model['file'],allow_pickle=False) as z:
        lookup={k:i for i,k in enumerate(z['sample_id'].tolist())}
        pred=z['prediction'][[lookup[k] for k in task['sample_id'].tolist()]].astype(np.float64).reshape(6,n,12)
    pc=pred[4];pf=pred[3];delta=pf-pc
    np.savez_compressed(out/'saved_points'/(model['id']+'.npz'),center_history_prediction=pc,full_history_prediction=pf)
    dlaw={}
    for law,y in ys.items():
        e=pc-y;d=np.abs(e)-np.abs(e+delta);dlaw[law]=d
        assert np.max(np.abs(d)-np.abs(delta))<1e-10
        for timing in sorted(set(task['timing'])):
            use=task['timing']==timing;dv=d[use];ev=e[use];change=delta[use]
            response=float(dv.mean());loss=float(np.abs(ev).mean());absdelta=float(np.abs(change).mean())
            checks.append(abs(response-float(np.abs(ev).mean()-np.abs(ev+change).mean())))
            base={'model':model['id'],'variant':model['variant'],'seed':model['seed'],'law':law,'timing':timing}
            primary.append({**base,'contexts':int(use.sum()),'loss_C':loss,'loss_F':float(np.abs(ev+change).mean()),
                'A_N_given_C':response,'conditional_loss_change_pct':-100*response/loss,
                'mean_abs_delta':absdelta,'p50_abs_delta':float(np.median(np.abs(change))),
                'p90_abs_delta':float(np.quantile(np.abs(change),.9)),
                'mean_delta':float(change.mean()),'delta_as_pct_C_loss':100*absdelta/loss,
                'underprediction_fraction':float((ev<0).mean()),'overprediction_fraction':float((ev>0).mean()),
                'mean_abs_point_response':float(np.abs(dv).mean()),
                'response_cancellation_ratio':abs(response)/float(np.abs(dv).mean()) if np.abs(dv).mean() else 0.,
                'helpful_point_fraction':float((dv>0).mean()),'harmful_point_fraction':float((dv<0).mean())})
            # These disjoint cases add exactly to the unrounded conditional response.
            masks={'unchanged':change==0,'from_exact_target':(change!=0)&(ev==0),
                'away_from_target':ev*change>0,
                'towards_without_crossing':(ev*change<0)&(ev*(ev+change)>=0),
                'crossing_target':(ev*change<0)&(ev*(ev+change)<0)}
            assert np.all(sum(mask.astype(int) for mask in masks.values())==1)
            for label,mask in masks.items():
                directions.append({**base,'group':label,'points':int(mask.sum()),'fraction':float(mask.mean()),
                    'response_contribution':float(dv[mask].sum()/dv.size),
                    'within_group_response':float(dv[mask].mean()) if mask.any() else '',
                    'mean_abs_delta_in_group':float(np.abs(change[mask]).mean()) if mask.any() else ''})
            for h in range(12):horizons.append({**base,'horizon':h+1,'A_N_given_C':float(dv[:,h].mean()),'mean_abs_delta':float(np.abs(change[:,h]).mean())})
            for pi,pid in enumerate(catalog['profile_id']):
                mask=use&(profile==pi)
                profiles.append({**base,'profile_id':pid,'sitecode':site_of_profile[pi],
                    'A_N_given_C':float(d[mask].mean()),'cohort_response_contribution':float(d[mask].sum()/d[use].size),
                    'mean_abs_delta':float(np.abs(delta[mask]).mean()),'contexts':int(mask.sum())})
            if timing=='cutoff_spanning':
                for kind,labels in [('center_profile',catalog['profile_id']),('center_site',np.unique(sites)),('any_input_site',np.unique(sites))]:
                    for label in labels:
                        if kind=='center_profile':remove=profile==int(np.flatnonzero(catalog['profile_id']==label)[0])
                        elif kind=='center_site':remove=sites==label
                        else:remove=(site_of_profile[assignment]==label).any(axis=1)
                        keep=use&~remove;remaining=int(keep.sum())
                        value=float(d[keep].mean()) if remaining else ''
                        omissions.append({**base,'omission_unit':kind,'omitted_id':label,'remaining_contexts':remaining,
                            'removed_contexts':int((use&remove).sum()),'original_A_N_given_C':response,
                            'remaining_A_N_given_C':value,'remaining_resolved_sign':('positive' if value>1e-3 else 'negative' if value<-1e-3 else 'ambiguous') if remaining else 'unsupported',
                            'status':'available' if remaining else 'no_remaining_contexts'})
    use=task['timing']=='cutoff_spanning';diff=dlaw['residual'][use]-dlaw['legacy'][use]
    target_changes.append({'model':model['id'],'identical_prediction_delta_under_both_laws':True,
        'legacy_A_N_given_C':float(dlaw['legacy'][use].mean()),'residual_A_N_given_C':float(dlaw['residual'][use].mean()),
        'target_only_response_shift':float(diff.mean()),'mean_abs_point_response_shift':float(np.abs(diff).mean()),
        'fraction_point_response_changed_by_target':float((diff!=0).mean()),
        'fraction_initial_error_sign_changed_by_target':float((np.sign(pc[use]-ys['legacy'][use])!=np.sign(pc[use]-ys['residual'][use])).mean())})
save('decomposition_all_timings.csv',primary);save('profile_contributions.csv',profiles);save('direction_contributions.csv',directions)
save('horizon_decomposition.csv',horizons);save('record_omission_sensitivity.csv',omissions);save('target_only_changes.csv',target_changes)
expected=read(PREV/'scores/SUMMARY.json')['primary_rows']
for row in primary:
    if row['timing']=='cutoff_spanning':
        ex=next(r for r in expected if r['model']==row['model'] and r['target_law']==row['law'])
        assert abs(ex['A_N_given_C']-row['A_N_given_C'])<1e-11
summary={'status':'COMPLETE','analysis_label':'post-confirmation descriptive decomposition; not a new confirmation',
    'new_inference':0,'new_training':0,'source_profiles':73,'source_sites':len(set(sites)),
    'response_identity_max_error':max(checks),'primary_rows':[r for r in primary if r['timing']=='cutoff_spanning'],
    'target_only_changes':target_changes,'confidence_intervals_or_independence_claims':False,
    'omissions_change_records_only_not_retraining_or_neighbor_reassignment':True}
(out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'COMPLETE','new_inference':0,'primary_decompositions':12,'all_timing_decompositions':len(primary),'omission_rows':len(omissions),'identity_max_error':max(checks)}),flush=True)
