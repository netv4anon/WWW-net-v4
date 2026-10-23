"""Fixed paired and crossed scores, including all timings and profile failures."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
RESOURCE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_WWW_input_scope_20261011/resource')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=read(ROOT/'paired/predictions/PREDICTION_SEAL.json');assert len(seal['forecasts'])==12
for r in seal['forecasts']:assert sha(ROOT/r['file'])==r['sha256']
tasks={}
for condition in ('original','hour_uniform'):
    with np.load(ROOT/'paired'/condition/'evaluation.npz',allow_pickle=False) as z:tasks[condition]={k:z[k].copy() for k in z.files}
n=len(tasks['original']['timing']);base=tasks['original'];timings=sorted(set(base['timing']))
with np.load(ROOT/'paired/original/catalog.npz',allow_pickle=False) as z:names=z['profile_id'].copy()
out=ROOT/'paired/scores';out.mkdir(exist_ok=False)
rows=[];profile_rows=[];control=[];baseline=[];cross_checks=[]
band=lambda x:'positive' if x>1e-3 else 'negative' if x<-1e-3 else 'ambiguous'
for variant in ('packaged','working'):
    for seed in (1,2,3):
        model_id=variant+'__GraphWaveNet_seed'+str(seed);ps={}
        for condition in ('original','hour_uniform'):
            row=next(r for r in seal['forecasts'] if r['model']==model_id and r['condition']==condition)
            with np.load(ROOT/row['file'],allow_pickle=False) as z:
                lookup={k:i for i,k in enumerate(z['sample_id'].tolist())}
                ps[condition]=z['prediction'][[lookup[k] for k in tasks[condition]['sample_id'].tolist()]].astype(np.float64).reshape(6,n,12)
            for timing in timings:
                if timing.startswith('future_'):
                    use=base['timing']==timing
                    control.append({'model':model_id,'condition':condition,'timing':timing,
                        'max_prediction_difference_across_views':float(np.abs(ps[condition][:,use]-ps[condition][0,use][None,:,:]).max())})
        with np.load(RESOURCE/'components/netv4_gwn/runs'/model_id/'mv_predictions.npz',allow_pickle=False) as z:
            lookup={k:i for i,k in enumerate(z['sample_id'].tolist())}
            old=z['prediction'][[lookup[k] for k in base['sample_id'].tolist()]].astype(np.float64).reshape(6,n,12)
        baseline.append({'model':model_id,'max_abs_saved_forecast_difference':float(np.abs(old-ps['original']).max())})
        for law in ('legacy','residual'):
            ys={condition:(task['target'].reshape(6,n,12,9)[3,:,:,0] if law=='legacy' else task['residual_full_target'][:,:,0]).astype(np.float64) for condition,task in tasks.items()}
            # R_ab indexes prediction representation a, target representation b.
            ds={}
            for a,ac in enumerate(('original','hour_uniform')):
                for b,bc in enumerate(('original','hour_uniform')):
                    ds[(a,b)]=np.abs(ps[ac][4]-ys[bc])-np.abs(ps[ac][3]-ys[bc])
            for timing in timings:
                use=base['timing']==timing
                values={key:float(d[use].mean()) for key,d in ds.items()}
                a0=float((np.abs(ps['original'][0,use]-ys['original'][use])-np.abs(ps['original'][3,use]-ys['original'][use])).mean())
                a1=float((np.abs(ps['hour_uniform'][0,use]-ys['hour_uniform'][use])-np.abs(ps['hour_uniform'][3,use]-ys['hour_uniform'][use])).mean())
                r00,r10,r01,r11=[values[key] for key in ((0,0),(1,0),(0,1),(1,1))]
                pred_path=r10-r00;target_path=r11-r10;reverse_target=r01-r00;reverse_pred=r11-r01
                cross_checks.append(abs((r11-r00)-(pred_path+target_path)))
                key={'model':model_id,'variant':variant,'seed':seed,'law':law,'timing':timing}
                row={**key,'contexts':int(use.sum()),'A_original':a0,'A_hour_uniform':a1,
                    'conditional_original':r00,'conditional_hour_uniform':r11,'paired_conditional_change':r11-r00,
                    'original_numerical_sign':band(r00),'hour_uniform_numerical_sign':band(r11),
                    'raw_conditional_sign_changed':bool(np.sign(r00)!=np.sign(r11)),
                    'R00':r00,'R10':r10,'R01':r01,'R11':r11,
                    'prediction_component_hold_original_target':pred_path,'target_component_hold_hour_uniform_predictions':target_path,
                    'target_component_hold_original_predictions':reverse_target,'prediction_component_hold_hour_uniform_target':reverse_pred,
                    'interaction':r11-r10-r01+r00,
                    'original_mean_abs_prediction_increment':float(np.abs(ps['original'][3,use]-ps['original'][4,use]).mean()),
                    'hour_uniform_mean_abs_prediction_increment':float(np.abs(ps['hour_uniform'][3,use]-ps['hour_uniform'][4,use]).mean())}
                rows.append(row)
                for index,pid in enumerate(names):
                    mask=use&(base['profile_index']==index)
                    profile_rows.append({**key,'profile_id':pid,'contexts':int(mask.sum()),
                        **{'R%d%d'%k:float(d[mask].mean()) for k,d in ds.items()},
                        'paired_conditional_change':float((ds[(1,1)][mask]-ds[(0,0)][mask]).mean())})
def save(name,rows):
    with (out/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
save('all_timings.csv',rows);save('profile_all_timings.csv',profile_rows);save('future_history_controls.csv',control);save('baseline_saved_prediction_comparison.csv',baseline)
primary=[r for r in rows if r['timing']=='cutoff_spanning'];save('primary_12.csv',primary)
assert max(r['max_prediction_difference_across_views'] for r in control)<=1e-3
summary={'status':'COMPLETE','identity':'post-confirmation paired temporal-representation sensitivity',
    'new_fits':0,'condition_forecast_runs':12,'frozen_models':6,'profiles':102,
    'primary_rows':primary,'primary_raw_sign_changes':sum(r['raw_conditional_sign_changed'] for r in primary),
    'primary_uniform_negative':sum(r['conditional_hour_uniform']<0 for r in primary),
    'max_crossed_path_identity_error':max(cross_checks),
    'future_control_max_difference':max(r['max_prediction_difference_across_views'] for r in control),
    'baseline_saved_forecast_max_difference':max(r['max_abs_saved_forecast_difference'] for r in baseline),
    'not_a_causal_identification_of_Basel_reversals':True,'independent_confidence_intervals':False}
(out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'COMPLETE','primary_raw_sign_changes':summary['primary_raw_sign_changes'],'primary_uniform_negative':summary['primary_uniform_negative'],'baseline_prediction_max_difference':summary['baseline_saved_forecast_max_difference'],'crossed_path_identity_error':max(cross_checks)}),flush=True)
