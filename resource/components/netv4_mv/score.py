"""Score all fixed models or a keyed reader forecast; preserve unfavorable views."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
TOL=1e-3
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_out(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def align(path,expected):
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'sample_id','prediction'}
        ids=z['sample_id'];p=z['prediction']
        assert ids.ndim==1 and ids.dtype.kind in 'US' and p.shape==(len(ids),12)
        assert p.dtype.kind in 'fiu' and np.isfinite(p).all()
        positions={k:i for i,k in enumerate(ids.tolist())}
        assert len(positions)==len(ids) and set(positions)==set(expected.tolist())
        return p[[positions[k] for k in expected]].astype(np.float64)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path);parser.add_argument('--out',type=Path,default=ROOT/'scores');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    task_binding=json.loads((ROOT/'task/BINDING.json').read_text(encoding='utf-8'))
    for r in task_binding['files']:assert sha(ROOT/'task'/r['path'])==r['sha256']
    with np.load(ROOT/'task/evaluation.npz',allow_pickle=False) as z:task={k:z[k] for k in z.files}
    n=len(task['timing']);old=task['target'].reshape(6,n,12,9)[:,:,:,0]
    replacement=task['residual_full_target'][:,:,0].astype(np.float64)
    models=(json.loads((ROOT/'predictions/MODELS.json').read_text(encoding='utf-8')) if args.predictions is None
            else [dict(id='submitted',file=str(args.predictions.resolve()),sha256=sha(args.predictions),center_only=False,kind='submitted',variant='user',seed=None)])
    scopes={'all':np.ones(n,dtype=bool)}
    scopes.update({name:task['timing']==name for name in ('history_completed','cutoff_spanning','future_early','future_late')})
    pulse=task['future_pulse_mask']
    pre=np.arange(12,24)[None,:]<task['start'][:,None]
    post=np.arange(12,24)[None,:]>=task['end'][:,None]
    phases={'all_points':np.ones((n,12),dtype=bool),'pulse':pulse,'pre_pulse':pre,'post_pulse':post}
    responses=[];accuracy=[];controls=[];profile_rows=[]
    for model in models:
        path=ROOT/model['file'];assert sha(path)==model['sha256']
        pred=align(path,task['sample_id']).reshape(6,n,12)
        center_diff=max(float(np.abs(pred[0]-pred[5]).max()),float(np.abs(pred[3]-pred[4]).max()))
        future_sel=np.isin(task['timing'],['future_early','future_late'])
        future_diff=float(np.abs(pred[:,future_sel]-pred[0,future_sel]).max())
        controls.append({'model':model['id'],'declared_center_only':model['center_only'],
                         'equal_center_prediction_max_difference':center_diff,
                         'equal_complete_future_starting_history_prediction_max_difference':future_diff,
                         'center_structural_check_passed':center_diff<=TOL if model['center_only'] else None,
                         'future_starting_zero_response_check_passed':future_diff<=TOL,
                         'submitted_determinism_certified':False})
        if args.predictions is None:
            assert future_diff<=TOL,(model['id'],future_diff)
            if model['center_only']:assert center_diff<=TOL,(model['id'],center_diff)
        for law,full in [('legacy',old[3]),('residual',replacement)]:
            loss_full=np.abs(pred-full[None,:,:])
            loss_bg=np.abs(pred[0]-old[0])
            for scope,selection in scopes.items():
                for phase,mask in phases.items():
                    use=mask&selection[:,None];points=int(use.sum())
                    row={'model':model['id'],'variant':model['variant'],'kind':model['kind'],'seed':model['seed'],
                         'target_law':law,'scope':scope,'phase':phase,'contexts':int(selection.sum()),'points':points}
                    if points:
                        losses=loss_full[:,use].mean(axis=1);bb=float(loss_bg[use].mean())
                        row.update({'L_BB':bb,'L_BF':float(losses[0]),'L_CF':float(losses[4]),'L_NF':float(losses[5]),'L_FF':float(losses[3]),
                                    'A':float(losses[0]-losses[3]),'T':float(losses[0]-bb),'D':float(losses[3]-bb),
                                    'neighbor_given_center':float(losses[4]-losses[3]),
                                    'neighbor_given_background_center':float(losses[0]-losses[5]),
                                    'center_given_background_neighbors':float(losses[0]-losses[4]),
                                    'center_given_event_neighbors':float(losses[5]-losses[3]),
                                    'interaction':float((losses[0]-losses[4])-(losses[5]-losses[3])),'status':'scored'})
                    else:
                        row.update({k:None for k in ('L_BB','L_BF','L_CF','L_NF','L_FF','A','T','D','neighbor_given_center',
                                  'neighbor_given_background_center','center_given_background_neighbors','center_given_event_neighbors','interaction')});row['status']='empty'
                    responses.append(row)
                for arm in range(4):
                    target=full if arm==3 else old[arm]
                    errors=pred[arm,selection]-target[selection]
                    accuracy.append({'model':model['id'],'target_law':law,'scope':scope,'arm':'BPRF'[arm],
                                     'contexts':int(selection.sum()),'mae':float(np.abs(errors).mean()),'rmse':float(np.sqrt((errors**2).mean()))})
            for group in range(102):
                use=(task['profile_index']==group)&(task['timing']=='cutoff_spanning')
                gain=(loss_full[4,use]-loss_full[3,use]).mean()
                profile_rows.append({'model':model['id'],'target_law':law,'profile_index':group,'contexts':int(use.sum()),'neighbor_given_center':float(gain)})
    csv_out(args.out/'responses.csv',responses);csv_out(args.out/'accuracy.csv',accuracy)
    csv_out(args.out/'structural_controls.csv',controls);csv_out(args.out/'profile_response.csv',profile_rows)
    primary=[r for r in responses if r['scope']=='cutoff_spanning' and r['phase']=='all_points' and r['kind']=='RidgeLocal9']
    gain_gate=any(abs(r['neighbor_given_center'])>.01 for r in primary) if args.predictions is None else None
    dump(args.out/'SUMMARY.json',{'resource':'NET-V4-MV','new_model_fits':0,'configurations':len(models),
          'contexts':n,'profiles':102,'input_rows':6*n,'primary_coupled_response_gate_passed':gain_gate,
          'primary_rows':primary,'structural_controls':controls,
          'prediction_sha256':{m['id']:m['sha256'] for m in models},'runner_sha256':sha(Path(__file__)),
          'protocol_sha256':sha(ROOT/'PROTOCOL_ZH.md'),'conditional_generated_cases_not_independent_incidents':True,
          'physical_or_causal_incident_effect_established':False,'old_raw_source_lineage_recovered':False})
    print(json.dumps({'primary_coupled_response_gate_passed':gain_gate,'primary_rows':primary,'models':len(models)}),flush=True)

if __name__=='__main__':main()
