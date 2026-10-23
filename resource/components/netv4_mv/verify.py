"""Independent direct-loss, coverage and semantic-view verification."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def records(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
def align(path,expected):
    with np.load(path,allow_pickle=False) as z:
        location={k:i for i,k in enumerate(z['sample_id'].tolist())}
        return z['prediction'][[location[k] for k in expected]].astype(np.float64)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'VERIFICATION.json');args=parser.parse_args()
    with np.load(ROOT/'task/evaluation.npz',allow_pickle=False) as z:task={k:z[k] for k in z.files}
    n=len(task['timing']);old=task['target'].reshape(6,n,12,9)[:,:,:,0].astype(np.float64)
    with np.load(ROOT/'task/inputs.npz',allow_pickle=False) as z:
        inputs=z['history'];location={k:i for i,k in enumerate(z['sample_id'].tolist())}
        x=inputs[[location[k] for k in task['sample_id']]].reshape(6,n,12,9)
    assert np.array_equal(x[:,:,:,0],x[:,:,:,1])
    assert np.array_equal(x[4,:,:,:2],x[3,:,:,:2]) and np.array_equal(x[4,:,:,2:],x[0,:,:,2:])
    assert np.array_equal(x[5,:,:,:2],x[0,:,:,:2]) and np.array_equal(x[5,:,:,2:],x[3,:,:,2:])
    future=np.isin(task['timing'],['future_early','future_late'])
    assert all(np.array_equal(x[a,future],x[0,future]) for a in range(6))
    assert np.array_equal(old[4],old[3]) and np.array_equal(old[5],old[3])
    coords=np.arange(12,24)[None,:]
    pulse=(coords>=task['start'][:,None])&(coords<task['end'][:,None])
    assert np.array_equal(pulse,task['future_pulse_mask']) and (pulse.sum(axis=0)>0).all()
    assert np.array_equal(task['profile_assignment'][:,0],task['profile_assignment'][:,1])
    assert all(len(set(r.tolist()))==8 for r in task['profile_assignment'])
    model_rows=json.loads((ROOT/'predictions/MODELS.json').read_text(encoding='utf-8'))
    maps={r['id']:r for r in model_rows};maximum=0.;checked=0
    phases={'all_points':np.ones((n,12),dtype=bool),'pulse':pulse,
            'pre_pulse':coords<task['start'][:,None],'post_pulse':coords>=task['end'][:,None]}
    scope_masks={s:np.ones(n,dtype=bool) if s=='all' else task['timing']==s
                 for s in ('all','history_completed','cutoff_spanning','future_early','future_late')}
    predictions={name:align(ROOT/r['file'],task['sample_id']).reshape(6,n,12) for name,r in maps.items()}
    for row in records(ROOT/'scores/responses.csv'):
        selection=phases[row['phase']]&scope_masks[row['scope']][:,None]
        assert int(selection.sum())==int(row['points'])
        if not selection.any():assert row['status']=='empty';continue
        p=predictions[row['model']]
        target=old[3] if row['target_law']=='legacy' else task['residual_full_target'][:,:,0].astype(np.float64)
        losses=[float(np.abs(p[a]-target)[selection].mean()) for a in (0,4,5,3)]
        bb=float(np.abs(p[0]-old[0])[selection].mean())
        expected={'L_BB':bb,'L_BF':losses[0],'L_CF':losses[1],'L_NF':losses[2],'L_FF':losses[3],
                  'A':losses[0]-losses[3],'T':losses[0]-bb,'D':losses[3]-bb,
                  'neighbor_given_center':losses[1]-losses[3],
                  'neighbor_given_background_center':losses[0]-losses[2],
                  'center_given_background_neighbors':losses[0]-losses[1],
                  'center_given_event_neighbors':losses[2]-losses[3],
                  'interaction':losses[0]-losses[1]-losses[2]+losses[3]}
        for k,v in expected.items():maximum=max(maximum,abs(v-float(row[k])));checked+=1
        assert abs(expected['D']-(expected['T']-expected['A']))<1e-10
    assert maximum<1e-10,maximum
    result={'input_view_semantics_checked':True,'duplicate_center_layout_checked':True,
            'all_twelve_horizons_have_pulse_support':True,'equal_history_future_cases_checked':True,
            'direct_loss_values_checked':checked,'direct_loss_max_absolute_difference':maximum,
            'new_model_fits':0,'model_configurations':len(maps),
            'does_not_validate_physical_adjacency_or_old_raw_source':True}
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
