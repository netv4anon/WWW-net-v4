# -*- coding: utf-8 -*-
"""Score supplied 12-step predictions on the frozen observed case/control pairs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
FIELDS=['case_pre','control_pre','case_post','control_post','delta_pre','delta_post','change']
def score(path,data):
    z=np.load(path,allow_pickle=False)
    p=z['predictions']; ids=z['ids'].tolist(); names=z['models'].tolist()
    truth=np.load(data/'targets.npz',allow_pickle=False)
    expected=truth['ids'].tolist()
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected): raise ValueError('Missing, unknown or duplicated forecast IDs')
    if len(names)!=len(set(names)): raise ValueError('Duplicated model names')
    if p.shape!=(len(names),len(ids),12) or not np.isfinite(p).all(): raise ValueError('Invalid prediction shape or nonfinite predictions')
    lookup={value:i for i,value in enumerate(ids)}
    p=p[:,[lookup[value] for value in expected]]
    y=truth['y'].astype('float64')
    with (data/'episodes.csv').open(encoding='utf-8',newline='') as f:
        episodes=list(csv.DictReader(f))
    pairs=json.loads((data/'pairs.json').read_text(encoding='utf-8'))
    loss=np.abs(p-y[None,:,:]).mean(axis=-1)
    values=np.empty((len(names),len(pairs),7),dtype='float64')
    for j,pair in enumerate(pairs):
        case=episodes[pair['case_episode_row']]
        controls=[episodes[i] for i in pair['control_episode_rows']]
        cp=loss[:,int(case['pre_prediction_row'])]
        ca=loss[:,int(case['post_prediction_row'])]
        bp=loss[:,[int(e['pre_prediction_row']) for e in controls]].mean(axis=1)
        ba=loss[:,[int(e['post_prediction_row']) for e in controls]].mean(axis=1)
        values[:,j,:]=np.stack([cp,bp,ca,ba,cp-bp,ca-ba,(ca-cp)-(ba-bp)],axis=-1)
    seeds=['STID_seed1','STID_seed2','STID_seed3']
    if all(name in names for name in seeds):
        values=np.concatenate([values,values[[names.index(n) for n in seeds]].mean(axis=0,keepdims=True)])
        names=names+['STID_seed_mean']
    aggregate=[dict(model=name,**dict(zip(FIELDS,values[i].mean(axis=0).tolist()))) for i,name in enumerate(names)]
    weeks=np.array([(p['day']-1)//7+1 for p in pairs])
    weekly=[dict(model=name,week=w,**dict(zip(FIELDS,values[i,weeks==w].mean(axis=0).tolist())))
            for i,name in enumerate(names) for w in range(1,5)]
    stations=np.array([p['station_id'] for p in pairs])
    omissions=[dict(model=name,omitted_station=int(s),retained_cases=int((stations!=s).sum()),
                   **dict(zip(FIELDS,values[i,stations!=s].mean(axis=0).tolist())))
               for i,name in enumerate(names) for s in np.unique(stations)]
    return {'cases':len(pairs),'control_memberships':sum(len(p['control_episode_rows']) for p in pairs),
            'stations':len(np.unique(stations)),'aggregate':aggregate,'weekly':weekly,'leave_one_station':omissions,
            'estimand':'case-weighted descriptive pre/post case-minus-control MAE contrast',
            'causal_effect_established':False,'independent_incident_replications_claimed':False}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite score output')
    result=score(a.predictions,a.data)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'cases':result['cases'],'models_scored':len(result['aggregate']),'output':str(a.output)}))
if __name__=='__main__': main()

