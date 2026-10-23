"""Score user point forecasts against both fixed target laws (offline evaluator)."""
import argparse,json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--predictions',required=True,type=Path)
parser.add_argument('--out',required=True,type=Path)
args=parser.parse_args()
task=Path(__file__).resolve().parent/'probe'
x=np.load(task/'INPUTS.npz',allow_pickle=False)
with np.load(args.predictions,allow_pickle=False) as supplied:
    if set(supplied.files)!={'seed','prediction'}:raise ValueError('Required arrays: seed and prediction')
    ids=supplied['seed'];forecast=supplied['prediction']
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer):raise ValueError('seed must be a 1D integer array')
    if len(ids)!=len(x['seed']) or len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(x['seed'].tolist()):raise ValueError('Supply each fixed seed exactly once')
    if forecast.shape!=(len(ids),24) or not np.issubdtype(forecast.dtype,np.number) or np.iscomplexobj(forecast) or not np.isfinite(forecast).all():raise ValueError('prediction must be finite real numeric [32,24]')
    positions={int(seed):i for i,seed in enumerate(ids)}
    forecast=forecast[[positions[int(seed)] for seed in x['seed']]].astype(np.float64)
y=np.load(task/'TARGETS.npz',allow_pickle=False)
mask=(np.arange(24)[None,:]>=x['event_start'][:,None])&(np.arange(24)[None,:]<(x['event_start']+x['event_duration'])[:,None])
rows=[]
for law in ['reference','pointwise']:
    error=np.abs(forecast-y[law]);full=error.mean(axis=1);event=(error*mask).sum(axis=1)/mask.sum(axis=1)
    rows.append({'target':law,'full_mae':float(full.mean()),'event_mae_case_equal':float(event.mean()),'event_mae_pooled':float(error[mask].mean()),'per_instance':[{'seed':int(seed),'full_mae':float(a),'event_mae':float(b)} for seed,a,b in zip(x['seed'],full,event)]})
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps({'scope':'32 constrained-source cases; two hypothetical target laws','scores':rows},indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in z.items() if k!='per_instance'} for z in rows],indent=2))
