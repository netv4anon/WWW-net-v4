"""Score identical saved/supplied NET-V4 forecasts against two frozen F targets.

This adapter exports an existing sensitivity experiment; it constructs no new
histories, targets or predictions. Only B/F comparisons use the alternative F
future. P and R are not reinterpreted as a new four-arm generator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate import load_task, align_prediction, numeric, write_csv

HERE=Path(__file__).resolve().parent
METRICS=('L_BB','L_BF','L_FF','L_FB','D','T','A')

def score_recovery(task, prediction, alternative, meta):
    b,legacy=task['target'][[0,3]]
    p_b,p_f=prediction[[0,3]]
    cohorts={'all':np.ones(len(task['contexts']),bool),
        'history_completed':task['timing']=='historical_only',
        'crossing_cut':task['timing']=='crossing_cut',
        'future_starting':task['timing']=='future_only',
        'test_origin':task['cohort']=='test_origin'}
    populations={'all_steps':np.ones(b.shape,bool),'pulse':task['pulse'],
                 'post_pulse':task['phase']=='post'}
    rows=[];pairs=[]
    def metrics(errors,use):
        values={k:float(v[use].mean()) if use.any() else None for k,v in errors.items()}
        values.update({'D':values['L_FF']-values['L_BB'],'T':values['L_BF']-values['L_BB'],
                       'A':values['L_BF']-values['L_FF']} if use.any() else {'D':None,'T':None,'A':None})
        return values
    for law,future in [('legacy',legacy),('residual_recovery',alternative)]:
        errors={'L_BB':np.abs(p_b-b),'L_BF':np.abs(p_b-future),
                'L_FF':np.abs(p_f-future),'L_FB':np.abs(p_f-b)}
        for group,selected in cohorts.items():
            for population,mask in populations.items():
                use=mask&selected[:,None]
                rows.append(dict(meta,law=law,cohort=group,population=population,
                    contexts=int(use.any(1).sum()),points=int(use.sum()),**metrics(errors,use)))
        for i,context in enumerate(task['contexts']):
            values={k:float(v[i].mean()) for k,v in errors.items()}
            values.update(D=values['L_FF']-values['L_BB'],T=values['L_BF']-values['L_BB'],A=values['L_BF']-values['L_FF'])
            pairs.append(dict(meta,law=law,context_id=int(context),**values))
    return rows,pairs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--predictions',type=Path)
    choice.add_argument('--replay-saved',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Output already exists; choose a new directory')
    task=load_task(HERE/'task')
    with np.load(HERE/'task/recovery_targets.npz',allow_pickle=False) as z:
        ids=z['context_id'];values=z['target']
        if ids.dtype.kind not in 'iu' or ids.shape!=task['contexts'].shape or len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(task['contexts'].tolist()):
            raise ValueError('Residual-target context IDs do not match the frozen task')
        lookup={int(k):i for i,k in enumerate(ids)}
        alternative=numeric(values,(len(ids),12),'residual target')[[lookup[int(i)] for i in task['contexts']]]
    models=json.loads((HERE/'MODELS.json').read_text()) if args.replay_saved else [dict(id='submitted',variant='user',run='submitted',model='submitted',seed=None)]
    rows=[];pairs=[];hashes={}
    for model in models:
        predpath=HERE/model['file'] if args.replay_saved else args.predictions
        digest=hashlib.sha256(predpath.read_bytes()).hexdigest()
        if args.replay_saved and digest!=model['sha256']:raise ValueError('Saved forecast hash mismatch: '+model['id'])
        prediction=align_prediction(task,predpath)
        a,c=score_recovery(task,prediction,alternative,{k:model[k] for k in ['variant','run','model','seed']})
        rows.extend(a);pairs.extend(c);hashes[model['id']]=digest
    args.out.mkdir(parents=True)
    write_csv(args.out/'metrics.csv',rows);write_csv(args.out/'pair_metrics.csv',pairs)
    summary={'scope':'Fixed predictions, two F-target constructions; original 12-to-12 NET-V4 B/F comparison',
        'prediction_sha256':hashes,'configurations_scored':len(models),'aggregate_rows':len(rows),'per_context_rows':len(pairs),
        'new_model_fits':0,'new_prediction_executions':0,'new_target_construction':False,'causal_effect_claim':False,
        'primary_rows':[r for r in rows if r['cohort']=='all' and r['population']=='all_steps']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['configurations_scored','aggregate_rows','per_context_rows']}))

if __name__=='__main__':main()
