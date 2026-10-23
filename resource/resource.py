"""Navigate and run the source-bound NET-V4 evaluation components locally."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def catalog():
    return json.loads((ROOT/'resources.json').read_text(encoding='utf-8'))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def contained(name):
    path=(ROOT/name).resolve()
    if ROOT not in path.parents:raise ValueError('Manifest path outside resource directory')
    return path

def verify(component=None):
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    prefix=catalog()['components'][component].get('path') if component else None
    rows=[r for r in manifest['files'] if prefix is None or r['path']==prefix or r['path'].startswith(prefix+'/')]
    if not rows:raise ValueError('No bound files for requested component')
    for row in rows:
        f=contained(row['path'])
        if not f.is_file() or f.stat().st_size!=row['bytes'] or digest(f)!=row['sha256']:
            raise ValueError('Resource binding mismatch: '+row['path'])
    return len(rows)

def execute(component,out,command):
    with (out/'commands.log').open('a',encoding='utf-8') as log:
        log.write(json.dumps(command,ensure_ascii=False)+'\n');log.flush()
        result=subprocess.run(command,cwd=str(component),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Component failed; see '+str(out/'commands.log'))

def validate_user(component,path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        if component=='observed':
            if set(z.files)!={'ids','models','predictions'}:raise ValueError('Observed schema requires ids, models, predictions')
            ids=z['ids'];names=z['models'];p=z['predictions']
            if ids.ndim!=1 or ids.dtype.kind not in 'US' or names.ndim!=1 or names.dtype.kind not in 'US':raise ValueError('IDs and model names must be string vectors')
            if not len(names) or len(set(names.tolist()))!=len(names) or any(not str(n).strip() for n in names):raise ValueError('Model names must be nonempty and unique')
            if 'STID_seed_mean' in names and all(s in names for s in ['STID_seed1','STID_seed2','STID_seed3']):raise ValueError('STID_seed_mean is reserved for the derived seed summary when all three seeds are supplied')
            with np.load(ROOT/'components/observed/data/inputs.npz',allow_pickle=False) as inp:expected=inp['ids'].tolist()
            if len(ids)!=len(set(ids.tolist())) or set(ids.tolist())!=set(expected):raise ValueError('Missing, unknown or duplicate observed forecast IDs')
            if p.shape!=(len(names),len(ids),12) or p.dtype.kind not in 'fiu' or not np.isfinite(p).all():raise ValueError('Observed forecasts must be finite real numeric [models,2062,12]')
        elif component=='netv4_mv':
            if set(z.files)!={'sample_id','prediction'}:raise ValueError('MV schema requires sample_id and prediction')
            ids=z['sample_id'];p=z['prediction']
            with np.load(ROOT/'components/netv4_mv/task/inputs.npz',allow_pickle=False) as inp:expected=inp['sample_id'].tolist()
            if ids.ndim!=1 or ids.dtype.kind not in 'US' or len(ids)!=len(set(ids.tolist())) or set(ids.tolist())!=set(expected):raise ValueError('Missing, unknown or duplicate MV IDs')
            if p.shape!=(len(ids),12) or p.dtype.kind not in 'fiu' or not np.isfinite(p).all():raise ValueError('MV forecasts must be finite real numeric [68544,12]')
        elif component in ['netv4','netv4_checks']:
            if set(z.files)!={'sample_id','prediction'}:raise ValueError('NET-V4 schema requires sample_id and prediction')
            ids=z['sample_id'];p=z['prediction']
            with np.load(ROOT/'components/netv4/task/inputs.npz',allow_pickle=False) as inp:expected=inp['sample_id'].tolist()
            if ids.ndim!=1 or ids.dtype.kind not in 'US' or len(ids)!=len(set(ids.tolist())) or set(ids.tolist())!=set(expected):raise ValueError('Missing, unknown or duplicate NET-V4 IDs')
            if p.shape!=(len(ids),12) or p.dtype.kind not in 'fiu' or not np.isfinite(p).all():raise ValueError('NET-V4 forecasts must be finite real numeric [1024,12]')
        else:
            if set(z.files)!={'seed','prediction'}:raise ValueError('CiK schema requires seed and prediction')
            ids=z['seed'];p=z['prediction']
            if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)!=32 or set(ids.tolist())!=set(range(32)):raise ValueError('Supply each frozen seed 0..31 exactly once')
            if p.shape!=(32,24) or p.dtype.kind not in 'fiu' or not np.isfinite(p).all():raise ValueError('CiK forecasts must be finite real numeric [32,24]')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    show=sub.add_parser('show');show.add_argument('component',choices=list(catalog()['components']))
    check=sub.add_parser('verify');check.add_argument('--component',choices=list(catalog()['components']))
    replay=sub.add_parser('replay');replay.add_argument('component',choices=['netv4','observed','cik_423339','cik_10130110','release_audit','station_lookup','station_exclusion','netv4_checks','netv4_mv']);replay.add_argument('--out',type=Path,required=True);replay.add_argument('--source',type=Path)
    score=sub.add_parser('score');score.add_argument('component',choices=['netv4','observed','cik_423339','cik_10130110','netv4_checks','netv4_mv']);score.add_argument('--predictions',type=Path,required=True);score.add_argument('--out',type=Path,required=True)
    score.add_argument('--prediction-mode',choices=['deterministic','single_realization_stochastic'],default='deterministic')
    args=parser.parse_args();c=catalog()
    if args.action=='list':
        for key,item in c['components'].items():print(key+': '+item['question'])
        return
    if args.action=='show':print(json.dumps(c['components'][args.component],indent=2,ensure_ascii=False));return
    if args.action=='verify':print(json.dumps({'verified_files':verify(args.component),'coverage':'Stored bytes only; not scientific validity or source regeneration'}));return
    info=c['components'][args.component];component=ROOT/info['path'];out=args.out.absolute().resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Choose an output directory outside the immutable resource directory')
    if out.exists():raise ValueError('Output already exists; choose a new directory')
    verify(args.component)
    if args.component in ['release_audit','station_lookup','station_exclusion']:
        if args.source is None or not args.source.is_file():raise ValueError(args.component+' requires --source for the pinned incident_all.npy; see show '+args.component)
        args.source=args.source.resolve()
        binding=info['source_binding']
        if args.source.stat().st_size!=binding['bytes'] or digest(args.source)!=binding['sha256']:raise ValueError('External source does not match the pinned release')
    if args.action=='score':args.predictions=args.predictions.resolve();validate_user(args.component,args.predictions)
    out.mkdir(parents=True)
    py=[sys.executable,'-B','-X','utf8'];calls=[]
    if args.component=='netv4_mv':
        if args.action=='replay':calls=[py+['replay.py','--out',str(out/'mv')]]
        else:calls=[py+['score.py','--predictions',str(args.predictions),'--out',str(out/'scores')]]
    elif args.component=='netv4_checks':
        verify('netv4')
        if args.action=='replay':
            calls=[py+['replay_core_checks.py','--archive',str(ROOT/'archives/netv4_review_candidate_v0_3.zip'),'--component',str(ROOT/'components/netv4'),'--out',str(out/'checks')]]
        else:
            calls=[py+['score_target_envelope.py','--component',str(ROOT/'components/netv4'),'--predictions',str(args.predictions),'--out',str(out/'target_segment')]]
    elif args.component=='netv4':
        if args.action=='replay':
            calls=[py+['replay_saved.py','--out',str(out/'original')],py+['score_recovery.py','--replay-saved','--out',str(out/'recovery')]]
        else:
            calls=[py+['evaluate.py','--task','task','--predictions',str(args.predictions),'--prediction-mode',args.prediction_mode,'--out',str(out/'original')],py+['score_recovery.py','--predictions',str(args.predictions),'--out',str(out/'recovery')]]
    elif args.component=='observed':
        pred=args.predictions if args.action=='score' else component/'saved_predictions.npz'
        calls=[py+['evaluate.py','--predictions',str(pred),'--data','data','--output',str(out/'scores.json')]]
    elif args.component.startswith('cik_'):
        if args.action=='score':calls=[py+['score_user_forecast.py','--predictions',str(args.predictions),'--out',str(out/'scores.json')]]
        else:
            work=out/'work';shutil.copytree(component,work);component=work
            calls=[py+[name] for name in ['construct_probe.py','predict_probe.py','score_probe.py','verify_probe.py']]
    elif args.component=='station_lookup':
        calls=[py+['replay_station_lookup.py','--source',str(args.source),'--out',str(out/'lookup')]]
    elif args.component=='station_exclusion':
        calls=[py+['audit_exclusion.py','--source',str(args.source),'--out',str(out/'audit')],
               py+['verify_replay.py','--out',str(out/'audit')]]
    else:calls=[py+['audit_release.py','--source',str(args.source),'--output',str(out/'audit')]]
    for command in calls:execute(component,out,command)
    (out/'RUN.json').write_text(json.dumps({'component':args.component,'action':args.action,'new_fit':False,'new_scientific_prediction_configuration':False,'scope':info['scope'],'prediction_sha256':digest(args.predictions) if args.action=='score' else None},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'completed','component':args.component,'output':str(out)}))

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError,RuntimeError) as exc:
        raise SystemExit(str(exc))
