# -*- coding: utf-8 -*-
"""Replay the inspected release using source-bound byte offsets; never unpickle data."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Original Contra_Costa incident_all.npy, version 1')
    parser.add_argument('--output',type=Path,required=True,help='A new output directory')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    spec=json.loads((root/'SOURCE.json').read_text(encoding='utf-8'))
    if args.source.stat().st_size!=spec['bytes'] or sha(args.source)!=spec['sha256']:
        raise ValueError('Source is not the pinned audited release; no results were generated')
    if args.output.exists(): raise ValueError('Output directory already exists; choose a new directory')
    args.output.mkdir(parents=True)
    with gzip.open(root/'ARRAY_OFFSETS.json.gz','rb') as f: blob=f.read()
    assert hashlib.sha256(blob).hexdigest()==spec['offset_manifest_sha256']
    arrays={(r['row'],r['role']):r for r in json.loads(blob.decode('utf-8-sig'))}
    expected=json.loads((root/'EXPECTED_RESULTS.json').read_text(encoding='utf-8'))
    mean,std=expected['mean'],expected['std']
    train=collections.defaultdict(list)
    targets={}
    all_steps=set()
    phases=[]
    rows=[]
    candidates=[]
    n=5585
    def traffic(b):
        return np.frombuffer(b,dtype='<f4').reshape(12,496,3)[:,:,0].astype('float64')*std+mean
    with args.source.open('rb') as f:
        def get(i,role):
            r=arrays[i,role]
            f.seek(r['offset'])
            b=f.read(r['bytes'])
            assert hashlib.sha256(b).hexdigest()==r['sha256']
            return b
        for i in range(3909):
            x=get(i,'x_data')
            y=get(i,'y_data')
            train[x].append(i)
            targets[i]=y
            all_steps.update(y[t*5952:(t+1)*5952] for t in range(12))
        assert all(len({targets[i] for i in group})==1 for group in train.values())
        # All histories are read before query-target scoring.
        query_histories=[get(i,'x_data') for i in range(3909,n)]
        for i in range(n):
            x=get(i,'x_data') if i<3909 else query_histories[i-3909]
            calendar=np.frombuffer(x,dtype='<f4').reshape(12,496,3)[:,:,1:]
            assert np.all(calendar==calendar[:,:1,:])
            slots=calendar[:,0,:].astype('float64')*np.array([288,7])
            assert np.max(np.abs(slots-np.rint(slots)))<.0001
            phases.append(int(round(slots[-1,0]))+288*int(round(slots[-1,1])))
        for i,x in enumerate(query_histories,start=3909):
            ids=train.get(x,[])
            last=np.tile(traffic(x)[-1],(12,1))
            memory=np.mean(np.stack([traffic(targets[j]) for j in ids]),axis=0) if ids else last.copy()
            # No query target used to choose matches or generate prediction.
            ybytes=get(i,'y_data')
            y=traffic(ybytes)
            row={'query_row':i,'split':'val' if i<4746 else 'test',
                 'history_hit':bool(ids),'target_vector_shared':any(ybytes[t*5952:(t+1)*5952] in all_steps for t in range(12))}
            for name,p in [('LastValue',last),('ExactHistory',memory)]:
                diff=p-y
                row[name+'_mae']=float(np.abs(diff).mean())
                row[name+'_mse']=float(np.square(diff).mean())
            candidates.append({'query_row':i,'training_rows':ids})
            rows.append(row)
    groups=[]
    for split in ['val','test']:
        filters={'all':lambda r:True,'history_hit':lambda r:r['history_hit'],'history_miss':lambda r:not r['history_hit'],
                 'any_train_target_vector_shared':lambda r:r['target_vector_shared'],
                 'no_train_target_vector_shared':lambda r:not r['target_vector_shared']}
        for label,check in filters.items():
            selected=[r for r in rows if r['split']==split and check(r)]
            result={'split':split,'group':label,'rows':len(selected)}
            for name in ['LastValue','ExactHistory']:
                result[name+'_mae']=float(np.mean([r[name+'_mae'] for r in selected]))
                result[name+'_rmse']=float(np.sqrt(np.mean([r[name+'_mse'] for r in selected])))
            result['mae_reduction']=result['LastValue_mae']-result['ExactHistory_mae']
            result['mae_reduction_percent']=100*result['mae_reduction']/result['LastValue_mae']
            original=next(g for g in expected['groups'] if g['split']==split and g['group']==label)
            assert result['rows']==original['rows']
            for key,v in result.items():
                if key not in ['rows','split','group']: assert abs(v-original[key])<1e-10,(key,v,original[key])
            groups.append(result)
    minimum_bins=sum((b-a)%2016 for a,b in zip(phases,phases[1:]))
    prefix_bins=sum((b-a)%2016 for a,b in zip(phases[:108],phases[1:108]))
    assert minimum_bins==5733143 and prefix_bins/288>372
    report={'status':'passed','source_sha256':spec['sha256'],'sample_rows':n,
            'calendar_minimum_bins':minimum_bins,'calendar_minimum_days':minimum_bins/288,
            'prefix_108_minimum_days':prefix_bins/288,'groups':groups,
            'replay_prediction_configurations':2,'new_optimization_fits':0,
            'scope':'one pinned released region; not a general chronology repair or original-paper reproduction'}
    (args.output/'REPLAY_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (args.output/'QUERY_MATCHES.json').write_text(json.dumps(candidates,indent=2)+'\n',encoding='utf-8')
    with (args.output/'ROW_METRICS.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'status':'passed','rows':len(rows),'calendar_minimum_days':minimum_bins/288,'prediction_configurations_replayed':2}))
if __name__=='__main__': main()

