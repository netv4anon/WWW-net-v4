"""Execute pinned source methods with a disclosed one-series loader."""
import ast,copy,hashlib,json,logging,types,warnings
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
binding=json.loads((p/'PROBE_SOURCE_BINDING.json').read_text())
assert binding['eligible']
frame=pd.read_csv(p/'probe/selected_series.csv',float_precision='round_trip')
series=pd.Series(frame.occupancy.to_numpy(),index=pd.to_datetime(frame.timestamp))
assert len(series)==1653 and np.all(np.diff(series.index.asi8)==3600*10**9)
scope={'np':np,'pd':pd,'logging':logging,'Union':Union}
functions=[]
def extracted(filename,name,classname=None):
    text=(p/'sources'/filename).read_text(); tree=ast.parse(text)
    body=tree.body if classname is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
    fn=next(n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    functions.append({'source':filename,'function':name,'class':classname,'start_line':fn.lineno,'end_line':fn.end_lineno,'ast_sha256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()})
    exec(compile(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[]),filename,'exec'),scope)
    return scope[name]
helper=extracted('cik_utils.py','get_random_window_univar')
extracted('cik_utils.py','datetime_to_str')
extracted('cik_data_pems.py','get_traffic_prediction_length')
extracted('cik_data_pems.py','get_traffic_history_factor')
method=extracted('cik_pred_change_tasks.py','random_instance','DecreaseInTrafficInPredictionTask')
captured={}
def local_loader(target,random):
    assert target=='Occupancy (%)'
    return series.copy(deep=True)
def window_capture(*args,**kwargs):
    w=helper(*args,**kwargs)
    captured['window']=w.copy(deep=True)
    return w
scope['load_traffic_series']=local_loader;scope['get_random_window_univar']=window_capture
histories=[];backgrounds=[];anchors=[];scaled=[];starts=[];durations=[];ratios=[];seeds=[];texts=[];failures=[];warning_records=[]
for seed in range(32):
    task=types.SimpleNamespace(random=np.random.RandomState(seed))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');method(task)
        warning_records.extend({'seed':seed,'category':str(w.category.__name__),'message':str(w.message)} for w in caught)
        before=captured['window'];h=before.iloc[:-24].to_numpy(copy=True);v=before.iloc[-24:].to_numpy(copy=True)
        s=task.region_of_interest.start;d=task.region_of_interest.stop-s
        # Read r from the supplied text, not the target or background future.
        import re
        r=float(re.search(r'there is ([\d.]+)% of the usual traffic',task.scenario).group(1))/100
        y=task.future_time.iloc[:,0].to_numpy(copy=True)
        assert len(h)==168 and len(v)==24
        assert np.array_equal(task.past_time.iloc[:,0].to_numpy(),h)
        scalar=np.array([r*v[s] if s<=j<s+d else v[j] for j in range(24)])
        assert np.array_equal(y,scalar)
        alt=np.array([r*v[j] if s<=j<s+d else v[j] for j in range(24)])
        histories.append(h);backgrounds.append(v);anchors.append(y);scaled.append(alt);starts.append(s);durations.append(d);ratios.append(r);seeds.append(seed)
        texts.append({'seed':seed,'forecast_start':str(before.index[-24]),'history_start':str(before.index[0]),'scenario':task.scenario,'background':task.background,'event_start_index':int(s),'event_duration_hours':int(d),'event_ratio':r})
    except Exception as e:failures.append({'seed':seed,'type':type(e).__name__,'message':str(e)})
assert seeds,'No valid configurations'
np.savez(p/'probe/INPUTS.npz',history=np.array(histories),event_start=np.array(starts,dtype=np.int64),event_duration=np.array(durations,dtype=np.int64),event_ratio=np.array(ratios),seed=np.array(seeds,dtype=np.int64))
np.savez(p/'probe/TARGETS.npz',background=np.array(backgrounds),reference=np.array(anchors),pointwise=np.array(scaled))
(p/'probe/SCENARIOS.json').write_text(json.dumps(texts,indent=2)+'\n',encoding='utf-8')
(p/'probe/CONSTRUCTION.json').write_text(json.dumps({'valid_seeds':seeds,'failed_instances':failures,'reference_operation_equal_to_scalar':True,'source_method_bodies':functions,'sampler_change':'One preselected source series replaces source sensor sampling; original random window/event methods retained','reference_benchmark_reproduction':False,'warnings':warning_records},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'valid':len(seeds),'failures':failures,'reference_operation_exact':True,'warnings':len(warning_records)},indent=2))
