"""Independent scalar replay; NumPy is used only for archives and pinned RNG."""
import csv,hashlib,json,math,re
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;data=p/'probe'
binding=json.loads((p/'PROBE_SOURCE_BINDING.json').read_text());result=json.loads((data/'RESULTS.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(p/'sources/trafficfresh_prefix.bin')==binding['prefix_sha256']
raw=(p/'sources/trafficfresh_prefix.bin').read_bytes();complete=(data/'source_prefix_complete.tsv').read_bytes()
assert complete==raw[:len(complete)]
groups=defaultdict(list)
with (data/'source_prefix_complete.tsv').open(encoding='utf-8',newline='') as f:
    for row in csv.DictReader(f,delimiter='\t'):
        try:value=float(row['Occupancy (%)'])
        except ValueError:value=float('nan')
        groups[int(row['sensor_id'])].append((datetime.strptime(row['Hour'],'%m/%d/%Y %H:%M'),value))
selected=None
for sid in sorted(groups):
    runs=[];run=[]
    for stamp,value in sorted(groups[sid]):
        valid=math.isfinite(value) and value>=0
        if (not valid) or (run and stamp-run[-1][0]!=timedelta(hours=1)):
            if len(run)>=193:runs.append(run)
            run=[]
        if valid:run.append((stamp,value))
    if len(run)>=193:runs.append(run)
    if runs:selected=runs[0];sensor=sid;break
assert sensor==binding['selected_sensor_id'] and len(selected)==binding['selected_series_rows']
x=np.load(data/'INPUTS.npz',allow_pickle=False);targets=np.load(data/'TARGETS.npz',allow_pickle=False);pred=np.load(data/'PREDICTIONS.npz',allow_pickle=False)
assert set(x.files)=={'history','event_start','event_duration','event_ratio','seed'}
texts=json.loads((data/'SCENARIOS.json').read_text());n=len(texts)
assert n==32 and x['seed'].tolist()==list(range(32))
max_history_error=0.;max_target_error=0.;max_prediction_error=0.;prediction_values=0;target_values=0
all_expected={name:[] for name in pred.files};all_targets={'reference':[],'pointwise':[]};mask=[]
for i,seed in enumerate(range(32)):
    rng=np.random.RandomState(seed)
    assert rng.choice(['traffic_fresh'])=='traffic_fresh'
    offset=int(rng.randint(0,len(selected)-192))
    stamps=[v[0] for v in selected[offset:offset+192]];values=[v[1] for v in selected[offset:offset+192]]
    h=values[:168];v=values[168:]
    assert not all(abs(a)<=1e-8 for a in h) and not all(abs(a)<=1e-8 for a in v)
    duration=int(rng.choice([1,2,3,4,5,6,7]))
    # Preserve the reference RNG's defined tie ordering for start candidates.
    start=int(rng.choice(np.argsort(np.array(v[:-(duration+1)]))[-5:][::-1]))
    ratio=float(rng.choice([.1,.2,.3,.4,.5]))
    assert (start,duration,ratio)==(int(x['event_start'][i]),int(x['event_duration'][i]),float(x['event_ratio'][i]))
    assert texts[i]['history_start']==str(stamps[0]) and texts[i]['forecast_start']==str(stamps[168])
    event_date=stamps[168+start].strftime('%Y-%m-%d %H:%M:%S')
    expected_scenario=f"Suppose that there is an accident on the road and there is {ratio*100}% of the usual traffic from {event_date} for {duration} {'hours' if duration>1 else 'hour'}."
    assert texts[i]['scenario']==expected_scenario
    max_history_error=max(max_history_error,max(abs(a-float(b)) for a,b in zip(h,x['history'][i])))
    event=[start<=j<start+duration for j in range(24)];mask.append(event)
    ys={'reference':[ratio*v[start] if event[j] else v[j] for j in range(24)],'pointwise':[ratio*v[j] if event[j] else v[j] for j in range(24)]}
    for target,y in ys.items():
        all_targets[target].append(y)
        max_target_error=max(max_target_error,max(abs(a-float(b)) for a,b in zip(y,targets[target][i])));target_values+=24
    bases={'last_value':[h[-1]]*24,'last_day':h[-24:],'mean_daily_profile':[math.fsum(h[day*24+j] for day in range(7))/7 for j in range(24)]}
    for base,q in bases.items():
        for response in ['none','anchor','pointwise']:
            expected=[(ratio*(q[start] if response=='anchor' else q[j]) if event[j] and response!='none' else q[j]) for j in range(24)]
            name=base+'__'+response;all_expected[name].append(expected)
            max_prediction_error=max(max_prediction_error,max(abs(a-float(b)) for a,b in zip(expected,pred[name][i])));prediction_values+=24
max_score_error=0.;score_values=0
for row in result['scores']:
    target=row['target'];name=row['predictor'];full=[];roi=[];pooled=[]
    for i in range(n):
        e=[abs(a-b) for a,b in zip(all_expected[name][i],all_targets[target][i])]
        ev=[v for j,v in enumerate(e) if mask[i][j]]
        full.append(math.fsum(e)/24);roi.append(math.fsum(ev)/len(ev));pooled.extend(ev)
    for actual,expected in zip(row['per_instance_full_mae']+row['per_instance_event_mae'],full+roi):
        max_score_error=max(max_score_error,abs(actual-expected));score_values+=1
    for k,expected in [('full_mae',math.fsum(full)/n),('event_mae_case_equal',math.fsum(roi)/n),('event_mae_pooled',math.fsum(pooled)/len(pooled))]:
        max_score_error=max(max_score_error,abs(row[k]-expected));score_values+=1
assert max(max_history_error,max_target_error,max_prediction_error,max_score_error)<1e-12
for name,h in json.loads((data/'CODE_AND_INPUT_SEAL.json').read_text())['files'].items():assert sha(p/name)==h
expected_signs=[False,False,True]
assert [z['event_mae_case_equal']['sign_changes'] for z in result['within_base_contrasts']]==expected_signs
report={'passed':True,'source_selection_independently_reproduced':True,'source_sensor':sensor,'source_series_records':len(selected),'seed_event_and_scenario_replays':32,'targets_values_checked':target_values,'forecast_values_checked_including_identical_cell':prediction_values,'score_values_checked':score_values,'max_source_history_error':max_history_error,'max_target_error':max_target_error,'max_prediction_error':max_prediction_error,'max_score_error':max_score_error,'all_seals_passed':True,'primary_sign_changes_by_base':dict(zip(['last_value','last_day','mean_daily_profile'],expected_signs)),'primary_sign_change_bases':1,'primary_base_families':3,'independent_execution':'CSV source and scalar Python arithmetic; no reference methods, pandas or scorer invoked; NumPy only archive decoding and pinned RNG/tie order','full_cik_benchmark_reproduced':False,'published_model_rankings_tested':False}
(data/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))

