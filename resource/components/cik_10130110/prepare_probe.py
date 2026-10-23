import csv,hashlib,io,json,re
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parent
raw=(p/'sources/trafficfresh_prefix.bin').read_bytes();hdr=(p/'sources/trafficfresh_prefix_headers_sanitized.txt').read_text()
assert len(raw)==1048576
status=re.findall(r'^HTTP/\S+\s+([^\r\n]+)',hdr,re.M)
ranges=re.findall(r'^content-range:\s*([^\r\n]+)',hdr,re.M|re.I)
assert status[-1].startswith('206') and ranges[-1]=='bytes 0-1048575/304418602',(status,ranges)
assert '72f9682ad1ce00006090e61adedff5898ec3f963' in hdr
for stop in reversed([m.end() for m in re.finditer(b'\n',raw)]):
    try:
        rows=list(csv.reader(io.StringIO(raw[:stop].decode('utf-8'),newline=''),delimiter='\t',strict=True))
        if len(rows)>1 and all(len(x)==len(rows[0]) for x in rows):break
    except (csv.Error,UnicodeDecodeError):pass
else:raise RuntimeError('No complete tabular prefix')
frame=pd.DataFrame(rows[1:],columns=rows[0]);print('columns',list(frame.columns));print('records',len(frame),'sensors',frame['sensor_id'].nunique())
frame['sensor_num']=pd.to_numeric(frame['sensor_id'],errors='raise')
frame['timestamp']=pd.to_datetime(frame['Hour'],format='%m/%d/%Y %H:%M',errors='raise')
frame['occupancy']=pd.to_numeric(frame['Occupancy (%)'],errors='coerce')
summary=[];selected=None
for sid,group in frame.groupby('sensor_num',sort=True):
    g=group.sort_values('timestamp').copy();good=np.isfinite(g.occupancy)&(g.occupancy>=0)
    gap=g.timestamp.diff().dt.total_seconds()!=3600
    breaks=(~good)|(~good.shift(1,fill_value=False))|gap
    segments=g.groupby(breaks.cumsum(),sort=False)
    eligible=[seg for _,seg in segments if len(seg)>=193 and np.isfinite(seg.occupancy).all() and (seg.occupancy>=0).all() and seg.timestamp.is_unique]
    summary.append({'sensor_id':int(sid),'rows_in_prefix':len(g),'eligible_runs':[len(z) for z in eligible]})
    if int(sid)==10130110 and eligible:selected=eligible[0].copy();selected_sid=int(sid)
if selected is None:
    (p/'PROBE_FEASIBILITY.json').write_text(json.dumps({'eligible':False,'source_counts':summary},indent=2));raise SystemExit('No eligible fixed-prefix source; no expansion')
(p/'probe').mkdir(exist_ok=True)
selected[['timestamp','occupancy']].to_csv(p/'probe/selected_series.csv',index=False,date_format='%Y-%m-%d %H:%M:%S',float_format='%.17g')
(p/'probe/source_prefix_complete.tsv').write_bytes(raw[:stop])
tree=json.loads((p/'sources/cik_tree_pinned.json').read_text())
files={'cik_traffic_tasks.py':'cik_benchmark/tasks/traffic_tasks.py','cik_pred_change_tasks.py':'cik_benchmark/tasks/pred_change_tasks.py','cik_pems_tasks.py':'cik_benchmark/tasks/pems_tasks.py','cik_data_pems.py':'cik_benchmark/data/pems.py','cik_base.py':'cik_benchmark/base.py','cik_evaluation.py':'cik_benchmark/evaluation.py','cik_utils.py':'cik_benchmark/utils/__init__.py'}
code=[]
for local,remote in files.items():
    data=(p/'sources'/local).read_bytes();item=next(x for x in tree['tree'] if x['path']==remote)
    git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert git==item['sha'],local
    code.append({'local':'sources/'+local,'remote':remote,'git_blob_sha1':git,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
record={'eligible':True,'source_commit':'72f9682ad1ce00006090e61adedff5898ec3f963','source_full_file_bytes':304418602,'source_range':ranges[-1],'prefix_bytes':len(raw),'prefix_sha256':hashlib.sha256(raw).hexdigest(),'complete_prefix_bytes':stop,'complete_prefix_rows':len(frame),'source_counts':summary,'selected_sensor_id':selected_sid,'selected_series_rows':len(selected),'selected_start':str(selected.timestamp.iloc[0]),'selected_end':str(selected.timestamp.iloc[-1]),'selection_before_scores':True,'selection_rule':'Frozen remaining sensor 10130110; earliest eligible run','primary_outcome_seen_before_replication_protocol':True,'selected_series_sha256':hashlib.sha256((p/'probe/selected_series.csv').read_bytes()).hexdigest(),'cik_git_commit':'73f46016f8c5643bf6799ca31875eab3e8d0d075','reference_source_files':code,'numpy':np.__version__,'pandas':pd.__version__}
(p/'PROBE_SOURCE_BINDING.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
def redact(m):
    u=urlsplit(m.group(1));return 'Location: '+urlunsplit((u.scheme,u.netloc,u.path,'',''))+' [query omitted]'
safe=re.sub(r'^location:\s*([^\r\n]+)',redact,hdr,flags=re.M|re.I)
safe=re.sub(r'^(set-cookie|cookie):[^\r\n]+',r'\1: [omitted]',safe,flags=re.M|re.I)
(p/'sources/trafficfresh_prefix_headers_sanitized.txt').write_text(safe,encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='reference_source_files'},indent=2))
