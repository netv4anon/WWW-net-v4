"""Apply the pre-fixed Basel source admission without viewing model outputs."""
import csv,hashlib,json
from collections import defaultdict,Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESOURCE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_WWW_input_scope_20261011/resource')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
spec=read(ROOT/'admission/SOURCE_INTAKE_FREEZE.json')
assert sha(ROOT/'admission/SOURCE_INTAKE_FREEZE.json')==(ROOT/'admission/SOURCE_INTAKE_SHA256.txt').read_text().strip()
assert not (ROOT/'profiles').exists()
raw=ROOT/'raw/basel_20240902_20240908.json'
records=read(raw)
assert isinstance(records,list) and records
groups=defaultdict(list)
for record in records:
    groups[(record['sitecode'],record['directionname'],record['lanecode'])].append(record)
local_zone=timezone(timedelta(hours=2))  # Frozen September week: Zurich summer offset; no clock transition.
start=datetime(2024,9,2,tzinfo=local_zone)
expected={start+timedelta(hours=i) for i in range(168)}
with np.load(RESOURCE/'components/netv4_mv/task/catalog.npz',allow_pickle=False) as z:
    old=z['profile'].astype(np.float64);old_ids=z['profile_id'].tolist()
old_means=old.mean(axis=(1,2))
accepted=[];names=[];audit=[];manifest=[];reasons=Counter()
for key,items in sorted(groups.items(),key=lambda p:tuple('' if k is None else str(k) for k in p[0])):
    site,direction,lane=key
    name='Basel100006_%s_L%s_%s'%(site,'NA' if lane is None else '%02d'%lane,hashlib.sha256(str(direction).encode()).hexdigest()[:8])
    failure=[];clock=[];values=[]
    if site is None or direction is None or lane is None:failure.append('missing_channel_identifier')
    for r in items:
        a=datetime.fromisoformat(r['datetimefrom']).astimezone(local_zone)
        b=datetime.fromisoformat(r['datetimeto']).astimezone(local_zone)
        clock.append(a)
        if (b-a).total_seconds()!=3600:failure.append('wrong_interval')
        if r['valuesapproved']!=1:failure.append('unapproved')
        if r['valuesedited']!=0:failure.append('edited')
        v=r['total']
        if not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 or v!=int(v):
            failure.append('invalid_count')
        values.append(v)
    if len(clock)!=168 or set(clock)!=expected:failure.append('incomplete_hour_grid')
    if len(set(clock))!=len(clock):failure.append('duplicate_hour')
    q=None;near_old=None;near_new=None
    if not failure:
        lookup=dict(zip(clock,values))
        hourly=np.array([lookup[start+timedelta(hours=i)] for i in range(168)],dtype=np.float64).reshape(7,24)
        q=np.repeat(hourly/4,4,axis=1).astype(np.float32)
        mean=float(q.mean())
        if mean<=0:failure.append('nonpositive_weekly_mean')
        else:
            distances=np.sqrt(((old-q.astype(np.float64))**2).mean(axis=(1,2)))/np.maximum(old_means,mean)
            if distances.min()<=.01:
                near_old=old_ids[int(distances.argmin())];failure.append('near_legacy_profile')
            if accepted:
                new=np.stack(accepted).astype(np.float64)
                distances_new=np.sqrt(((new-q)**2).mean(axis=(1,2)))/np.maximum(new.mean(axis=(1,2)),mean)
                if distances_new.min()<=.01:
                    near_new=names[int(distances_new.argmin())];failure.append('near_retained_new_profile')
    reasons.update(set(failure))
    audit.append({'profile_id':name,'sitecode':site,'directionname':direction,'lane':lane,'source_rows':len(items),
                  'eligible':not failure,'reasons':'|'.join(sorted(set(failure))),
                  'near_legacy':near_old or '', 'near_retained_new':near_new or ''})
    if not failure:
        names.append(name);accepted.append(q)
        manifest.append({'profile_id':name,'sitecode':site,'directionname':direction,'lanecode':lane,
            'sitename':items[0]['sitename'],'geo_point_2d':items[0]['geo_point_2d'],
            'source_intervals':168,'quarter_values':672,'conversion':'each hourly total/4 repeated for four quarters',
            'profile_float32_sha256':hashlib.sha256(q.tobytes()).hexdigest(),
            'quarter_mean':float(q.mean()),'quarter_min':float(q.min()),'quarter_max':float(q.max())})
out=ROOT/'profiles';out.mkdir()
with (out/'QUALITY_ALL_CHANNELS.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
result={'status':'ADMITTED' if len(accepted)>=8 else 'STOP_INSUFFICIENT_ELIGIBLE_PROFILES',
        'source_raw_rows':len(records),'source_channels':len(groups),'eligible_profiles':len(accepted),
        'excluded_channels':len(groups)-len(accepted),'failure_reason_counts':dict(reasons),
        'source_sha256':sha(raw),'source_bytes':raw.stat().st_size,
        'source_http_headers_sha256':sha(ROOT/'raw/basel_week_headers.txt'),
        'source_intake_sha256':sha(ROOT/'admission/SOURCE_INTAKE_FREEZE.json'),
        'legacy_catalog_sha256':sha(RESOURCE/'components/netv4_mv/task/catalog.npz'),
        'input_units':'vehicle counts per nominal quarter hour; hourly-to-quarter uniform allocation assumption',
        'original_high_frequency_detail_observed':False,'all_input_slots_from_this_new_catalog':True,
        'profiles_are_not_independent_incidents':True,'new_inference':0,'new_scoring':0,
        'eligible_manifest':manifest}
(out/'ADMISSION_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if len(accepted)>=8:
    order=np.argsort(np.array(names),kind='stable')
    np.savez_compressed(out/'catalog.npz',profile_id=np.array(names)[order],profile=np.stack(accepted)[order])
print(json.dumps({k:v for k,v in result.items() if k!='eligible_manifest'},ensure_ascii=True))
