"""Fix one source, calendar window and all quality rules before fetching its cohort."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace')
spec_path=ROOT/'admission/SOURCE_INTAKE_FREEZE.json'
assert not spec_path.exists(), 'Source intake already fixed'
pattern=r'data\.bs\.ch|\bbasel\b|\bbasler\b|verkehrszaehl|verkehrszähl'
command=['rg','-l','-i','--glob','*.md','--glob','*.json','--glob','*.py','--glob','*.tex',
         '--glob','!**/vendor/**','--glob','!**/site-packages/**','--glob','!**/.venv*/**',
         '--glob','!**/tokenizer.json','--glob','!**/vocab.json',pattern,str(ARCHIVE)]
search=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert search.returncode==1 and not search.stdout and not search.stderr, 'Prior source match or search failure'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
params={'select':'sitecode,zst_nr,sitename,directionname,lanecode,lanename,datetimefrom,datetimeto,total,valuesapproved,valuesedited,geo_point_2d',
        'where':"datetimefrom >= date'2024-09-01T22:00:00Z' AND datetimefrom < date'2024-09-08T22:00:00Z'",
        'order_by':'sitecode,directionname,lanecode,datetimefrom','timezone':'UTC'}
url='https://data.bs.ch/api/explore/v2.1/catalog/datasets/100006/exports/json?'+urlencode(params)
spec={'fixed_utc':datetime.now(timezone.utc).isoformat(),'source':'Open Data Basel-Stadt dataset 100006',
    'publisher':'Amt fuer Mobilitaet, Basel-Stadt','license':'CC BY 4.0',
    'metadata_sha256':sha(ROOT/'admission/basel_dataset_metadata.json'),
    'request':url,'window_local_half_open':['2024-09-02T00:00:00+02:00','2024-09-09T00:00:00+02:00'],
    'population':'all sitecode/directionname/lanecode channels returned for the fixed week; no count cap or outcome selection',
    'source_units':'vehicles per observed one-hour lane/direction interval',
    'profile_units':'vehicles per nominal 15-minute interval, using fixed uniform allocation of each hourly count into four quarters',
    'allocation':'q[d,4*h:4*h+4] = hourly_total/4; no interpolation or smoothing added at intake',
    'allocation_limitation':'quarter-hour detail is assumed uniform, not observed; original constructor smoothing/noise stays unchanged',
    'quality':{'expected_unique_hourly_intervals':168,'interval_seconds':3600,
        'require_nonnegative_finite_integer_count':True,'require_valuesapproved':1,'require_valuesedited':0,
        'require_complete_hour_grid':True,'any_duplicate_interval_rejects_channel':True,
        'imputation':'none','weekly_mean':'strictly positive'},
    'duplicate_rule':{'metric':'RMS difference / maximum of the two weekly means',
        'threshold':0.01,'exclude_against_all_102_legacy_profiles':True,
        'within_new':'lexicographic greedy representatives; discard exact/near duplicates against previously retained representatives',
        'all_slots_use_admitted_catalog_only':True},
    'source_exposure_audit':{'scope':str(ARCHIVE),'command':command,'exit_code':1,'matches':[],
        'interpretation':'No use found in searchable retained project code/docs/metadata. Not a proof of stochastic independence or an author memory attestation.'},
    'pre_intake_source_decisions':[
        {'source':'UK WebTRIS','decision':'excluded: prior project use'},
        {'source':'Finland Digitraffic','decision':'excluded: prior project use; documentation fetch failed, no cohort fetched'},
        {'source':'Basel','decision':'selected after metadata/schema check; hourly allocation assumption fixed before cohort fetch'}],
    'minimum_profiles':8,'no_source_switch_after_inference_or_scoring':True,
    'if_admission_fails':'stop and retain frozen paper; no outcome-driven source retry',
    'new_cohort_fetched':False,'new_inference':0,'new_scoring':0,'public_preregistration':False}
spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'admission/SOURCE_REQUEST_URL.txt').write_text(url,encoding='utf-8')
(ROOT/'admission/SOURCE_INTAKE_SHA256.txt').write_text(sha(spec_path)+'\n',encoding='utf-8')
print(json.dumps({'source_fixed':spec['source'],'intake_manifest_sha256':sha(spec_path),'request':url}))
