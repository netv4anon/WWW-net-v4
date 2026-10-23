"""Freeze one post-Basel paired sensitivity design before any paired inference."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert not (ROOT/'paired/predictions').exists()
files=[]
for directory in ('paired','portable','models','incoming_pro'):
    for p in sorted((ROOT/directory).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
for name in ('construct_pair.py','run_pair.py','score_pair.py','freeze_pair.py'):
    p=ROOT/'scripts'/name;files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
design={'frozen_utc':datetime.now(timezone.utc).isoformat(),'analysis_identity':'post-Basel follow-up sensitivity; not an independent confirmatory replacement',
    'known_prior_results':'Basel all 12 total responses positive; conditional signs 7 negative/5 positive. Existing saved-prediction decomposition was already examined.',
    'one_intervention':'Apply H to all 102 original processed 7x96 profiles: repeat each within-hour four-value arithmetic mean four times.',
    'fixed':'Original center/reference assignments, slot order and center duplicate, profile IDs, seeds/draws, dates, severity, timing, constructor mechanism, six actual checkpoints, original scalers and all settings.',
    'targets':'Targets regenerate with the paired background representation. They are not held identical across representations. Four fixed crossed scores separate prediction/target contributions descriptively, along both paths.',
    'primary_comparisons':12,'primary_cohort':'All 2856 original-profile cutoff-spanning scenarios; 102 equal-weight profiles and all seeds',
    'primary_quantities':['A_original','A_hour_uniform','A_N_given_C_original','A_N_given_C_hour_uniform','paired_change_in_A_N_given_C'],
    'additional_fixed_outputs':['All timings and profile results','Raw and 1e-3 numerical sign bands','Same-target crossed responses R00,R10,R01,R11','Both path decompositions and interaction','Prediction increment magnitudes','Equal-history future-start controls','Baseline forecast discrepancy versus previously saved output'],
    'no_practical_significance_threshold_claimed':True,'no_independence_based_intervals':True,
    'backend':'CUDA FP32 channels_last, batch256, deterministic algorithms, TF32 off; float64 loss accumulation',
    'new_training':0,'new_models':0,'condition_evaluations':12,'model_checkpoints':6,
    'no_source_search_or_threshold_selection':True,'stop_rule':'Run these two representations once for all six checkpoints; preserve failures/near-zero effects. No new source, architecture, seed, interpolation or amplitude search.',
    'cannot_establish':'Does not identify the cause of Basel reversals, recover true quarter-hour dynamics, or calibrate actual incidents.',
    'construction_check':read(ROOT/'admission/PAIR_CONSTRUCTION_CHECK.json'),'files':files}
path=ROOT/'admission/FOLLOWUP_DESIGN_FREEZE.json';assert not path.exists()
path.write_text(json.dumps(design,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'admission/FOLLOWUP_DESIGN_SHA256.txt').write_text(sha(path)+'\n',encoding='utf-8')
print(json.dumps({'frozen':True,'files':len(files),'sha256':sha(path),'paired_inference_at_freeze':0}),flush=True)
