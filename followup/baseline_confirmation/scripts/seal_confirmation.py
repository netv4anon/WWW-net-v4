"""Seal all admitted inputs and numerical choices before the first new inference."""
import hashlib,json,py_compile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESOURCE=Path('F:/docphase/Traffic_NETV4_交通评测资源与方法研究/workspace/NETV4_WWW_input_scope_20261011/resource')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
out=ROOT/'admission/CONFIRMATION_FREEZE.json'
assert not out.exists() and not (ROOT/'predictions').exists() and not (ROOT/'scores').exists()
assert read(ROOT/'profiles/ADMISSION_RESULT.json')['status']=='ADMITTED'
assert read(ROOT/'admission/CONSTRUCTOR_REGRESSION.json')['all_original_array_elements_exact']
assert read(ROOT/'admission/LOCAL_MODEL_READINESS.json')['all_six_weights_hash_matched']
files=[p for p in sorted((ROOT/'scripts').glob('*.py'))]
files += [ROOT/p for p in ('profiles/catalog.npz','profiles/ADMISSION_RESULT.json','profiles/QUALITY_ALL_CHANNELS.csv',
    'task/inputs.npz','task/evaluation.npz','task/catalog.npz','task/BINDING.json','task/DESCRIPTION.json',
    'admission/SOURCE_INTAKE_FREEZE.json','admission/LOCAL_MODEL_READINESS.json','admission/CONSTRUCTOR_REGRESSION.json',
    'raw/basel_20240902_20240908.json','incoming_pro/VALIDATION_ADMISSION_ADDENDUM_zh.md',
    'incoming_pro/VALIDATION_READINESS.json','incoming_pro/NET_V4_WWW_frozen_baseline.tex')]
for p in (ROOT/'scripts').glob('*.py'):
    py_compile.compile(str(p),doraise=True)
gwn=RESOURCE/'components/netv4_gwn'
bindings=read(ROOT/'admission/LOCAL_MODEL_READINESS.json')['models']
for m in bindings:assert sha(Path(m['actual_weight']))==m['expected_weight_sha256']
freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'local_design_freeze_not_public_preregistration':True,
    'primary_question':'On all admitted new-profile cutoff-spanning contexts, each of six frozen GWN models under both target laws: A>0 and A_N_given_C<0',
    'models':bindings,'gwn_component':str(gwn),'backend':'cuda-FP32-channels_last',
    'numerics':{'deterministic_algorithms':True,'cudnn_benchmark':False,'TF32':False,
        'batch_size':256,'loss_accumulation':'float64','effect_zero_tolerance_mae':.001,
        'raw_signs_also_reported':True,'robust_confirmation':'A>0.001 and A_N_given_C<-0.001 in every primary combination',
        'structural_prediction_agreement_tolerance':.001,
        'no_post_effect_backend_selection':True},
    'fixed_task':{'history':12,'future':12,'slots':9,'center_copy':1,'timings':[[8,4],[10,8],[14,4],[18,6]],
        'clock_hours':[3,9,15,21],'weekdays':list(range(7)),'seed':2026100907,'target_laws':['legacy','residual'],
        'same_forecasts_scored_against_both_targets':True,'all_73_profiles_in_all_input_slots':True},
    'source_intake_sha256':sha(ROOT/'admission/SOURCE_INTAKE_FREEZE.json'),
    'hourly_uniform_quarter_allocation_assumed':True,
    'campaign_files':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],
    'pinned_external_code':[{'path':str(gwn/p),'sha256':sha(gwn/p)} for p in ('model_adapter.py','reader_adapter.py','vendor/model.py')],
    'all_primary_and_profile_timing_failures_retained':True,'no_training_or_model_selection':True,
    'stop_after_one_complete_source_confirmation':True,'new_inference_at_freeze':0,'new_scoring_at_freeze':0}
out.write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'admission/CONFIRMATION_FREEZE_SHA256.txt').write_text(sha(out)+'\n',encoding='utf-8')
(ROOT/'admission/MECHANICAL_NOTES.json').write_text(json.dumps({
    'source_fetch':'Sandbox schannel credential failure; rerun the identical fixed URL outside sandbox succeeded.',
    'profile_intake':'First formatting pass encountered a null lane ID before writing outputs; missing channel IDs are rejected, per the required lane/variable identity contract. No inference or loss used.',
    'numeric_scope':'1e-3 is fixed before any new inference as an ambiguity band; it is not claimed as a CPU/CUDA error guarantee.',
    'constructor_regression':'All 16702704 old elements exactly regenerated; no original model outputs recomputed.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'READY_FOR_ONE_FROZEN_CONFIRMATION','freeze_sha256':sha(out),'files_bound':len(files),
                  'models':len(bindings),'new_inference':0,'new_scoring':0}),flush=True)
