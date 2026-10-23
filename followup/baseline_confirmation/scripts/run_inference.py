"""Generate all six frozen forecasts without opening targets or computing effects."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,importlib.util,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
freeze=read(ROOT/'admission/CONFIRMATION_FREEZE.json')
assert sha(ROOT/'admission/CONFIRMATION_FREEZE.json')==(ROOT/'admission/CONFIRMATION_FREEZE_SHA256.txt').read_text().strip()
for item in freeze['campaign_files']:
    assert sha(ROOT/item['path'])==item['sha256'],item['path']
for item in freeze['pinned_external_code']:
    assert sha(Path(item['path']))==item['sha256']
assert freeze['backend']=='cuda-FP32-channels_last'
torch.set_num_threads(4)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
gwn=Path(freeze['gwn_component'])
sys.path.insert(0,str(gwn))
spec=importlib.util.spec_from_file_location('confirmation_frozen_reader',gwn/'reader_adapter.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
with np.load(ROOT/'task/inputs.npz',allow_pickle=False) as z:
    ids=z['sample_id'].copy();history=z['history'].copy()
directory=ROOT/'predictions';directory.mkdir(exist_ok=False)
records=[]
for binding in freeze['models']:
    model_id='%s__GraphWaveNet_seed%d'%(binding['variant'],binding['seed'])
    assert sha(Path(binding['actual_weight']))==binding['expected_weight_sha256']
    tic=time.perf_counter()
    print(json.dumps({'stage':'predicting','model':model_id,'input_rows':len(ids)}),flush=True)
    prediction=reader.predict(history,binding['variant'],binding['seed'],'cuda')
    assert prediction.shape==(len(ids),12) and np.isfinite(prediction).all()
    path=directory/(model_id+'.npz')
    np.savez_compressed(path,sample_id=ids,prediction=prediction)
    row={'id':model_id,'variant':binding['variant'],'seed':binding['seed'],'file':str(path.relative_to(ROOT)),
         'sha256':sha(path),'weight_sha256':binding['expected_weight_sha256'],'scaler':binding['scaler'],
         'rows':len(ids),'seconds':time.perf_counter()-tic,'finite':True,'inference_utc':datetime.now(timezone.utc).isoformat()}
    records.append(row)
    (directory/'PROGRESS.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'stage':'prediction_saved',**row}),flush=True)
    reader._CACHE.clear();torch.cuda.empty_cache()
seal={'all_six_saved_before_effect_scoring':True,'sealed_utc':datetime.now(timezone.utc).isoformat(),
      'freeze_sha256':sha(ROOT/'admission/CONFIRMATION_FREEZE.json'),'models':records,
      'new_training_runs':0,'target_laws_reuse_same_predictions':True,'backend':freeze['backend']}
(directory/'PREDICTION_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'stage':'all_six_sealed','rows_per_model':len(ids)}),flush=True)
