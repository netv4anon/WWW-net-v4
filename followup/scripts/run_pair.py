"""One sealed pass of both representations through the same six frozen weights."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,importlib.util,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=read(ROOT/'admission/FOLLOWUP_DESIGN_FREEZE.json')
assert sha(ROOT/'admission/FOLLOWUP_DESIGN_FREEZE.json')==(ROOT/'admission/FOLLOWUP_DESIGN_SHA256.txt').read_text().strip()
for r in freeze['files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True);torch.manual_seed(0)
sys.path.insert(0,str(ROOT/'models'))
import reader_adapter
out=ROOT/'paired/predictions';out.mkdir(exist_ok=False)
models=[]
for variant in ('packaged','working'):
    for seed in (1,2,3):
        model_id=variant+'__GraphWaveNet_seed'+str(seed)
        for condition in ('original','hour_uniform'):
            tic=time.perf_counter()
            with np.load(ROOT/'paired'/condition/'inputs.npz',allow_pickle=False) as z:
                ids=z['sample_id'].copy();history=z['history'].copy()
            prediction=reader_adapter.predict(history,variant,seed,'cuda')
            assert prediction.shape==(len(ids),12) and np.isfinite(prediction).all()
            file=out/(model_id+'__'+condition+'.npz')
            np.savez_compressed(file,sample_id=ids,prediction=prediction)
            row={'model':model_id,'variant':variant,'seed':seed,'condition':condition,
                'file':file.relative_to(ROOT).as_posix(),'sha256':sha(file),'rows':len(ids),
                'weight_sha256':sha(ROOT/'models/runs'/model_id/'best_weights.pt'),
                'finished_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-tic}
            models.append(row);print(json.dumps(row),flush=True)
        reader_adapter._CACHE.clear();torch.cuda.empty_cache()
seal={'all_12_saved_before_pair_effect_scoring':True,'sealed_utc':datetime.now(timezone.utc).isoformat(),
    'design_sha256':sha(ROOT/'admission/FOLLOWUP_DESIGN_FREEZE.json'),'forecasts':models,'new_fits':0,
    'backend':freeze['backend'],'conditions':2,'checkpoints':6,'new_observed_records':0}
(out/'PREDICTION_SEAL.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'ALL_TWELVE_SAVED','new_fits':0}),flush=True)
