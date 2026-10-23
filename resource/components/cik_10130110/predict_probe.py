"""Prediction step reads histories and numeric scenario fields only."""
import hashlib,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent/'probe'
x=np.load(p/'INPUTS.npz',allow_pickle=False)
h=x['history'];n=len(h)
bases={'last_value':np.repeat(h[:,-1:],24,axis=1),'last_day':h[:,-24:].copy(),'mean_daily_profile':h.reshape(n,7,24).mean(axis=1)}
outputs={}
for base,q in bases.items():
    outputs[base+'__none']=q.copy()
    for response in ['anchor','pointwise']:
        pred=q.copy()
        for i,(start,duration,ratio) in enumerate(zip(x['event_start'],x['event_duration'],x['event_ratio'])):
            stop=start+duration
            pred[i,start:stop]=ratio*(q[i,start] if response=='anchor' else q[i,start:stop])
        outputs[base+'__'+response]=pred
np.savez(p/'PREDICTIONS.npz',**outputs)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
hashes={name:hashlib.sha256(a.tobytes()).hexdigest() for name,a in outputs.items()}
seal={'input_file_sha256':sha(p/'INPUTS.npz'),'prediction_file_sha256':sha(p/'PREDICTIONS.npz'),'prediction_code_sha256':sha(Path(__file__)),'declared_response_cells':len(outputs),'distinct_prediction_arrays':len(set(hashes.values())),'array_sha256':hashes,'target_arrays_read':False,'learnt_models':0,'fits':0}
(p/'PREDICTION_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in seal.items() if k!='array_sha256'},indent=2))
