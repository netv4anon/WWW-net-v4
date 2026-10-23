"""Run thirteen predeclared frozen predictors; predictor reads only input arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    for row in json.loads((ROOT/'PINS.json').read_text(encoding='utf-8'))['files']:
        assert sha(ROOT/row['path'])==row['sha256']
    with np.load(ROOT/'task/inputs.npz',allow_pickle=False) as z:ids=z['sample_id'];history=z['history']
    assert history.shape==(len(ids),12,9)
    out=ROOT/'predictions';out.mkdir(exist_ok=False)
    models=[]
    def save(name,pred,**fields):
        pred=np.maximum(pred,0).astype(np.float32)
        assert pred.shape==(len(ids),12) and np.isfinite(pred).all()
        path=out/(name+'.npz')
        np.savez_compressed(path,sample_id=ids,prediction=pred)
        models.append(dict(id=name,file=path.relative_to(ROOT).as_posix(),sha256=sha(path),**fields))
        print(json.dumps({'prediction_complete':name,'rows':len(ids)}),flush=True)
    center=history[:,:,0]
    save('LastValue',np.repeat(center[:,-1:],12,axis=1),variant='shared',kind='LastValue',seed=None,center_only=True)
    save('HistoricalMean',np.repeat(center.mean(axis=1,keepdims=True),12,axis=1),variant='shared',kind='HistoricalMean',seed=None,center_only=True)
    slope=(center[:,-1]-center[:,-4])/3
    save('LinearDrift',center[:,-1:]+slope[:,None]*np.arange(1,13)[None,:],variant='shared',kind='LinearDrift',seed=None,center_only=True)
    spec=importlib.util.spec_from_file_location('frozen_stid_model',ROOT/'models/original_train.py')
    architecture=importlib.util.module_from_spec(spec);spec.loader.exec_module(architecture)
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for variant in ('packaged','working'):
        for kind in ('RidgeCenter','RidgeLocal9'):
            path=ROOT/'models'/variant/kind/'weights.npz'
            with np.load(path,allow_pickle=False) as z:
                features=center.copy() if kind=='RidgeCenter' else history.reshape(len(ids),-1).copy()
                # Match StandardScaler's float32 in-place transform used by the original entry point.
                features-=z['mean'];features/=z['scale']
                prediction=features.astype(np.float64)@z['coef'].T+z['intercept']
            save(variant+'__'+kind,prediction,variant=variant,kind=kind,seed=None,
                 center_only=kind=='RidgeCenter',weight_sha256=sha(path))
        for seed in (1,2,3):
            name='STID_NoCalendar_seed'+str(seed);directory=ROOT/'models'/variant/name
            scaler=json.loads((directory/'scaler.json').read_text(encoding='utf-8'))
            model=architecture.stid(ROOT/'models').to(device)
            model.load_state_dict(torch.load(directory/'best_weights.pt',map_location=device,weights_only=True))
            model.eval();chunks=[]
            with torch.no_grad():
                for start in range(0,len(ids),1024):
                    inputs=(history[start:start+1024]-scaler['mean'])/scaler['std']
                    tensor=torch.from_numpy(inputs[:,:,:,None]).to(device)
                    result=model(tensor,None,batch_seen=0,epoch=0,train=False)[:,:,0,0]*scaler['std']+scaler['mean']
                    chunks.append(result.cpu().numpy())
            save(variant+'__'+name,np.concatenate(chunks),variant=variant,kind='STID_NoCalendar',seed=seed,
                 center_only=True,weight_sha256=sha(directory/'best_weights.pt'))
    dump(out/'MODELS.json',models)
    dump(out/'SEAL.json',{'protocol_sha256':sha(ROOT/'PROTOCOL_ZH.md'),
                          'inputs_sha256':sha(ROOT/'task/inputs.npz'),'runner_sha256':sha(Path(__file__)),
                          'new_model_fits':0,'new_forecast_configurations':len(models),
                          'forecast_rows_per_configuration':len(ids),'predictor_opened_evaluation_targets':False,
                          'frozen_prediction_files':models})

if __name__=='__main__':main()
