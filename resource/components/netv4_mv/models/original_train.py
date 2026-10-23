# -*- coding: utf-8 -*-
"""Train documented NET-V4 baselines from the portable snapshot files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def data(root,variant):
    directory=root/"data"/variant
    manifest=json.loads((root/"MANIFEST.json").read_text(encoding="utf-8"))
    expected={r["path"]:r["sha256"] for r in manifest["files"]}
    for name in ["history.npz","targets.npz","splits.npz"]:
        path=directory/name
        assert digest(path)==expected[path.relative_to(root).as_posix()],name
    with np.load(directory/"history.npz",allow_pickle=False) as z:x=z["history"].astype(np.float32)
    with np.load(directory/"splits.npz",allow_pickle=False) as z:
        train=z["train"];val=z["validation"];test=z["test"]
    assert len(np.unique(np.concatenate([train,val,test])))==len(x)
    with np.load(directory/"targets.npz",allow_pickle=False) as z:
        raw=z["future"]
        # Explicitly exclude held-out targets from the training interface.
        y=np.full(raw.shape,np.nan,dtype=np.float32)
        y[train]=raw[train];y[val]=raw[val]
    assert np.isnan(y[test]).all()
    assert np.isfinite(x).all() and np.isfinite(y[train]).all() and np.isfinite(y[val]).all()
    return x,y,train,val,test

def stid(root):
    sys.path.insert(0,str(root/"vendor/stid"))
    from arch import STID
    return STID(num_nodes=9,node_dim=32,input_len=12,input_dim=1,embed_dim=64,output_len=12,num_layer=3,
                temp_dim_tid=0,temp_dim_diw=0,time_of_day_size=96,day_of_week_size=7,
                if_T_i_D=False,if_D_i_W=False,if_node=True)

def fit_ridge(model,x,y,train,test,output):
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    features=x[:,:,0] if model=="RidgeCenter" else x.reshape(len(x),-1)
    scaler=StandardScaler().fit(features[train])
    fitted=Ridge(alpha=1.,solver="cholesky").fit(scaler.transform(features[train]).astype(np.float64),y[train,:,0].astype(np.float64))
    pred=fitted.predict(scaler.transform(features[test]).astype(np.float64))
    np.savez_compressed(output/"weights.npz",coef=fitted.coef_,intercept=fitted.intercept_,mean=scaler.mean_,scale=scaler.scale_)
    return pred,{"alpha":1.,"solver":"cholesky","input_features":features.shape[1],"sklearn":sklearn.__version__}

def fit_stid(root,x,y,train_idx,val_idx,test_idx,seed,device_arg,output):
    import torch
    torch.set_num_threads(4)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_arg=="auto" else device_arg)
    model=stid(root).to(device)
    mean=float(x[train_idx].mean());std=float(x[train_idx].std())+1e-6
    inputs=torch.from_numpy(((x-mean)/std)[:,:,:,None]).to(device)
    targets=torch.from_numpy(((y-mean)/std)[:,:,:,None]).to(device)
    train=torch.from_numpy(train_idx).to(device);val=torch.from_numpy(val_idx).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best=float("inf");best_epoch=None;log=[]
    for epoch in range(30):
        model.train();order=train[torch.randperm(len(train),device=device)]
        for batch,start in enumerate(range(0,len(train),256)):
            take=order[start:start+256]
            pred=model(inputs[take],None,batch_seen=batch,epoch=epoch,train=True)
            loss=torch.abs(pred-targets[take]).mean()
            assert torch.isfinite(loss).item(),"Nonfinite loss"
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step()
        model.eval();total=0.
        with torch.no_grad():
            for start in range(0,len(val),512):
                take=val[start:start+512]
                pred=model(inputs[take],None,batch_seen=0,epoch=epoch,train=False)[:,:,0,0]*std+mean
                truth=targets[take,:,0,0]*std+mean
                total+=torch.abs(pred.clamp_min(0)-truth).sum().item()
        metric=total/(len(val)*12)
        assert np.isfinite(metric)
        log.append({"epoch":epoch+1,"val_center_mae":metric})
        if metric<best:
            best=metric;best_epoch=epoch+1
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output/"best_weights.pt")
        dump(output/"epoch_log.json",log)
        if (epoch+1)%10==0:print(json.dumps({"epoch":epoch+1,"validation_mae":metric}),flush=True)
    model.load_state_dict(torch.load(output/"best_weights.pt",map_location=device,weights_only=True));model.eval()
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(test_idx),512):
            pred=model(inputs[test_idx[start:start+512]],None,batch_seen=0,epoch=best_epoch,train=False)[:,:,0,0]*std+mean
            predictions.append(pred.cpu().numpy())
    dump(output/"scaler.json",{"mean":mean,"std":std})
    return np.concatenate(predictions),{"epochs":30,"selected_epoch":best_epoch,"val_center_mae":best,"batch_size":256,
            "learning_rate":.001,"train_loss":"nine-slot L1","parameters":sum(p.numel() for p in model.parameters()),
            "torch":torch.__version__,"device":str(device),"device_name":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--variant",choices=["packaged","working"],required=True)
    parser.add_argument("--model",choices=["RidgeCenter","RidgeLocal9","STID_NoCalendar"],required=True)
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.artifact.resolve();output=args.output.resolve()
    assert output!=root and root not in output.parents,"Keep new training outputs outside the frozen artifact"
    output.mkdir(parents=True,exist_ok=False)
    state={"status":"started","pid":os.getpid(),"variant":args.variant,"model":args.model,"seed":args.seed if args.model=="STID_NoCalendar" else None,
           "started_utc":datetime.now(timezone.utc).isoformat(),"training_script_sha256":digest(Path(__file__)),"fits_started":1}
    dump(output/"state.json",state);t0=time.time()
    try:
        x,y,train,val,test=data(root,args.variant)
        if args.model=="STID_NoCalendar":pred,extra=fit_stid(root,x,y,train,val,test,args.seed,args.device,output)
        else:pred,extra=fit_ridge(args.model,x,y,train,test,output)
        prediction=np.maximum(pred,0).astype(np.float32)
        assert np.isfinite(prediction).all()
        np.savez_compressed(output/"predictions.npz",prediction=prediction,test_indices=test)
        dump(output/"training_result.json",{**state,**extra,"elapsed_seconds":time.time()-t0,"test_targets_excluded_from_fit":True,
             "train_samples":len(train),"validation_samples":len(val),"test_samples":len(test),"prediction_sha256":digest(output/"predictions.npz")})
        state["status"]="completed"
    except BaseException as e:
        state["status"]="failed";state["error"]=repr(e);raise
    finally:
        state["finished_utc"]=datetime.now(timezone.utc).isoformat();dump(output/"state.json",state)
    print(json.dumps(state),flush=True)

if __name__=="__main__":main()
