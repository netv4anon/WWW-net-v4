"""Replay four fixed local-history diagnostics from the exact external source."""
import argparse, hashlib, json, mmap
from pathlib import Path
import numpy as np
from primary_reference import predict_station, CONFIGS

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(4194304),b""):h.update(c)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    src=args.source.resolve();out=args.out.resolve()
    if out.exists():raise ValueError("Use a new output directory.")
    binding=read(HERE/"SOURCE.json")
    if src.stat().st_size!=binding["source_bytes"] or sha(src)!=binding["source_sha256"]:
        raise ValueError("Source bytes do not match the pinned Contra_Costa v1 release.")
    for name,key in [("contract.npz","contract_sha256"),("stats.npz","stats_sha256")]:
        if sha(HERE/name)!=binding[key]:raise ValueError("Contract binding failed: "+name)
    contract=np.load(HERE/"contract.npz",allow_pickle=False)
    stats=np.load(HERE/"stats.npz",allow_pickle=False)
    mean,std=float(stats["mean"]),float(stats["std"])
    out.mkdir(parents=True)
    x=np.empty((5585,12,496),dtype="<f4")
    ytrain=np.empty((3909,12,496),dtype="<f4")
    calendar=np.empty((5585,12,2),dtype="<f4")
    with src.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for i in range(5585):
            a=np.frombuffer(mm,dtype="<f4",count=17856,offset=int(contract["x_offset"][i])).reshape(12,496,3)
            x[i]=a[:,:,0];calendar[i]=a[:,0,1:]
            if i<3909:
                a=np.frombuffer(mm,dtype="<f4",count=17856,offset=int(contract["y_offset"][i])).reshape(12,496,3)
                ytrain[i]=a[:,:,0]
            del a
    counts=np.empty((4,839,496),np.int32)
    unique=np.empty_like(counts);rep=np.empty_like(counts)
    allpred={name:np.lib.format.open_memmap(out/(name+".npy"),mode="w+",dtype="<f8",shape=(839,12,496)) for name in CONFIGS}
    for j in range(496):
        for k,name in enumerate(CONFIGS):
            condition,key=name.split("__")
            p,counts[k,:,j],unique[k,:,j],rep[k,:,j]=predict_station(
                np.ascontiguousarray(x[:3909,:,j]),np.ascontiguousarray(ytrain[:,:,j]),
                np.ascontiguousarray(x[4746:,:,j]),calendar[:3909],calendar[4746:],
                contract[condition],key!="flow")
            allpred[name][:,:,j]=p*std+mean
    for p in allpred.values():p.flush()
    del allpred
    predhash={}
    for name in CONFIGS:
        p=np.load(out/(name+".npy"),allow_pickle=False,mmap_mode="r")
        predhash[name]=hashlib.sha256(p.tobytes()).hexdigest()
        if predhash[name]!=binding["expected_prediction_data_sha256"][name]:
            raise AssertionError("Prediction mismatch: "+name)
    write(out/"PREDICTION_SEAL.json",dict(prediction_data_sha256=predhash,
        query_future_values_used_in_lookup=False,
        inherited_training_mask_uses_query_targets=True,prior_development_exposure=True))
    qy=np.empty((839,12,496),dtype="<f4")
    with src.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for i in range(839):
            a=np.frombuffer(mm,dtype="<f4",count=17856,offset=int(contract["y_offset"][4746+i])).reshape(12,496,3)
            qy[i]=a[:,:,0]
            del a
    target=qy.astype(float)*std+mean
    base=np.repeat(x[4746:,-1:,:].astype(float)*std+mean,12,axis=1)
    bmae=np.abs(base-target).mean(1)
    bmse=((base-target)**2).mean(1)
    varying=np.ptp(qy,axis=1)>0
    zero=(~varying)&(np.abs(target[:,0,:])<=.001)
    residual=contract["prior_residual"]
    expected=read(HERE/"EXPECTED_RESULTS.json")
    scores={};maxerr=0.;tested=0
    for k,name in enumerate(CONFIGS):
        p=np.load(out/(name+".npy"),allow_pickle=False,mmap_mode="r")
        loss=np.abs(p-target).mean(1);mse=((p-target)**2).mean(1)
        hit=counts[k]>0;one=unique[k]==1
        groups={"all":np.ones_like(hit),"varying_target":varying,
            "constant_near_zero":zero,"constant_other":(~varying)&(~zero),
            "history_hit":hit,"history_miss":~hit,"varying_history_hit":varying&hit,
            "varying_unique_history_hit":varying&hit&one,
            "varying_conflicting_history_hit":varying&hit&(~one),
            "prior_residual_varying_calendar_pair":residual,"residual_complement":~residual}
        scores[name]={}
        for group,m in groups.items():
            s=dict(cells=int(m.sum()),nodes=int(m.any(0).sum()),queries=int(m.any(1).sum()))
            if m.any():
                bm=float(bmae[m].mean());pm=float(loss[m].mean())
                s.update(lastvalue_mae=bm,lookup_mae=pm,mae_gain=bm-pm,
                    mae_gain_percent=100*(bm-pm)/bm if bm else None,
                    lastvalue_rmse=float(np.sqrt(bmse[m].mean())),lookup_rmse=float(np.sqrt(mse[m].mean())))
            ref=expected["scores"][name][group]
            assert s.keys()==ref.keys()
            for key,val in s.items():
                if val is None:assert ref[key] is None
                else:
                    err=abs(val-ref[key]);maxerr=max(maxerr,err);tested+=1
                    assert err<1e-9,(name,group,key,err)
            scores[name][group]=s
    np.savez_compressed(out/"MATCHES.npz",matching_train_rows=counts,
        distinct_train_futures=unique,representative_train_row=rep,configurations=np.array(CONFIGS))
    write(out/"SCORES.json",scores)
    write(out/"REPLAY_VERIFICATION.json",dict(passed=True,configurations=4,
        prediction_values_verified=4*839*12*496,aggregate_values_verified=tested,
        max_abs_metric_error=maxerr,source_sha256=binding["source_sha256"],
        semantic_prediction_hashes_all_match=True,fresh_environment_installation_verified=False))
    print(json.dumps(dict(passed=True,configurations=4,
        primary=scores["TargetSupport__flow"]["all"],max_abs_metric_error=maxerr)),flush=True)

if __name__=="__main__":main()
