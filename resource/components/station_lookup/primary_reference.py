from pathlib import Path
from datetime import datetime, timezone
import collections, hashlib, json, mmap, time
import numpy as np

HERE=Path(__file__).resolve().parent
B=HERE.parent
OLD=B/"target_support_repair_2026-10-08"
A=B/"incident_split_feasibility_2026-10-03"
SCOPE=B/"support_scope_audit_2026-10-08"
CONDITIONS=["Original","TargetSupport"]
KEYS=["flow","flow_history_calendar"]
CONFIGS=[c+"__"+k for c in CONDITIONS for k in KEYS]
NT=3909; Q0=4746; NQ=839; NN=496; H=12

def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):
    s=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4194304),b""):s.update(chunk)
    return s.hexdigest()
def dump(n,o):
    (HERE/n).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def now():return datetime.now(timezone.utc).isoformat()

def predict_station(train_x,train_y,query_x,train_calendar,query_calendar,rows,use_calendar):
    # Query futures are intentionally absent.
    index=collections.defaultdict(list)
    for r in rows:
        key=train_x[r].tobytes()
        if use_calendar:key+=train_calendar[r].tobytes()
        index[key].append(int(r))
    keys=[]
    for q in range(len(query_x)):
        key=query_x[q].tobytes()
        if use_calendar:key+=query_calendar[q].tobytes()
        keys.append(key)
    pred=np.repeat(query_x[:,-1,None].astype(np.float64),H,axis=1)
    count=np.zeros(len(query_x),np.int32)
    unique=np.zeros(len(query_x),np.int32)
    representative=np.full(len(query_x),-1,np.int32)
    cache={}
    for q,key in enumerate(keys):
        ids=index.get(key)
        if ids is None:continue
        if key not in cache:
            future=train_y[ids]
            cache[key]=(future.mean(axis=0,dtype=np.float64),len({x.tobytes() for x in future}))
        pred[q],unique[q]=cache[key]
        count[q]=len(ids);representative[q]=ids[0]
    return pred,count,unique,representative

def main():
    assert not (HERE/"START.json").exists(),"Do not overwrite a completed or active run."
    expected=read(OLD/"PREPARED_INPUT_BINDINGS.json")
    bound={}
    for filename in ["HISTORY_FLOW.npy","TARGET_FLOW.npy","TRAIN_ROWS.npz"]:
        p=OLD/filename
        assert sha(p)==expected[filename],filename
        bound[p.relative_to(B).as_posix()]=expected[filename]
    source=A/"sources/contra_complete.npy"
    assert sha(source)=="bc6c5608457f88864625d683f9a66d17bc0fc31fe5acba366b6033708a099279"
    for p in [source,A/"ARRAY_OFFSETS.json",A/"sources/contra_stats.npz",
              SCOPE/"HIT_MAPS.npz",SCOPE/"VERIFICATION.json",HERE/"PROTOCOL.md",Path(__file__).resolve()]:
        bound[p.relative_to(B).as_posix()]=sha(p)
    dump("START.json",dict(created_utc=now(),bindings=bound,configurations=CONFIGS,
        new_prediction_configurations=4,new_fits=0,prior_target_exposure=True,
        inherited_mask_uses_test_targets=True,query_targets_used_by_prediction_function=False))
    print("Fixed protocol/source/code bindings saved.",flush=True)
    X=np.load(OLD/"HISTORY_FLOW.npy",allow_pickle=False)
    train_y=np.array(np.load(OLD/"TARGET_FLOW.npy",allow_pickle=False,mmap_mode="r")[:NT])
    rowsfile=np.load(OLD/"TRAIN_ROWS.npz",allow_pickle=False)
    rows={c:rowsfile[c] for c in CONDITIONS}
    assert len(rows["Original"])==3909 and len(rows["TargetSupport"])==2665
    offsets={(r["row"],r["role"]):r for r in read(A/"ARRAY_OFFSETS.json")}
    cal=np.empty((len(X),12,2),dtype="<f4")
    with source.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for i in range(len(X)):
            r=offsets[i,"x_data"]
            a=np.frombuffer(mm,dtype="<f4",count=12*496*3,offset=r["offset"]).reshape(12,496,3)
            cal[i]=a[:,0,1:]
            del a
    np.save(HERE/"HISTORY_CALENDAR.npy",cal,allow_pickle=False)
    stats=np.load(A/"sources/contra_stats.npz",allow_pickle=False)
    mean,std=float(stats["mean"]),float(stats["std"])
    count=np.empty((4,NQ,NN),np.int32)
    unique=np.empty_like(count);rep=np.empty_like(count)
    folder=HERE/"predictions";folder.mkdir()
    preds={name:np.lib.format.open_memmap(folder/(name+".npy"),mode="w+",dtype="<f8",shape=(NQ,H,NN))
           for name in CONFIGS}
    started=time.perf_counter()
    for j in range(NN):
        tx=np.ascontiguousarray(X[:NT,:,j])
        ty=np.ascontiguousarray(train_y[:,:,j])
        qx=np.ascontiguousarray(X[Q0:,:,j])
        for k,name in enumerate(CONFIGS):
            condition,key=name.split("__")
            p,count[k,:,j],unique[k,:,j],rep[k,:,j]=predict_station(
                tx,ty,qx,cal[:NT],cal[Q0:],rows[condition],key!="flow")
            preds[name][:,:,j]=p*std+mean
        if (j+1)%100==0:print("Predicted nodes",j+1,"elapsed",round(time.perf_counter()-started,1),flush=True)
    for p in preds.values():p.flush()
    del preds
    np.savez_compressed(HERE/"MATCHES.npz",matching_train_rows=count,
        distinct_train_futures=unique,representative_train_row=rep,configurations=np.array(CONFIGS))
    seal=dict(created_utc=now(),predictions={name:sha(folder/(name+".npy")) for name in CONFIGS},
        matches_sha256=sha(HERE/"MATCHES.npz"),history_calendar_sha256=sha(HERE/"HISTORY_CALENDAR.npy"),
        query_target_values_used_before_prediction_seal=False,
        source_file_bytes_hashed_before_prediction=True,
        prior_query_target_exposure=True,inherited_mask_uses_test_targets=True)
    dump("PREDICTION_SEAL.json",seal)
    print("All four prediction arrays sealed; query-target scoring starts next.",flush=True)
    Y=np.load(OLD/"TARGET_FLOW.npy",allow_pickle=False,mmap_mode="r")
    target=Y[Q0:].astype(np.float64)*std+mean
    baseline=np.repeat(X[Q0:,-1:, :].astype(np.float64)*std+mean,H,axis=1)
    base_mae=np.abs(baseline-target).mean(axis=1)
    base_mse=((baseline-target)**2).mean(axis=1)
    varying=np.ptp(Y[Q0:],axis=1)>0
    const_zero=(~varying)&(np.abs(target[:,0,:])<=.001)
    const_other=(~varying)&(~const_zero)
    assert int(varying.sum())==341680
    saved=np.load(SCOPE/"HIT_MAPS.npz",allow_pickle=False)
    residual=saved["calendar_paired"][2]&saved["varying_target"]
    assert int(residual.sum())==30685
    scores={};details={};metrics={}
    for k,name in enumerate(CONFIGS):
        p=np.load(folder/(name+".npy"),allow_pickle=False,mmap_mode="r")
        loss=np.abs(p-target).mean(axis=1)
        mse=((p-target)**2).mean(axis=1)
        hits=count[k]>0
        single=unique[k]==1
        recovered=np.zeros((NQ,NN),bool)
        for j in range(NN):
            take=np.flatnonzero(single[:,j])
            if len(take):
                a=np.ascontiguousarray(train_y[rep[k,take,j],:,j]).view(np.uint8).reshape(len(take),-1)
                c=np.ascontiguousarray(Y[Q0+take,:,j]).view(np.uint8).reshape(len(take),-1)
                recovered[take,j]=np.all(a==c,axis=1)
        groups={"all":np.ones_like(hits),"varying_target":varying,
            "constant_near_zero":const_zero,"constant_other":const_other,
            "history_hit":hits,"history_miss":~hits,
            "varying_history_hit":hits&varying,
            "varying_unique_history_hit":hits&single&varying,
            "varying_conflicting_history_hit":hits&(~single)&varying,
            "prior_residual_varying_calendar_pair":residual,
            "residual_complement":~residual}
        scores[name]={}
        for group,mask in groups.items():
            n=int(mask.sum())
            s={"cells":n,"nodes":int(mask.any(0).sum()),"queries":int(mask.any(1).sum())}
            if n:
                bmae=float(base_mae[mask].mean());pmae=float(loss[mask].mean())
                s.update(lastvalue_mae=bmae,lookup_mae=pmae,mae_gain=bmae-pmae,
                    mae_gain_percent=100*(bmae-pmae)/bmae if bmae else None,
                    lastvalue_rmse=float(np.sqrt(base_mse[mask].mean())),
                    lookup_rmse=float(np.sqrt(mse[mask].mean())))
            scores[name][group]=s
        node_gain=np.full(NN,np.nan)
        eligible=(hits&varying).any(0)
        for j in range(NN):
            if varying[:,j].any():node_gain[j]=float((base_mae[:,j]-loss[:,j])[varying[:,j]].mean())
        positive=int(np.sum(eligible&(node_gain>1e-10)))
        negative=int(np.sum(eligible&(node_gain < -1e-10)))
        tied=int(eligible.sum())-positive-negative
        details[name]=dict(history_hit_cells=int(hits.sum()),
            unique_training_future_hit_cells=int((hits&single).sum()),
            conflicting_training_future_hit_cells=int((hits&(~single)).sum()),
            raw_exact_unique_future_recovered_cells=int(recovered.sum()),
            varying_raw_exact_unique_future_recovered_cells=int((recovered&varying).sum()),
            prior_residual_witnesses_uniquely_recovered=int((recovered&residual).sum()),
            prior_residual_witnesses_conflicting_train_future=int((residual&hits&(~single)).sum()),
            eligible_stations_for_varying_gain=int(eligible.sum()),
            positive_stations=positive,negative_stations=negative,tied_stations=tied,
            misses_identical_to_lastvalue=bool(np.array_equal(p.transpose(0,2,1)[~hits],baseline.transpose(0,2,1)[~hits])))
        assert details[name]["misses_identical_to_lastvalue"]
        metrics[name+"_mae"]=loss;metrics[name+"_mse"]=mse
        metrics[name+"_raw_unique_recovery"]=recovered
        metrics[name+"_varying_station_gain"]=node_gain
        print(json.dumps({"configuration":name,"all":scores[name]["all"],"varying":scores[name]["varying_target"],
            "diagnostics":details[name]},ensure_ascii=False),flush=True)
    np.savez_compressed(HERE/"CELL_METRICS.npz",lastvalue_mae=base_mae,lastvalue_mse=base_mse,
        varying_target=varying,constant_near_zero=const_zero,prior_residual=residual,**metrics)
    primary="TargetSupport__flow"
    gate={
      "all_mae_improved":scores[primary]["all"]["mae_gain"]>1e-10,
      "varying_mae_improved":scores[primary]["varying_target"]["mae_gain"]>1e-10,
      "majority_eligible_stations_improved":details[primary]["positive_stations"]>details[primary]["eligible_stations_for_varying_gain"]/2}
    dump("RESULTS.json",dict(status="primary_complete_pending_independent_verification",
        configurations=CONFIGS,training_rows={c:len(rows[c]) for c in CONDITIONS},
        query_rows=NQ,node_positions=NN,all_cells=NQ*NN,
        score_population="All fixed 12-step query/node cells; same-position matching only",
        scores=scores,diagnostics=details,primary_configuration=primary,
        primary_gate=gate,primary_gate_passed=all(gate.values()),
        new_fits=0,new_prediction_configurations=4,new_source_records=0,
        chronological_repair=False,original_model_inflation_established=False,
        new_algorithm_claim=False,development_exposed=True,
        inherited_training_mask_uses_query_targets=True))
    print("Primary gate:",gate,flush=True)

if __name__=="__main__":main()

