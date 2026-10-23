"""Exact per-station input exclusion; no forecasts, targets, or training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse, collections, csv, hashlib, json, mmap, time
import numpy as np

HERE=Path(__file__).resolve().parent
CONDITIONS=["Original","TargetSupport"]
KEYS=["flow","flow_history_calendar"]
CONFIGS=[c+"__"+k for c in CONDITIONS for k in KEYS]
N=5585;NT=3909;Q0=4746;Q=839;H=12;J=496
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):
    s=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(4194304),b""):s.update(c)
    return s.hexdigest()
def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def now():return datetime.now(timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    source=args.source.resolve();out=args.out.resolve()
    assert not out.exists(),"Use a new output directory."
    binding=read(HERE/"SOURCE.json")
    assert source.stat().st_size==binding["source_bytes"]
    assert sha(source)==binding["source_sha256"]
    assert sha(HERE/"contract.npz")==binding["contract_sha256"]
    contract=np.load(HERE/"contract.npz",allow_pickle=False)
    views={name:contract[name] for name in CONDITIONS}
    assert np.array_equal(views["Original"],np.arange(NT))
    assert len(views["TargetSupport"])==2665
    out.mkdir(parents=True)
    write(out/"START.json",dict(created_utc=now(),source_sha256=binding["source_sha256"],
        code_sha256=sha(Path(__file__).resolve()),protocol_sha256=sha(HERE/"PROTOCOL.md"),
        contract_sha256=binding["contract_sha256"],configurations=CONFIGS,
        new_fits=0,new_predictions=0,new_source_records=0,
        query_targets_parsed=False,inherited_targetsupport_uses_query_targets=True,
        raw_source_hashing_does_not_mean_semantic_target_access=True))
    X=np.empty((N,H,J),dtype="<f4")
    cal=np.empty((N,H,2),dtype="<f4")
    with source.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for i,offset in enumerate(contract["x_offset"]):
            raw=mm[int(offset):int(offset)+71424]
            assert hashlib.sha256(raw).hexdigest()==str(contract["x_sha256"][i])
            a=np.frombuffer(raw,dtype="<f4").reshape(H,J,3)
            assert np.array_equal(a[:,:,1:],np.broadcast_to(a[:,0:1,1:],(H,J,2)))
            X[i]=a[:,:,0];cal[i]=a[:,0,1:]
    assert np.isfinite(X).all() and np.isfinite(cal).all()
    print("Decoded and verified historical arrays only:",N,flush=True)
    variation=np.ptp(X,axis=1)>0
    last=X[:,-1,:].astype(np.float64)*float(contract["source_std"])+float(contract["source_mean"])
    zero=(~variation)&(np.abs(last)<=.001)
    other=(~variation)&(~zero)
    keep=np.zeros((4,NT,J),bool)
    witnesses=np.full((4,NT,J),-2,np.int32)
    before=np.zeros((4,Q,J),np.int32)
    after=np.zeros_like(before)
    flow_after=np.zeros_like(before)
    pernode=[]
    started=time.perf_counter()
    for j in range(J):
        flow=[v.tobytes() for v in np.ascontiguousarray(X[:,:,j])]
        full=[f+c.tobytes() for f,c in zip(flow,cal)]
        for k,name in enumerate(CONFIGS):
            condition,key=name.split("__")
            ids=views[condition]
            keys=flow if key=="flow" else full
            qkeys=keys[Q0:]
            first={}
            for qi,qkey in enumerate(qkeys):first.setdefault(qkey,qi+Q0)
            rejected=np.array([keys[i] in first for i in ids],bool)
            kept=ids[~rejected];removed=ids[rejected]
            keep[k,kept,j]=True
            witnesses[k,kept,j]=-1
            witnesses[k,removed,j]=[first[keys[i]] for i in removed]
            tc=collections.Counter(keys[i] for i in ids)
            rc=collections.Counter(keys[i] for i in kept)
            fc=collections.Counter(flow[i] for i in kept)
            before[k,:,j]=[tc[qkey] for qkey in qkeys]
            after[k,:,j]=[rc[qkey] for qkey in qkeys]
            flow_after[k,:,j]=[fc[qkey] for qkey in flow[Q0:]]
            assert not after[k,:,j].any()
            row=dict(configuration=name,node_position=j,candidate_rows=len(ids),
                retained_rows=len(kept),removed_rows=len(removed),
                retained_fraction=len(kept)/len(ids),
                candidate_distinct_input_keys=len(tc),retained_distinct_input_keys=len(rc),
                retained_distinct_flow_histories=len(fc),
                query_distinct_input_keys=len(set(qkeys)),query_distinct_flow_histories=len(set(flow[Q0:])),
                query_hit_cells_before=int(np.count_nonzero(before[k,:,j])),
                query_hit_cells_after=int(np.count_nonzero(after[k,:,j])),
                flow_only_query_hits_after=int(np.count_nonzero(flow_after[k,:,j])))
            for label,mask in [("varying",variation),("constant_near_zero",zero),("constant_other",other)]:
                row["candidate_"+label+"_histories"]=int(mask[ids,j].sum())
                row["retained_"+label+"_histories"]=int(mask[kept,j].sum())
                row["query_"+label+"_histories"]=int(mask[Q0:,j].sum())
            pernode.append(row)
        if (j+1)%100==0:print("Audited nodes",j+1,"elapsed",round(time.perf_counter()-started,1),flush=True)
    common=keep.all(axis=2)
    np.savez_compressed(out/"STATION_TRAIN_MASKS.npz",keep=keep,configs=np.array(CONFIGS),
        common_rows=common,original_training_rows=np.arange(NT))
    np.savez_compressed(out/"REMOVAL_WITNESSES.npz",query_row=witnesses,configs=np.array(CONFIGS))
    np.savez_compressed(out/"QUERY_MEMBERSHIP.npz",before=before,after=after,
        flow_after=flow_after,configs=np.array(CONFIGS))
    with (out/"PER_NODE.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(pernode[0]))
        writer.writeheader();writer.writerows(pernode)
    write(out/"PER_NODE.json",pernode)
    summary={}
    for k,name in enumerate(CONFIGS):
        records=[d for d in pernode if d["configuration"]==name]
        retained=np.array([d["retained_rows"] for d in records])
        uniq=np.array([d["retained_distinct_flow_histories"] for d in records])
        empty=retained==0
        qvar=np.array([d["query_varying_histories"] for d in records])
        qzero=np.array([d["query_constant_near_zero_histories"] for d in records])
        rvar=np.array([d["retained_varying_histories"] for d in records])
        bins={"zero":int(empty.sum()),"1_to_12":int(((retained>=1)&(retained<=12)).sum()),
            "13_to_99":int(((retained>=13)&(retained<=99)).sum()),
            "100_to_499":int(((retained>=100)&(retained<=499)).sum()),
            "at_least_500":int((retained>=500).sum())}
        summary[name]=dict(candidate_rows=records[0]["candidate_rows"],
            retained_station_training_cells=int(retained.sum()),
            retained_rows_min=int(retained.min()),retained_rows_max=int(retained.max()),
            retained_rows_quantiles=dict(zip(["p10","p25","p50","p75","p90"],
                np.quantile(retained,[.1,.25,.5,.75,.9]).tolist())),
            retained_distinct_flow_histories_min=int(uniq.min()),
            retained_distinct_flow_histories_median=float(np.median(uniq)),
            retained_distinct_flow_histories_max=int(uniq.max()),
            retention_bins=bins,empty_node_positions=np.flatnonzero(empty).tolist(),
            empty_nodes_with_any_varying_query_history=int((empty&(qvar>0)).sum()),
            empty_nodes_all_queries_constant_near_zero=int((empty&(qzero==Q)).sum()),
            nonempty_nodes_with_only_constant_training_histories=int(((retained>0)&(rvar==0)).sum()),
            all_node_views_nonempty=bool(np.all(retained>0)),
            query_key_hits_before=int(np.count_nonzero(before[k])),
            query_key_hits_after=int(np.count_nonzero(after[k])),
            residual_flow_only_query_hits=int(np.count_nonzero(flow_after[k])),
            residual_flow_only_affected_nodes=int(np.any(flow_after[k]>0,axis=0).sum()),
            common_rows_retained_for_all_nodes=int(common[k].sum()),
            rows_retained_for_any_node=int(keep[k].any(axis=1).sum()),
            all_query_nodes_preserved=True)
        print(json.dumps({"configuration":name,**summary[name]},ensure_ascii=False),flush=True)
    write(out/"RESULTS.json",dict(status="primary_complete_pending_independent_verification",
        configurations=CONFIGS,source_rows=N,original_train_rows=NT,test_rows=Q,node_positions=J,
        summary=summary,query_histories_with_variation=int(variation[Q0:].sum()),
        query_histories_constant_near_zero=int(zero[Q0:].sum()),
        query_histories_constant_other=int(other[Q0:].sum()),
        quantile_method="NumPy default linear interpolation",
        retention_bins_are_not_trainability_thresholds=True,
        query_targets_parsed=False,new_rule_uses_query_targets=False,
        inherited_targetsupport_uses_query_targets=True,
        source_identity_recovered=False,chronological_repair=False,
        new_fits=0,new_predictions=0,new_scores=0,new_raw_records=0,
        new_mask_configurations=4,per_station_training_views=4*J))
    write(out/"OUTPUT_SEAL.json",{
        "created_utc":now(),"files":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})

if __name__=="__main__":main()
