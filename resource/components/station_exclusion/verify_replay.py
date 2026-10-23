from pathlib import Path
import argparse,hashlib,json
import numpy as np
h=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=argparse.ArgumentParser()
a.add_argument("--out",required=True,type=Path)
args=a.parse_args();out=args.out.resolve()
manifest=read(h/"PACKAGE_MANIFEST.json")
for f in manifest["files"]:
    assert sha(h/f["path"])==f["sha256"],f["path"]
for base in [h/"expected",out]:
    for name,digest in read(base/"OUTPUT_SEAL.json")["files"].items():
        assert sha(base/name)==digest,name
arrays=0;elements=0
for name in ["STATION_TRAIN_MASKS.npz","REMOVAL_WITNESSES.npz","QUERY_MEMBERSHIP.npz"]:
    old=np.load(h/"expected"/name,allow_pickle=False)
    new=np.load(out/name,allow_pickle=False)
    assert set(old.files)==set(new.files)
    for key in old.files:
        assert old[key].dtype==new[key].dtype and np.array_equal(old[key],new[key]),(name,key)
        arrays+=1;elements+=int(old[key].size)
for name in ["RESULTS.json","PER_NODE.json"]:
    assert read(h/"expected"/name)==read(out/name),name
assert (h/"expected/PER_NODE.csv").read_bytes()==(out/"PER_NODE.csv").read_bytes()
start=read(out/"START.json")
for key in ["source_sha256","contract_sha256","protocol_sha256","code_sha256"]:
    assert start[key]==read(h/"expected/START.json")[key],key
v=dict(passed=True,array_fields=arrays,array_elements=elements,
    result_and_per_node_json_exact=True,csv_bytes_exact=True,package_integrity_passed=True,
    output_seals_verified=True,new_fits=0,new_predictions=0,
    scope="Extracted same-machine replay; no fresh dependency environment")
(out/"REPLAY_VERIFICATION.json").write_text(json.dumps(v,indent=2)+"\n",encoding="utf-8")
print(json.dumps(v))
