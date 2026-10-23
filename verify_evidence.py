"""Recompute saved losses and optionally raw-source matching in a new directory."""
import argparse,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,help='Optional exact upstream Contra_Costa incident_all.npy')
ap.add_argument('--out',type=Path,required=True,help='New output directory outside the immutable resource')
args=ap.parse_args();out=args.out.resolve()
assert ROOT not in out.parents and out!=ROOT,'Use an output directory outside this artifact'
out.mkdir(parents=True,exist_ok=False)
p=ROOT/'verification/verify_existing_evidence_historical.py'
spec=importlib.util.spec_from_file_location('evidence_checks',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.RESOURCE=ROOT/'resource';m.OUT=out
if args.source:
    # The inherited check resolves one fixed suffix; use a direct source resolver.
    class SourceRoot:
        def __truediv__(self,_):return args.source.resolve()
    m.PROGRAM=SourceRoot();m.verify_reuse()
m.verify_losses()
try:
    import torch
except ImportError:
    configuration='not rerun: PyTorch is optional; see supplied configuration/weight/log checks'
else:
    m.verify_configuration();configuration='PASS'
print(json.dumps({'status':'PASS','saved_loss_recomputation':'PASS','raw_source_matches':'PASS' if args.source else 'not run: upstream source must be supplied','configuration':configuration,'new_fits':0,'new_inference':0}))
