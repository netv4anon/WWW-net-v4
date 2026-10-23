"""Run a reader's multisequence point predictor using only shuffled history inputs."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--adapter',required=True,help='Python file:function');parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    with np.load(args.inputs,allow_pickle=False) as z:
        assert set(z.files)=={'sample_id','history'}
        ids=z['sample_id'];history=z['history']
    assert history.ndim==3 and history.shape[1:]==(12,9) and np.isfinite(history).all()
    file,name=args.adapter.rsplit(':',1)
    spec=importlib.util.spec_from_file_location('reader_adapter',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=np.asarray(getattr(module,name)(history))
    assert p.shape==(len(ids),12) and p.dtype.kind in 'fiu' and np.isfinite(p).all()
    np.savez_compressed(args.out,sample_id=ids,prediction=p)

if __name__=='__main__':main()
