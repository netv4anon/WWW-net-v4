# -*- coding: utf-8 -*-
"""Replay one delivered neural predictor without evaluator files."""
import argparse
import numpy as np
from pathlib import Path
from reader_adapter import predict

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['packaged', 'working'], default='working')
    p.add_argument('--seed', choices=[1, 2, 3], type=int, default=1)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    assert not args.out.exists(), 'Choose a new output file'
    with np.load(args.inputs, allow_pickle=False) as z:
        assert set(z.files) == {'sample_id', 'history'}
        ids, history = z['sample_id'], z['history']
    result = predict(history, args.variant, args.seed, args.device)
    assert result.shape == (len(ids), 12)
    np.savez_compressed(args.out, sample_id=ids, prediction=result)
    print(str(args.out))

if __name__ == '__main__': main()
