"""Call a local predictor with only a (batch, 12) history array, then attach IDs.

Example: python predict.py --inputs task/inputs.npz --adapter last_value.py:predict
         --out last_value_predictions.npz
The callable returns (batch, 12) real finite point forecasts. It receives no IDs.
This separation is an interface contract, not a security sandbox for arbitrary code.
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as archive:
        if set(archive.files) != {'history', 'sample_id'}:
            raise ValueError('Input archive must contain history and sample_id only')
        history, ids = archive['history'], archive['sample_id']
    if ids.ndim != 1 or ids.dtype.kind not in 'US' or len(set(ids.tolist())) != len(ids):
        raise ValueError('Invalid or duplicate input IDs')
    if any(not key.strip() for key in ids.tolist()):
        raise ValueError('Empty input ID')
    if history.shape != (len(ids), 12) or history.dtype.kind not in 'fiu' or not np.isfinite(history).all():
        raise ValueError('Expected finite real histories of shape (N, 12)')
    # rsplit preserves the colon in an absolute Windows path.
    filename, function = args.adapter.rsplit(':', 1)
    specification = importlib.util.spec_from_file_location('netv4_user_predictor', filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    prediction = np.asarray(getattr(module, function)(history.copy()))
    if prediction.shape != (len(ids), 12) or prediction.dtype.kind not in 'fiu' or not np.isfinite(prediction).all():
        raise ValueError('Predictor must return finite real point forecasts of shape (N, 12)')
    with args.out.open('xb') as stream:
        np.savez_compressed(stream, sample_id=ids, prediction=prediction)
    print('Predicted %d histories with no target or metadata argument' % len(ids))


if __name__ == '__main__':
    main()
