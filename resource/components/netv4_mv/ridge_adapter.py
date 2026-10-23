"""Example coupled reader predictor; put its frozen weights beside this file."""
from pathlib import Path
import numpy as np

def predict(history):
    with np.load(Path(__file__).with_name('weights.npz'),allow_pickle=False) as z:
        x=history.reshape(len(history),-1).astype(np.float32).copy()
        x-=z['mean'];x/=z['scale']
        return np.maximum(x.astype(np.float64)@z['coef'].T+z['intercept'],0).astype(np.float32)
