"""Minimal example of the history-only user interface; no evaluator imports."""
import numpy as np


def predict(history):
    return np.repeat(history[:, -1:], 12, axis=1)
