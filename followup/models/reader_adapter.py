# -*- coding: utf-8 -*-
"""Historical arrays -> frozen Graph WaveNet center point forecasts."""
import json
from pathlib import Path
import numpy as np
import torch
from model_adapter import build_model, point_forecast

ROOT = Path(__file__).resolve().parent
_CACHE = {}

def predict(history, variant='working', seed=1, device='cpu'):
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3 or history.shape[1:] != (12, 9) or not np.isfinite(history).all():
        raise ValueError('Supply finite historical inputs [N,12,9]')
    if variant not in ('packaged', 'working') or seed not in (1, 2, 3):
        raise ValueError('Choose a delivered training variant and seed')
    key = (variant, seed, device)
    if key not in _CACHE:
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        directory = ROOT / 'runs' / (variant + '__GraphWaveNet_seed' + str(seed))
        scaler = json.loads((directory / 'scaler.json').read_text(encoding='utf-8'))
        model = build_model(device)
        model.load_state_dict(torch.load(directory / 'best_weights.pt', map_location=device, weights_only=True))
        model.eval()
        _CACHE[key] = model, scaler
    model, scaler = _CACHE[key]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(history), 256):
            data = (history[start:start + 256] - scaler['mean']) / scaler['std']
            tensor = torch.from_numpy(data).to(device)
            pred = (point_forecast(model, tensor)[:, :, 0] * scaler['std'] + scaler['mean']).clamp_min(0)
            chunks.append(pred.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)
