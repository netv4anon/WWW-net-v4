# -*- coding: utf-8 -*-
"""Fit the fixed six configurations and seal all forecasts before MV scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from model_adapter import build_model, point_forecast

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def stamp():
    return datetime.now(timezone.utc).isoformat()

def load_training(variant):
    directory = ROOT / 'data' / variant
    with np.load(directory / 'history.npz', allow_pickle=False) as z:
        x = z['history'].astype(np.float32)
    with np.load(directory / 'splits.npz', allow_pickle=False) as z:
        train, val, test = z['train'], z['validation'], z['test']
    with np.load(directory / 'targets.npz', allow_pickle=False) as z:
        raw = z['future']
        y = np.full(raw.shape, np.nan, dtype=np.float32)
        y[train] = raw[train]
        y[val] = raw[val]
    assert x.shape == y.shape == (33076, 12, 9)
    assert np.isnan(y[test]).all()
    assert len(np.unique(np.concatenate([train, val, test]))) == len(x)
    mean, std = float(x[train].mean()), float(x[train].std()) + 1e-6
    inputs = torch.from_numpy((x - mean) / std).to(DEVICE)
    targets = torch.from_numpy((y - mean) / std).to(DEVICE)
    return inputs, targets, train, val, test, mean, std

def main():
    pins = json.loads((ROOT / 'PINS.json').read_text(encoding='utf-8'))
    for row in pins['files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    assert not (ROOT / 'FORECAST_SEAL.json').exists()
    with np.load(ROOT / 'task/inputs.npz', allow_pickle=False) as z:
        ids, mv_history = z['sample_id'], z['history']
    assert mv_history.shape == (68544, 12, 9)
    models = []
    began = time.time()
    ledger = json.loads((ROOT / 'SCIENTIFIC_LEDGER.json').read_text(encoding='utf-8'))
    ledger.update({'status': 'running', 'pid': os.getpid(), 'started_utc': stamp(),
        'training_runner_sha256': sha(Path(__file__)), 'model_adapter_sha256': sha(ROOT / 'model_adapter.py'),
        'torch': torch.__version__, 'numpy': np.__version__, 'device': str(DEVICE),
        'device_name': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})
    dump(ROOT / 'SCIENTIFIC_LEDGER.json', ledger)
    for variant in ('packaged', 'working'):
        inputs, targets, train, val, test, mean, std = load_training(variant)
        train_idx = torch.as_tensor(train, device=DEVICE)
        val_idx = torch.as_tensor(val, device=DEVICE)
        for seed in (1, 2, 3):
            name = variant + '__GraphWaveNet_seed' + str(seed)
            directory = ROOT / 'runs' / name
            directory.mkdir(parents=True, exist_ok=False)
            ledger['fit_attempts_started'] = ledger.get('fit_attempts_started', 0) + 1
            dump(ROOT / 'SCIENTIFIC_LEDGER.json', ledger)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = build_model(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            best, best_epoch, logs = float('inf'), None, []
            started = time.time()
            dump(directory / 'state.json', {'status': 'training', 'started_utc': stamp(), 'seed': seed, 'variant': variant})
            for epoch in range(30):
                model.train()
                order = train_idx[torch.randperm(len(train_idx), device=DEVICE)]
                total_train, seen_train = 0., 0
                for start in range(0, len(order), 256):
                    take = order[start:start + 256]
                    output = point_forecast(model, inputs[take])
                    loss = torch.abs(output - targets[take]).mean()
                    assert torch.isfinite(loss).item(), 'Nonfinite training loss'
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    total_train += loss.item() * len(take)
                    seen_train += len(take)
                model.eval()
                total_val = 0.
                with torch.no_grad():
                    for start in range(0, len(val_idx), 256):
                        take = val_idx[start:start + 256]
                        pred = (point_forecast(model, inputs[take])[:, :, 0] * std + mean).clamp_min(0)
                        truth = targets[take, :, 0] * std + mean
                        total_val += torch.abs(pred - truth).sum().item()
                metric = total_val / (len(val_idx) * 12)
                assert np.isfinite(metric)
                if metric < best:
                    best, best_epoch = metric, epoch + 1
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / 'best_weights.pt')
                row = {'epoch': epoch + 1, 'train_normalized_mae': total_train / seen_train,
                    'validation_center_mae': metric, 'best_validation_center_mae': best,
                    'elapsed_seconds': time.time() - started}
                logs.append(row)
                dump(directory / 'epoch_log.json', logs)
                dump(directory / 'state.json', {'status': 'training', 'model': name, **row})
                print(json.dumps({'model': name, **row}), flush=True)
            model.load_state_dict(torch.load(directory / 'best_weights.pt', map_location=DEVICE, weights_only=True))
            model.eval()
            dump(directory / 'scaler.json', {'mean': mean, 'std': std})
            original_predictions = []
            with torch.no_grad():
                for start in range(0, len(test), 256):
                    take = test[start:start + 256]
                    pred = (point_forecast(model, inputs[take])[:, :, 0] * std + mean).clamp_min(0)
                    original_predictions.append(pred.cpu().numpy())
            np.savez_compressed(directory / 'original_test_predictions.npz', test_indices=test,
                prediction=np.concatenate(original_predictions).astype(np.float32))
            mv_predictions = []
            with torch.no_grad():
                for start in range(0, len(ids), 256):
                    history = torch.from_numpy((mv_history[start:start + 256] - mean) / std).to(DEVICE)
                    pred = (point_forecast(model, history)[:, :, 0] * std + mean).clamp_min(0)
                    mv_predictions.append(pred.cpu().numpy())
            prediction = np.concatenate(mv_predictions).astype(np.float32)
            assert prediction.shape == (68544, 12) and np.isfinite(prediction).all()
            np.savez_compressed(directory / 'mv_predictions.npz', sample_id=ids, prediction=prediction)
            record = {'id': name, 'variant': variant, 'seed': seed, 'kind': 'GraphWaveNet',
                'weight_file': str((directory / 'best_weights.pt').relative_to(ROOT)),
                'weight_sha256': sha(directory / 'best_weights.pt'),
                'prediction_file': str((directory / 'mv_predictions.npz').relative_to(ROOT)),
                'prediction_sha256': sha(directory / 'mv_predictions.npz'),
                'original_prediction_sha256': sha(directory / 'original_test_predictions.npz'),
                'scaler_sha256': sha(directory / 'scaler.json'), 'best_epoch': best_epoch,
                'best_validation_center_mae': best, 'parameters': sum(p.numel() for p in model.parameters()),
                'fit_seconds': time.time() - started, 'training_test_targets_nan': True,
                'convolution_storage': 'FP32 channels_last',
                'mv_targets_decoded_for_fit_or_prediction': False}
            models.append(record)
            dump(directory / 'TRAINING_RESULT.json', record)
            dump(directory / 'state.json', {'status': 'completed_and_forecast_sealed', **record})
            ledger.update({'fits_completed': len(models), 'mv_frozen_prediction_configurations_completed': len(models),
                'elapsed_seconds': time.time() - began, 'latest_completed': name})
            dump(ROOT / 'SCIENTIFIC_LEDGER.json', ledger)
            print(json.dumps({'completed': name, 'fits_completed': len(models), 'elapsed_seconds': time.time() - began}), flush=True)
            del model, optimizer
        del inputs, targets
    dump(ROOT / 'FORECAST_SEAL.json', {'models': models, 'sealed_utc': stamp(),
        'inputs_sha256': sha(ROOT / 'task/inputs.npz'), 'protocol_sha256': sha(ROOT / 'PROTOCOL_ZH.md'),
        'runner_sha256': sha(Path(__file__)), 'adapter_sha256': sha(ROOT / 'model_adapter.py'),
        'mv_evaluation_targets_decoded_before_sealing': False, 'prior_development_exposure': True})
    ledger.update({'status': 'fits_and_predictions_complete_awaiting_evaluation', 'finished_utc': stamp()})
    dump(ROOT / 'SCIENTIFIC_LEDGER.json', ledger)

if __name__ == '__main__':
    main()
