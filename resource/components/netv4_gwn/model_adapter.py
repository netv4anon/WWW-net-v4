# -*- coding: utf-8 -*-
"""Official adaptive-only Graph WaveNet on nine local numeric slots."""
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

def build_model(device='cpu'):
    # The old source declares 2-D kernels via Conv1d in exactly three locations.
    source = (ROOT / 'vendor/model.py').read_text(encoding='utf-8-sig')
    assert source.count('nn.Conv1d(') == 3
    namespace = {'__name__': 'pinned_graph_wavenet_compat'}
    exec(compile(source.replace('nn.Conv1d(', 'nn.Conv2d('), str(ROOT / 'vendor/model.py'), 'exec'), namespace)
    return namespace['gwnet'](torch.device(device), num_nodes=9, dropout=0.3, supports=None,
        gcn_bool=True, addaptadj=True, aptinit=None, in_dim=1, out_dim=12,
        residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
        kernel_size=2, blocks=4, layers=2).to(device).to(memory_format=torch.channels_last)

def point_forecast(model, normalized_history):
    """[B,12,9] -> [B,12,9], with no identifiers or evaluator inputs."""
    if normalized_history.ndim != 3 or tuple(normalized_history.shape[1:]) != (12, 9):
        raise ValueError('Expected historical numeric input [B,12,9]')
    return model(normalized_history.transpose(1, 2).unsqueeze(1))[..., -1]
