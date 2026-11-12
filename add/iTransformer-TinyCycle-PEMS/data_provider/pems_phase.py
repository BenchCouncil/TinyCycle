"""PEMS flow splits and rolling windows for the phase-aligned benchmark."""
import hashlib

import numpy as np
import torch

DATASETS = {
    'PEMS03': (26208, 358, '651add9bb9eaf7f5eda2f2ee8778a182'),
    'PEMS04': (16992, 307, '5238832691101da85fcac772a32051da'),
    'PEMS07': (28224, 883, '978d3d9b85fe640a446983a34271a48d'),
    'PEMS08': (17856, 170, '2a528d169c0d90294c9e24288a430132'),
}

def load_data(name, root, test=False):
    n, channels, md5 = DATASETS[name]
    path = root / f'{name}.npz'
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'md5').hexdigest() != md5:
            raise ValueError(f'Unexpected dataset version: {path}')
    a, b = int(n * .6), int(n * .8)
    stop = n if test else b
    with np.load(path, allow_pickle=False) as archive:
        x = archive['data'][:stop, :, 0].astype(np.float64)
    if x.shape != (stop, channels) or not np.isfinite(x).all():
        raise ValueError(f'Unexpected flow data in {path}: {x.shape}')
    mu, sig = x[:a].mean(0), x[:a].std(0)
    sig[sig == 0] = 1
    return torch.from_numpy(((x - mu) / sig).astype(np.float32)).cuda(), a, b


def input_at(data, ids, lookback):
    return data[ids[:, None] + torch.arange(-lookback, 0, device=data.device)]
