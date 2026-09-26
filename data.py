"""Chronological splits; search never loads numeric rows from the test interval."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def describe_dataset(name, path, bounds=None):
    path = Path(path)
    # Only row-count metadata is needed to define proportional boundaries.
    # The first column (normally date) is not used as a model/selection input.
    total_rows = sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=65536))
    if bounds is not None:
        train_end, val_end, test_end = bounds
    elif name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        month = 30 * 24 * (4 if name.startswith('ETTm') else 1)
        train_end, val_end, test_end = 12 * month, 16 * month, 20 * month
    else:
        train_end, val_end, test_end = int(total_rows * .7), total_rows - int(total_rows * .2), total_rows
    if not 0 < train_end < val_end < test_end <= total_rows:
        raise ValueError(f'Invalid chronological split or truncated CSV: {path}')
    return {'dataset': name, 'csv_path': str(path.resolve()), 'total_csv_rows': total_rows,
            'train_end': train_end, 'val_end': val_end, 'test_end': test_end}


def read_values(spec, end):
    if end not in [spec['train_end'], spec['val_end'], spec['test_end']]:
        raise ValueError('end must be a declared split boundary')
    frame = pd.read_csv(spec['csv_path'], nrows=end)
    if len(frame) != end:
        raise ValueError('CSV is shorter than the declared split')
    if 'date' in frame:
        dates = pd.to_datetime(frame.pop('date'), errors='raise')
        # The standard Weather CSV has a repeated timestamp. Benchmark splits
        # and cycle lags are defined by row index: never sort or deduplicate it.
        if dates.isna().any() or not dates.is_monotonic_increasing:
            raise ValueError('date must be finite and chronologically nondecreasing')
    if 'OT' in frame and not spec['dataset'].startswith('ETT'):
        frame = frame[[c for c in frame if c != 'OT'] + ['OT']]
    values = frame.to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('All modeled columns must be numeric and finite; no cross-split imputation is performed')
    return values, list(frame.columns)


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def fit_scaler(train):
    scaler = StandardScaler().fit(train)
    return {'fit_split': 'train', 'n_samples': len(train),
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'var': scaler.var_.tolist()}


def transform(values, scaler):
    if scaler['fit_split'] != 'train' or values.shape[1] != len(scaler['mean']):
        raise ValueError('Invalid frozen training scaler')
    values = values.copy()
    # Match StandardScaler's float32 in-place transform and existing TinyCycle data protocol.
    values -= np.asarray(scaler['mean'], dtype=np.float64)
    values /= np.asarray(scaler['scale'], dtype=np.float64)
    return torch.from_numpy(values)


def development_data(spec):
    raw, columns = read_values(spec, spec['val_end'])
    scaler = fit_scaler(raw[:spec['train_end']])
    identity = {**spec, 'columns': columns, 'channels': len(columns),
                'train_sha256': array_hash(raw[:spec['train_end']]),
                'development_sha256': array_hash(raw), 'numeric_rows_loaded': spec['val_end']}
    return transform(raw, scaler), scaler, identity


def test_data(identity, scaler):
    """Called only after all selected weights and configurations are frozen."""
    raw, columns = read_values(identity, identity['test_end'])
    if columns != identity['columns']:
        raise RuntimeError('CSV columns changed after selection')
    if array_hash(raw[:identity['val_end']]) != identity['development_sha256']:
        raise RuntimeError('Train/validation data changed after selection')
    return transform(raw, scaler), array_hash(raw[identity['val_end']:])
