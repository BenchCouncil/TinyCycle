"""Exhaustive validation-only search using exact pooled MSE statistics."""
import csv
import gzip
import time
from pathlib import Path

import numpy as np
import torch

from all_order_stats import build_all_stats, stats_for_q
from config import PATHS, search_spec
from data import describe_dataset, development_data
from fit_boundary import mark_train_stats, fit_gain_train, score_fixed_gain
from io_utils import checked_json, read_json, save_json, sha256, verify_files
from kernel_scoring import ar_kernel, score_kernel
from train import sample_times, sharing_penalty, solve_weights
from selection import select_shared


def rank(row):
    return tuple(row[k] for k in ['validation_mse', 'parameters', 'lookback',
                                  'regularization', 'revin', 'tau_cycles', 'period'])


def source_hashes():
    root = Path(__file__).resolve().parent
    paths = sorted(root.glob('*.py')) + sorted((root / 'models').glob('*.py'))
    return {str(p.relative_to(root)): sha256(p) for p in paths}


@torch.no_grad()
def fit_all_weights(train, period, orders, strengths, fit_samples):
    qmax = max(orders)
    start = period * min(30, 720 // period)
    times = sample_times(start, len(train), fit_samples)
    scale = train.std(0, unbiased=True).clamp_min(1e-4).double()
    gram = torch.zeros(qmax, qmax, dtype=torch.float64)
    cross = torch.zeros(qmax, dtype=torch.float64)
    lags = period * torch.arange(1, qmax + 1)
    batch = max(1, min(256, 2_000_000 // (train.shape[1] * qmax)))
    for at in times.split(batch):
        x = train[at[:, None] - lags].permute(0, 2, 1).double() / scale[None, :, None]
        y = train[at].double() / scale[None, :]
        x, y = x.reshape(-1, qmax), y.reshape(-1)
        gram += x.T @ x
        cross += x.T @ y
    weights = {}
    for n in orders:
        g, c = gram[:n, :n], cross[:n]
        penalty = g.diagonal().mean().clamp_min(1e-12) * sharing_penalty(g)
        for li, strength in enumerate(strengths):
            theta = solve_weights(g + strength * penalty, c).numpy()
            if not np.isfinite(theta).all():
                raise ArithmeticError('Nonfinite training weights')
            weights[n, li] = theta
    return weights


def kernel_for(weights, horizon, period, centered):
    kernel = np.pad(ar_kernel(weights, horizon, period), ((0, 0), (0, 1)))
    if centered:
        kernel[:, -1] = 1 - kernel[:, :-1].sum(1)
    return kernel


def run_period(folder, train, development, identity, grid, period):
    record_path = folder / f'p{period}_complete.json'
    if record_path.exists():
        record = read_json(record_path)
        verify_files(folder, record['artifacts'])
        print(f'Resume {identity["dataset"]} P={period}: verified cached validation scores', flush=True)
        return record
    began = time.perf_counter()
    orders = grid['cycles'][str(period)]
    qmax = max(orders)
    weights = fit_all_weights(train, period, orders, grid['regularizations'], grid['fit_samples'])
    weight_path = folder / f'p{period}_train_weights.npz'
    np.savez_compressed(weight_path, **{f'{q}_{li}': w for (q, li), w in weights.items()})
    score_path = folder / f'p{period}_validation_scores.csv.gz'
    temporary = score_path.with_suffix('.gz.tmp')
    fields = ['horizon', 'period', 'cycles', 'regularization', 'revin', 'tau_cycles',
              'boundary_gain', 'parameters', 'lookback', 'validation_mse', 'status']
    winners, boundaries, total, invalid = {}, [], 0, 0
    start = period * min(30, 720 // period)
    with gzip.open(temporary, 'wt', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for horizon in grid['horizons']:
            train_origins = sample_times(start, len(train) - horizon + 1, grid['fit_samples']).numpy()
            # These arrays are separate prefixes. Test values cannot enter either accumulator.
            train_all = build_all_stats(train, train_origins, horizon, period, qmax, grid['taus'])
            val_origins = np.arange(identity['train_end'], identity['val_end'] - horizon + 1)
            val_all = build_all_stats(development, val_origins, horizon, period, qmax, grid['taus'])
            for n in orders:
                train_stats = stats_for_q(train_all, n)
                mark_train_stats(train_stats, train_end=len(train), data_rows=len(train), origins=train_origins)
                val_stats = stats_for_q(val_all, n)
                val_stats.split = 'validation'
                for li, strength in enumerate(grid['regularizations']):
                    for centered in grid['centering']:
                        kernel = kernel_for(weights[n, li], horizon, period, centered)
                        base = score_kernel(val_stats, kernel)['mse']
                        for tau in grid['taus']:
                            gain, mse = 0., float('inf')
                            if np.isfinite(kernel).all() and np.isfinite(base):
                                gain = fit_gain_train(train_stats, kernel, tau)
                                mse = score_fixed_gain(val_stats, kernel, base, tau, gain)
                            finite = bool(np.isfinite(mse))
                            row = dict(horizon=horizon, period=period, cycles=n, regularization=strength,
                                       revin=centered, tau_cycles=tau, boundary_gain=gain,
                                       parameters=n + 1, lookback=n * period,
                                       validation_mse=mse, status='ok' if finite else 'nonfinite')
                            writer.writerow(row)
                            total += 1
                            invalid += int(not finite)
                            if finite and (str(horizon) not in winners or rank(row) < rank(winners[str(horizon)])):
                                winners[str(horizon)] = row
            boundaries.append({'horizon': horizon, 'weight_fit_first_target': start,
                               'weight_fit_last_target': len(train) - 1,
                               'gain_fit_origins': len(train_origins),
                               'gain_fit_last_target_exclusive': int(train_origins[-1]) + horizon,
                               'validation_first_origin': int(val_origins[0]),
                               'validation_last_origin': int(val_origins[-1]),
                               'validation_target_end_exclusive': int(val_origins[-1]) + horizon,
                               'validation_origins': len(val_origins)})
            del train_all, val_all
            print(f'Search {identity["dataset"]} P={period} L={horizon}: full validation scored', flush=True)
    temporary.replace(score_path)
    expected = len(orders) * len(grid['regularizations']) * len(grid['centering']) * len(grid['taus']) * len(grid['horizons'])
    if total != expected:
        raise RuntimeError('Incomplete candidate enumeration')
    record = {'period': period, 'rows': total, 'nonfinite_candidates': invalid,
              'fit_split': 'train', 'selection_split': 'validation', 'winners': winners,
              'bounds': boundaries, 'seconds': time.perf_counter() - began,
              'artifacts': {p.name: sha256(p) for p in [weight_path, score_path]}}
    save_json(record_path, record)
    return record


def search_dataset(dataset, args):
    folder = args.output_dir / dataset
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise RuntimeError(f'Output exists: {folder}. Use --resume with identical settings or a new output directory.')
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = args.data_root / (args.data_path or PATHS[dataset])
    bounds = (args.train_end, args.val_end, args.test_end) if args.train_end is not None else None
    spec = describe_dataset(dataset, csv_path, bounds)
    grid = search_spec(args)
    for horizon in grid['horizons']:
        if spec['val_end'] - spec['train_end'] < horizon or spec['test_end'] - spec['val_end'] < horizon:
            raise ValueError(f'{dataset}: validation/test interval is shorter than horizon {horizon}')
    if spec['train_end'] <= max(p * min(30, 720 // p) for p in grid['periods']) + max(grid['horizons']) - 1:
        raise ValueError('Training split has no complete gain-fitting window for the requested search grid')
    development, scaler, identity = development_data(spec)
    train = development[:spec['train_end']].clone()
    checked_json(folder / 'search_spec.json', grid)
    checked_json(folder / 'source_hashes.json', source_hashes())
    checked_json(folder / 'dataset.json', identity)
    checked_json(folder / 'scaler.json', scaler)
    finished = folder / 'search_finished.json'
    if finished.exists():
        verify_files(folder, read_json(finished)['artifacts'])
        print(f'Resume {dataset}: complete validation search verified', flush=True)
        return
    records = [run_period(folder, train, development, identity, grid, p) for p in grid['periods']]
    def candidates():
        for period in grid['periods']:
            with gzip.open(folder / f'p{period}_validation_scores.csv.gz', 'rt', newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    if int(row['period']) != period:
                        raise ValueError('Wrong period in validation table')
                    yield row
    selected = {'dataset': dataset, 'fit_split': 'train',
                'selection_metric': grid['selection_metric'],
                **select_shared(candidates(), grid['horizons'], grid['candidates_per_horizon'])}
    save_json(folder / 'selected.json', selected)
    artifacts = ['selected.json', 'search_spec.json', 'source_hashes.json', 'dataset.json', 'scaler.json']
    artifacts += [f'p{p}_complete.json' for p in grid['periods']]
    artifacts += [name for record in records for name in record['artifacts']]
    save_json(finished, {'selection_split': 'validation', 'candidate_count': sum(r['rows'] for r in records),
                        'test_values_used': False, 'artifacts': {name: sha256(folder / name) for name in artifacts}})
    print(f'Selected {dataset}: {selected["shared_configuration"]}; '
          f'normalized validation score={selected["shared_selection_score"]:.9f}', flush=True)
