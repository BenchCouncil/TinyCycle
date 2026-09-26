"""Validation-only selection, followed by the paper's fixed lookback extension."""
import argparse
import math
from pathlib import Path
from selection import RULE

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic', 'Exchange']
PATHS = {**{d: f'ETT-small/{d}.csv' for d in DATASETS[:4]},
         'Weather': 'weather/weather.csv', 'Electricity': 'electricity/electricity.csv',
         'Traffic': 'traffic/traffic.csv', 'Exchange': 'exchange_rate/exchange_rate.csv'}
LAMBDAS = [0., .001, .003, .01, .03, .1, .3, 1., 3., 10., 30., 100., 300., 1000., 3000., 10000.]
FINAL_LOOKBACK_RULE = 'keep_validation_selected_period_and_set_cycles_to_floor_720_over_period'


def final_configuration(row, horizon, grid):
    """Apply the Table 1 extension without inspecting any new score."""
    return {'seq_len': (720 // row['period']) * row['period'],
            'pred_len': horizon, 'period': row['period'],
            'regularization': row['regularization'], 'revin': row['revin'],
            'tau_cycles': row['tau_cycles'], 'fit_samples': grid['fit_samples'],
            'batch_size': grid['batch_size']}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='TinyCycle: horizon-shared validation selection, train-only fit, frozen test')
    p.add_argument('--stage', choices=['all', 'search', 'train', 'test'], default='all')
    p.add_argument('--data-root', type=Path, default=Path('dataset'))
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--data-path', help='CSV relative to data-root; requires one dataset name')
    p.add_argument('--train-end', type=int, help='Custom dataset: exclusive training row boundary')
    p.add_argument('--val-end', type=int, help='Custom dataset: exclusive validation row boundary')
    p.add_argument('--test-end', type=int, help='Custom dataset: exclusive final row boundary')
    p.add_argument('--horizons', type=int, nargs='+', default=[96, 192, 336, 720])
    p.add_argument('--periods', type=int, nargs='+', default=[4, 8, 24, 96])
    p.add_argument('--cycles', type=int, nargs='+', help='Default: all legal n=1..29 for each period')
    p.add_argument('--regularizations', type=float, nargs='+', default=LAMBDAS)
    p.add_argument('--taus', type=float, nargs='+', default=[1., 3., 7., 14., 28.])
    p.add_argument('--centering', type=int, nargs='+', choices=[0, 1], default=[0, 1])
    p.add_argument('--fit-samples', type=int, default=4096)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output-dir', type=Path, default=Path('results/paper'))
    p.add_argument('--resume', action='store_true', help='Reuse only matching, hash-verified search caches')
    p.add_argument('--dry-run', action='store_true', help='Print search counts without reading dataset values')
    args = p.parse_args(argv)
    if args.data_path and len(args.datasets) != 1:
        p.error('--data-path requires exactly one --datasets name')
    if not args.data_path and any(d not in DATASETS for d in args.datasets):
        p.error('unknown dataset requires --data-path and explicit --train-end/--val-end/--test-end')
    if any('/' in d or '\\' in d or d in ['.', '..'] for d in args.datasets):
        p.error('dataset names must be plain folder names')
    bounds = [args.train_end, args.val_end, args.test_end]
    if any(v is not None for v in bounds):
        if not all(v is not None for v in bounds) or not (0 < bounds[0] < bounds[1] < bounds[2]):
            p.error('supply all three boundaries with 0 < train-end < val-end < test-end')
        if len(args.datasets) != 1 or args.datasets[0] in DATASETS:
            p.error('custom boundaries are only available for one custom dataset')
    elif args.datasets[0] not in DATASETS:
        p.error('custom datasets require all three explicit boundaries')
    for field in ['datasets', 'horizons', 'periods', 'cycles', 'regularizations', 'taus', 'centering']:
        values = getattr(args, field)
        if values is not None and len(values) != len(set(values)):
            p.error(f'--{field} contains duplicate values')
    if any(h < 1 for h in args.horizons) or any(not 1 <= v <= 720 for v in args.periods):
        p.error('horizons must be positive; periods must lie in [1,720]')
    if args.cycles and any(not 1 <= n <= 29 for n in args.cycles):
        p.error('cycles must lie in [1,29]')
    if any(not math.isfinite(v) or v < 0 for v in args.regularizations):
        p.error('regularizations must be finite and nonnegative')
    if any(not math.isfinite(v) or v <= 0 for v in args.taus):
        p.error('taus must be finite and positive')
    if any(getattr(args, f) < 1 for f in ['fit_samples', 'batch_size', 'threads']):
        p.error('fit-samples, batch-size and threads must be positive')
    if any(not legal_cycles(args, period) for period in args.periods):
        p.error('each period must have at least one cycle count satisfying nP<=720')
    return args


def legal_cycles(args, period):
    return [n for n in (args.cycles or range(1, 30)) if n * period <= 720]


def search_spec(args):
    return {'periods': list(args.periods),
            'cycles': {str(p): legal_cycles(args, p) for p in args.periods},
            'regularizations': list(args.regularizations), 'taus': list(args.taus),
            'centering': list(args.centering), 'horizons': list(args.horizons),
            'fit_samples': args.fit_samples, 'batch_size': args.batch_size,
            'max_lookback': 720, 'search_max_parameters': 30, 'fit_split': 'train',
            'final_lookback_rule': FINAL_LOOKBACK_RULE,
            'selection_split': 'validation', 'selection_metric': RULE,
            'shared_fields': ['period', 'cycles', 'regularization', 'revin', 'tau_cycles'],
            'boundary_gain_fit': 'train_only_separately_for_each_horizon',
            'final_fit_split': 'train', 'age_exponent': 2,
            'tie_break': ['parameters', 'lookback', 'regularization', 'revin', 'tau_cycles', 'period'],
            'candidates_per_horizon': sum(len(legal_cycles(args, p)) for p in args.periods)
                * len(args.regularizations) * len(args.taus) * len(args.centering)}
