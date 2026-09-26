#!/usr/bin/env python3
"""Post-test verification only; never called by fitting or selection code."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Compare a completed 32-task reproduction with the reference')
    parser.add_argument('--results', type=Path, required=True, help='Directory containing completed summary.csv')
    parser.add_argument('--atol', type=float, default=1e-7)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error('--atol must be finite and nonnegative')
    reference = pd.read_csv(Path(__file__).with_name('reference_32_tasks.csv'))
    actual = pd.read_csv(args.results / 'summary.csv')
    keys = ['dataset', 'horizon']
    expected_keys = set(map(tuple, reference[keys].to_numpy()))
    if len(actual) != 32 or set(map(tuple, actual[keys].to_numpy())) != expected_keys:
        raise SystemExit('Expected exactly the 8 datasets x 4 horizons in the default replication protocol')
    joined = reference.merge(actual, on=keys, suffixes=('_reference', '_actual'), validate='one_to_one')
    configuration_columns = ['period', 'cycles', 'regularization', 'revin', 'tau_cycles', 'parameters', 'lookback']
    same_config = all(np.array_equal(joined[k + '_reference'], joined[k + '_actual']) for k in configuration_columns)
    deltas = {}
    for key in ['validation_mse', 'test_mse', 'test_mae']:
        expected = joined[key + '_reference'].to_numpy(dtype=float)
        measured = joined[key + '_actual'].to_numpy(dtype=float)
        # Check every value before reduction: pandas reductions can skip NaNs.
        if not np.isfinite(expected).all() or not np.isfinite(measured).all():
            raise SystemExit(f'{key} contains missing or nonfinite values')
        deltas[key] = float(np.max(np.abs(expected - measured)))
    passed = same_config and all(np.isfinite(v) and v <= args.atol for v in deltas.values())
    result = {'passed': passed, 'tasks': len(joined), 'all_selected_configurations_match': same_config,
              'max_absolute_differences': deltas, 'absolute_tolerance': args.atol,
              'actual_mean_test_mse': float(actual.test_mse.mean()),
              'actual_mean_test_mae': float(actual.test_mae.mean()),
              'reference_mean_test_mse': float(reference.test_mse.mean()),
              'reference_mean_test_mae': float(reference.test_mae.mean()),
              'usage': 'Post-test comparison only; reference metrics never enter training or selection.'}
    (args.results / 'reproduction_check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
