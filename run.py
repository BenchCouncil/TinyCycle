#!/usr/bin/env python3
"""Standalone search -> final training -> frozen test pipeline."""
import json
import os

from config import FINAL_LOOKBACK_RULE, parse_args, search_spec
from selection import RULE


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        grid = search_spec(args)
        print(json.dumps({'datasets': args.datasets, 'search': grid,
                          'total_candidates': len(args.datasets) * len(args.horizons) * grid['candidates_per_horizon']}, indent=2))
        return
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    from io_utils import read_json, save_json, verify_files, write_csv
    from search import search_dataset
    from pipeline import freeze_selected, test_selected
    if args.stage in ['all', 'search']:
        for dataset in args.datasets:
            search_dataset(dataset, args)
    if args.stage in ['all', 'train']:
        for dataset in args.datasets:
            freeze_selected(dataset, args)
    # All requested datasets/horizons have been selected and trained before any test evaluation.
    if args.stage in ['all', 'test']:
        # Also enforce the group boundary for a standalone test invocation:
        # never test one dataset before discovering that another is not frozen.
        for dataset in args.datasets:
            folder = args.output_dir / dataset
            manifest = folder / 'frozen.json'
            if not manifest.exists():
                raise RuntimeError(f'Run search then train for every requested dataset before test: {folder}')
            verify_files(folder, read_json(manifest)['artifacts'])
        rows = []
        for dataset in args.datasets:
            rows.extend(test_selected(dataset, args))
        write_csv(args.output_dir / 'summary.csv', rows)
        summary = {'tasks': len(rows), 'fit_split': 'train', 'selection_split': 'validation',
                   'selection_rule': RULE,
                   'final_lookback_rule': FINAL_LOOKBACK_RULE,
                   'mean_test_mse': sum(r['test_mse'] for r in rows) / len(rows),
                   'mean_test_mae': sum(r['test_mae'] for r in rows) / len(rows),
                   'mean_parameters': sum(r['parameters'] for r in rows) / len(rows),
                   'mean_lookback': sum(r['lookback'] for r in rows) / len(rows)}
        save_json(args.output_dir / 'summary.json', summary)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
