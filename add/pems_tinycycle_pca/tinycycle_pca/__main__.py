"""Run the fixed-hyperparameter TinyCycle-PCA train/test experiment."""

import argparse
import warnings
from pathlib import Path

from .experiment import run_dataset, write_summary
from .settings import DATASETS


def main():
    parser = argparse.ArgumentParser(prog="python -m tinycycle_pca", description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing PEMS NPZ files")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fixed"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=tuple(DATASETS))
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets contains duplicate datasets")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with warnings.catch_warnings():
        # macOS Accelerate may raise spurious matmul flags; the numerical
        # checks in fitting and testing still reject nonfinite results.
        warnings.filterwarnings(
            "ignore", message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        for name in args.datasets:
            rows.extend(run_dataset(name, args.data_dir, args.output_dir))
    write_summary(args.output_dir, rows, args.datasets)


if __name__ == "__main__":
    main()
