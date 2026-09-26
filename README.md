# TinyCycle

TinyCycle is a compact model for multivariate time-series forecasting. It learns shared cycle weights and a boundary correction from training data using small linear systems, then forecasts recursively. The portable implementation runs on CPU; no GPU or pretrained checkpoint is required.

This repository provides one paper reproduction pipeline:

1. Search the original grid of **15,040 candidates per dataset and forecast horizon**.
2. Select one shared configuration per dataset using validation MSE, normalized separately for horizons **96, 192, 336, and 720**, then averaged equally.
3. Keep the selected period, regularization, mean-centering switch, and decay constant; set only the cycle count to `Q = floor(720 / P)`.
4. Refit on the training split, freeze all final models, and evaluate the test split.

The original search uses `Q <= 29`. The final extension is mandatory, including when `--cycles` is supplied. There is no full-window reselection or test-based selection mode.

The historical reference for the final 32 tasks is:

| Equal-weight mean over 8 datasets × 4 horizons | Value |
|---|---:|
| Test MSE | 0.320801909286 |
| Test MAE | 0.344603592425 |
| Learned parameters | 25.25 |
| Input lookback | 708 |

A fresh run completed all 481,280 validation scores and 32 final models. All final configurations match the reference; the largest per-task metric difference is `2.22e-16`. The 20 regression tests also pass. See [verification details](docs/REPRODUCIBILITY.md), the [machine-readable report](reproduction/verification.json), and the [full protocol](docs/PROTOCOL.md).

## Installation

Use a Python environment with the dependencies in [requirements.txt](requirements.txt). Python 3.11 is used by the CPU unit-test workflow.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[requirements-tested.txt](requirements-tested.txt) records the pinned dependency versions from the historical verification environment. The same package versions were used for the complete release verification.

## Data

Supply the benchmark CSV files yourself in this layout:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/weather.csv
├── electricity/electricity.csv
├── traffic/traffic.csv
└── exchange_rate/exchange_rate.csv
```

Each CSV contains a `date` column and numeric feature columns. All numeric channels are modeled. Preserve the original rows and column values: do not shuffle, resample, sort, deduplicate, or impute across splits. In particular, the reference Weather file contains a repeated timestamp, which is retained.

ETT is available from the [official ETDataset repository](https://github.com/zhouhaoyi/ETDataset). The [Time-Series-Library README](https://github.com/thuml/Time-Series-Library) provides benchmark data download links. Arrange the downloaded CSV files as shown above; raw datasets are not redistributed here.

The [protocol](docs/PROTOCOL.md#data-and-splits) defines chronological splits and training-only standardization. [Data fingerprints](docs/REPRODUCIBILITY.md#data-version-fingerprints) identify the reference files and explain how to compare completed-run metadata with [reproduction/data_fingerprints.json](reproduction/data_fingerprints.json). This reference file is only for checking data identity and never enters training or selection.

## Run the paper protocol

Run these commands from the repository root:

```bash
# Inspect the grid without loading dataset values.
python run.py --dry-run

# Complete all eight datasets and all four horizons.
python -u run.py --data-root dataset --output-dir results/paper

# Resume the same run with unchanged data, code, and search settings.
python -u run.py --data-root dataset --output-dir results/paper --resume
```

The default dry run reports 15,040 candidates per horizon and **481,280 candidate scores** in total. All requested searches finish before final training, and all requested final models are frozen before any test evaluation.

The shell wrapper also works when invoked from another working directory:

```bash
# From the repository root:
DATA_ROOT=dataset OUTPUT_DIR=results/paper bash shell/run_all.sh

# From its parent directory; explicit relative paths use the calling directory:
DATA_ROOT=shared-data OUTPUT_DIR=runs/tinycycle \
  bash TinyCycle/shell/run_all.sh --resume
```

Without overrides, the wrapper uses `dataset/` and `results/paper/` inside the repository. `DATA_ROOT` and `OUTPUT_DIR` may contain spaces. `PYTHON` optionally selects a Python executable; additional arguments are forwarded to `run.py`. Explicit `--data-root` or `--output-dir` arguments override the corresponding environment default.

## Run individual stages

```bash
python -u run.py --stage search --data-root dataset --output-dir results/paper
python -u run.py --stage train --output-dir results/paper
python -u run.py --stage test --output-dir results/paper
```

`--stage all` is the default. `search` fits candidates using training data and selects on validation data. `train` verifies the selected search candidate, applies the deterministic lookback extension, refits on training data, and freezes checkpoints. `test` evaluates those checkpoints and writes aggregate results.

The `train` and `test` stages read the saved data location and search specification. Keep the data files at that location. Data-path and search-grid options supplied to `train` or `test` do not override the saved specification; start a new `search` run to change them. Use the same `--datasets` list and output directory for every stage; for example, add `--datasets ETTh1` to all three commands for a single-dataset run. Checkpoints and cached results are verified before reuse. Changed data, program sources, or search settings require a new output directory.

For paper reproduction, retain all four horizons together, the default grid, and `--fit-samples 4096`. Four separately selected horizons do not implement the shared selection rule. Changing the grid, horizons, fitting sample limit, or splits defines a different experiment. `--cycles` restricts only the original search cycle counts; it never disables the final extension or requests a final cycle count. In particular, `--cycles 30` is not a valid search setting.

## Results and checks

The output directory contains `summary.csv` and `summary.json`, plus one directory per dataset. Each dataset directory records the search specification, data and source fingerprints, validation candidate scores, selected configuration, training scaler, frozen checkpoints, and test metrics.

`selected.json` describes the original search winner. The final `h*_config.json` files and `frozen.json` describe the extended models. Output fields prefixed with `selection_` preserve the search candidate's cycle count, lookback, and validation MSE; the final `cycles`, `lookback`, and `validation_mse` describe the extended model. Its validation score is diagnostic and does not trigger another selection.

```bash
# Small synthetic-data checks; no benchmark download is needed.
python -m unittest discover -s tests -v

# Only after completing the default 32-task test stage:
python reproduction/compare_results.py --results results/paper
```

The comparison command checks the final configurations and validation/test metrics against [reference_32_tasks.csv](reproduction/reference_32_tasks.csv), with an absolute tolerance of `1e-7`, and writes `reproduction_check.json`. The reference file and the documented expected configurations are for **documentation and post-test comparison only**; they are not training inputs, candidate caches, or shortcuts around search.

## Scope and limitations

The runtime pipeline fits coefficients and standardization on training data, selects hyperparameters on validation data, and evaluates frozen models on test data. **Method development previously considered test feedback when choosing among validation-selection schemes.** This history does not support a claim that the method or its development process never encountered the test set.

Paper Table 1 reports historical timing on an Intel Core i5-13490F using four physical performance cores and an optimized CPU implementation. It measures fitting an already selected model and a complete test pass, excluding hyperparameter search. This portable Python pipeline does not claim to reproduce those timings or to finish search within those times. See the [timing scope](docs/REPRODUCIBILITY.md#timing-scope).
