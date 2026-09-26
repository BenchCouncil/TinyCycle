# Reproducibility and evidence status

A fresh, uncached run completed **481,280 validation candidate scores and all 32 final models**. The final configurations, lookbacks, parameter counts, and data fingerprints match the paper reference. All **20 regression tests pass** in the pinned environment. The machine-readable evidence is [reproduction/verification.json](../reproduction/verification.json).

| Check | Result |
|---|---:|
| Mean test MSE | 0.3208019092855713 |
| Mean test MAE | 0.3446035924245721 |
| Mean parameters / lookback | 25.25 / 708 |
| Maximum final validation MSE difference | 2.220446049250313e-16 |
| Maximum test MSE / MAE difference | 1.6653345369377348e-16 |
| Maximum fitted-weight difference from paper checkpoints | 3.587408148320037e-15 |
| Data fingerprints and frozen-artifact checks | Passed |

The run used Python 3.9.6, the pinned package versions below, and two CPU threads. It started from an empty output directory and did not load historical candidates, configurations, or weights. References were consulted only after completion. After this run, the CLI gained a preflight check that verifies every requested dataset is frozen before standalone testing starts. That orchestration-only change was tested separately; all search, fitting, selection, and evaluation sources remain byte-identical to the full run. Both source manifests are recorded in the report.

## Historical reference provenance

The reference has two distinct stages of evidence:

1. The original short-window implementation's README and verification record document an uncached search over 8 datasets × 4 horizons, 481,280 validation scores, and 32 refitted frozen models. Its historical mean test MSE/MAE was `0.32352427445850485 / 0.34598437552041933`, with 20.875 parameters and a 576-step mean lookback. This supports the original shared-selection result; it is not verification of the extended public pipeline.
2. The subsequent fixed-hyperparameter extension retained those selected `P`, `λ`, `revin`, and `τ`, set `Q = floor(720 / P)`, and refitted on training data. The resulting 32-task report supplies the final reference distributed as [reproduction/reference_32_tasks.csv](../reproduction/reference_32_tasks.csv). The paper's Table 1 uses these extended model configurations, with timing measured separately in an optimized CPU implementation.

The distributed CSV is byte-for-byte identical to the historical fixed-hyperparameter extension report. Its SHA-256 is:

```text
5c9b752e7903cba71a9485e412091678e321d3b594139dc2100d443c25f06bfe
```

For provenance, the archived original short-window reference CSV has SHA-256:

```text
1ac63495b44464eeff8aef4078b3557d81d20d1d1a027c016632a96fd00ac9cf
```

That older reference is not the final target and is not required to run this repository. These fingerprints identify historical artifacts; they do not certify the execution of a new run. Private archive locations are not needed by the public pipeline.

## Expected configurations and metrics

These are historical outcomes to check **after a run**, not dataset-specific defaults to load before search. Each row applies to all four horizons. `Q_search` is from the original shared-selection reference; `Q_final` and the final metrics are from the distributed reference CSV.

| Dataset | P | Q_search | λ | revin | τ | Q_final | Final lookback | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 24 | 29 | 30 | 0 | 14 | 30 | 720 | 31 |
| ETTh2 | 24 | 9 | 10 | 1 | 14 | 30 | 720 | 31 |
| ETTm1 | 96 | 4 | 0 | 0 | 1 | 7 | 672 | 8 |
| ETTm2 | 96 | 7 | 3 | 1 | 3 | 7 | 672 | 8 |
| Weather | 24 | 29 | 0 | 0 | 3 | 30 | 720 | 31 |
| Electricity | 24 | 29 | 0 | 0 | 1 | 30 | 720 | 31 |
| Traffic | 24 | 29 | 0.01 | 1 | 1 | 30 | 720 | 31 |
| Exchange | 24 | 23 | 1000 | 1 | 28 | 30 | 720 | 31 |

The equal-weight means of the final 32 tasks are:

| Quantity | Historical target |
|---|---:|
| Test MSE | 0.320801909286 |
| Test MAE | 0.344603592425 |
| Learned parameters | 25.25 |
| Lookback | 708 |

The reference CSV retains per-task precision. Its `validation_mse` describes the extended final model and is a diagnostic score, not the original candidate score used for selection. Do not compare it to `selection_validation_mse` or use it to repeat selection. None of the reference metrics or expected configurations may be read by training or search.

## Data-version fingerprints

The following SHA-256 values identify the raw benchmark CSV bytes used by the historical reference. Paths are relative to `--data-root`. These fingerprints were checked against the local source files. Data entry points are the [official ETDataset repository](https://github.com/zhouhaoyi/ETDataset) and the benchmark downloads linked in the [Time-Series-Library README](https://github.com/thuml/Time-Series-Library). A download's data version should still be checked against these fingerprints. Raw data are not included in this repository.

| Relative CSV path | SHA-256 |
|---|---|
| `ETT-small/ETTh1.csv` | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| `ETT-small/ETTh2.csv` | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` |
| `ETT-small/ETTm1.csv` | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` |
| `ETT-small/ETTm2.csv` | `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd` |
| `weather/weather.csv` | `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63` |
| `electricity/electricity.csv` | `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad` |
| `traffic/traffic.csv` | `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16` |
| `exchange_rate/exchange_rate.csv` | `48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842` |

Line-ending or CSV-format changes alter a file hash. The separate [reproduction/data_fingerprints.json](../reproduction/data_fingerprints.json) contains SHA-256 values for row-major float32 numeric arrays after the documented column ordering and before standardization. These numeric hashes are distinct from the raw CSV hashes above.

After a run, compare each dataset's `dataset.json` fields `train_sha256` and `development_sha256`, along with its split endpoints and channel count, against the matching entry in that file. The `test_data_sha256` in `test_metrics.json` corresponds to the reference `test_sha256`. The runtime also checks channel order and the training scaler. The reference fingerprint file is for post-run comparison only and is never read by training or search. Investigate mismatches rather than silently treating them as the same data version.

## Verification procedure

Run commands from the repository root. Use a new output directory for the fresh run; do not copy candidate caches, selected configurations, checkpoints, or reports into it.

### 1. Record environment and check small tests

```bash
python --version
python -m pip freeze
python -m unittest discover -s tests -v
python run.py --dry-run
```

Record the operating system, CPU, Python and dependency versions, thread count, exact command, source revision or hashes, and data fingerprints alongside the eventual report. The historical original verification used Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, PyTorch 2.8.0, scikit-learn 1.6.1, and SciPy 1.13.1 on an ARM CPU with two threads. The pinned dependency file preserves those package versions; it does not imply that every platform has been benchmarked.

The dry run must show 15,040 candidates per horizon and 481,280 rows for all default datasets/horizons. Unit tests exercise synthetic-data selection, numerical scoring, data isolation, fitting boundaries, the final lookback extension, the 720-step input limit, freezing, and resume behavior. The GitHub Actions workflow runs CPU unit tests only, without benchmark data; its success is not full benchmark verification.

### 2. Run all stages from scratch

```bash
python -u run.py --data-root dataset --output-dir results/fresh
```

Retain the default four horizons, complete search grid, and `--fit-samples 4096`. An interrupted run may continue with the same command plus `--resume`, but record which artifacts were reused. A run that starts from old complete caches must not be described as an uncached search.

### 3. Audit the saved evidence

Check the following for each of the eight dataset directories:

- `search_spec.json` and the `p*_complete.json` records cover the original grid and all four horizons. The four `p*_validation_scores.csv.gz` files contain 60,160 rows in total per dataset; record any nonfinite candidates and their exclusion.
- `selected.json` contains one shared configuration, four per-horizon validation normalizers, and the original shared selection score. Compare the selected configuration to the historical `Q_search` table only after the run.
- Search cycle-weight targets start at `P * min(30, 720 // P)`. All weight targets are below `train_end`; all gain-fitting target windows end at or before `train_end`.
- The final `h*_config.json` files retain selected `P`, `λ`, `revin`, and `τ`, with `seq_len = P * floor(720 / P)`. Final fitting starts at `max(P * min(30, 720 // P), seq_len)` and uses training data only.
- `validation_metrics.csv` distinguishes the original candidate checks from final validation diagnostics. Direct scoring of the original selected candidate agrees with its search score within the implemented numerical tolerance. No extended-model score initiates reselection.
- `frozen.json` is written before numeric test data are loaded. The training scaler, model configurations, and checkpoints remain unchanged during testing, with recorded artifact hashes verified.
- Each test task covers `test_end - val_end - H + 1` origins and that count times `H × channels` target elements. `summary.csv` has exactly 32 unique dataset/horizon rows.

### 4. Compare completed results

```bash
python reproduction/compare_results.py --results results/fresh
```

The comparator requires the default 32-task coverage, exact agreement of the final configuration fields, and maximum absolute differences no greater than `1e-7` for final validation MSE, test MSE, and test MAE. It writes `reproduction_check.json` and exits unsuccessfully on a mismatch. This command is separate from fitting and selection; only this post-test step reads the reference CSV.

Check the mean parameter count and lookback in `summary.csv` as well. Investigate differences in data, grid, horizons, fitting origins, and numeric environment before changing any tolerance. Agreement with a reference is not permission to replace a failed search with the expected configuration.

### 5. Release acceptance

The complete run passed the procedure above. The 20 regression tests include perturbing and poisoning test values without changing selection or fitted weights, exact versus direct scoring, deterministic window extension, immutable checkpoints, rejecting missing/nonfinite comparison metrics, and rejecting incomplete dataset groups before testing. The result-comparison tolerance remains `1e-7`; the observed differences are substantially smaller.

The two source manifests distinguish the completed benchmark from the subsequent CLI preflight safeguard. Raw logs, run directories, and checkpoints are not distributed because they contain local paths; the public report includes only portable evidence and hashes.

## Timing scope

The paper's Table 1 timing is a separate historical measurement on an Intel Core i5-13490F, with four fixed physical performance cores and four computation threads. It uses optimized CPU fitting and inference, including a native inference kernel. The final configurations are the same extended configurations described above.

- **Train:** one complete fit of an already selected configuration, including its internal kernel preparation. Data loading, standardization, model initialization, validation, and hyperparameter search are excluded.
- **Test:** the median of three complete passes per task, including forecast-kernel construction, input conversion, prediction at all valid origins and channels, error aggregation, and compressed per-origin result saving. Warm-up and native-library compilation/loading are recorded outside this interval.

The historical means are approximately `0.526 s` for selected-model fitting and `0.063 s` for a complete test pass. They are not end-to-end search times. The portable Python workflow has different implementation overheads and does **not** claim to reproduce Table 1 timing. Its own `train_seconds` and `test_seconds` fields describe that execution only and must not be substituted for the paper's timing protocol.

## Development limitation

Earlier method development considered test feedback when choosing among validation-selection schemes. The public pipeline's training-only fitting, validation-only selection, and frozen testing describe its computational boundaries; they do not erase that history or establish that the overall method was developed without test exposure.
