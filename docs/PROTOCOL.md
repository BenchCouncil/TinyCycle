# TinyCycle reproduction protocol

This document specifies the single public pipeline: original-grid validation search, shared selection across forecast horizons, deterministic lookback extension, training-only refit, and frozen testing. Expected outcomes in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) are comparison data, not inputs to this pipeline.

## Notation and model

`P` is the cycle period, `Q` is the number of consecutive cycles, `L = P Q` is the input lookback (`seq_len`), and `H` is the forecast horizon (`pred_len`). The code uses `cycles` for `Q`, `regularization` for λ, `revin` for the input-window mean-centering switch, and `tau_cycles` for τ in cycle units. Here, `revin` denotes this mean correction, not a separately learned normalization layer.

The model has `Q` cycle weights shared across channels and phases, plus one scalar boundary gain: `Q + 1` learned parameters. Future cycles are generated recursively. The boundary feature uses only the current history and decays with time constant `τ P`. Its gain is fitted on training data separately for each horizon and clipped to `[0, 1]`. Cycle weights are unconstrained; they are not required to sum to one. The age exponent in the sharing penalty is fixed at 2, and the shared center is eliminated algebraically rather than counted as a learned parameter.

## Data and splits

Use the CSV layout in the [README](../README.md#data). Row indices are zero-based; intervals below are half-open.

| Dataset | Training rows | Validation rows | Test rows |
|---|---|---|---|
| ETTh1, ETTh2 | `[0, 8640)` | `[8640, 11520)` | `[11520, 14400)` |
| ETTm1, ETTm2 | `[0, 34560)` | `[34560, 46080)` | `[46080, 57600)` |
| Weather, Electricity, Traffic, Exchange | `[0, floor(0.7 N))` | `[floor(0.7 N), N - floor(0.2 N))` | `[N - floor(0.2 N), N)` |

`N` is the total number of CSV data rows. ETT rows beyond the specified test endpoint are unused. Proportional splitting uses row-count metadata; numeric data loaded for search and final fitting stop at `val_end`, before test values.

Preserve the original row order, including Weather's repeated timestamp. The `date` column is removed from model inputs and checked for nondecreasing order. Numeric columns must be finite. No cross-split filling or imputation is performed. The loader preserves numeric column order except that a non-ETT `OT` column is moved to the last position.

Fit a per-channel `StandardScaler` on training rows only and reuse its statistics for validation and test. Raw numeric arrays and standardized stored inputs are float32; model coefficients and fitting/scoring accumulations use float64. MSE and MAE are reported in the scale standardized with training statistics, without inverse transformation.

## Original search grid

For each dataset, evaluate every candidate at each of `H = 96, 192, 336, 720`.

| Search field | Values |
|---|---|
| `P` | `4, 8, 24, 96` |
| `Q` | Integers `1..min(29, floor(720 / P))` |
| `λ` | `0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000` |
| `revin` | `0, 1` |
| `τ` | `1, 3, 7, 14, 28` |

There are `29 + 29 + 29 + 7 = 94` period/order pairs and `94 × 16 × 2 × 5 = 15,040` candidates per dataset/horizon. Thus each dataset has 60,160 validation score rows, and the default 32 tasks have 481,280 rows. A candidate with nonfinite validation MSE at any requested horizon is ineligible for shared selection; it is not silently replaced by another grid point.

The original `Q <= 29` restriction and the common training-origin rule below preserve the historical selection experiment. Full-window candidates are not substituted into this search. `--cycles` can restrict the original legal `Q` values, but the resulting grid is a different experiment unless it matches the default grid.

### Search fitting boundaries

For each period, define the common starting index

```text
s_search(P) = P * min(30, 720 // P).
```

It is shared across all search cycle counts for that period; it is not replaced by each candidate's shorter input length. The default starts are 120, 240, 720, and 672 for periods 4, 8, 24, and 96, respectively.

Cycle-weight fitting uses one-step target indices `t` with `s_search <= t < train_end`. Gain fitting uses origins `t` with `s_search <= t <= train_end - H`. Each fitting set uses at most 4,096 evenly spaced legal indices, retaining all when fewer are available. All target values are in the training split. The cycle weights depend on `P`, `Q`, and `λ`; the boundary gain also depends on the horizon, centering, and decay setting. No epochs, learning rate, or early stopping are used.

Validation scoring covers every origin `t = train_end, ..., val_end - H`, all forecast steps, and all channels. Exact pooled squared-error statistics accelerate candidate scoring. The selected candidate is independently refitted and checked using direct model predictions before the final extension.

## Shared validation selection

For a fixed dataset, let `c = (P, Q, λ, revin, τ)`, and let `V_H(c)` be its complete validation MSE at horizon `H`. Compute a separate minimum for each horizon over the original candidate grid:

$$
b_H = \min_c V_H(c).
$$

Select one configuration jointly across the four horizons:

$$
c^* = \arg\min_c \frac{1}{4}
\sum_{H\in\{96,192,336,720\}} \frac{V_H(c)}{b_H}.
$$

Every configuration must have one row for every horizon. The normalizers come only from those validation scores. The implementation substitutes `1e-12` only when a minimum is exactly zero; the historical benchmark minima are positive. Exact ties are resolved by smaller search parameter count, lookback, λ, centering switch, τ, and P, in that order.

Selection is independent for each dataset. All five hyperparameters are shared across its four horizons; the fitted boundary gain remains horizon-specific. Validation MAE is descriptive and does not enter the ranking. This is not a forecast ensemble. Changing the horizon set changes the selection objective.

## Mandatory extension and final fitting

After selecting `c*`, apply this fixed transformation without consulting any further score:

```text
P_final     = P_selected
Q_final     = 720 // P_selected
L_final     = P_selected * Q_final
lambda, revin, tau = their selected values
s_final     = max(s_search(P_selected), L_final)
```

Refit the cycle weights and each horizon's boundary gain on training data, using `s_final` for the first legal fitting index and the same target-end constraints and sample limit as above. Do not concatenate validation data to the training split. Even a restricted `--cycles` search is followed by this extension. If the selected window already equals the final window, the independently refitted model already satisfies the final configuration.

For the historical selected periods, `P = 24` yields `Q_final = 30`, `L_final = 720`, and 31 parameters; `P = 96` yields `Q_final = 7`, `L_final = 672`, and 8 parameters. The original 30-parameter search cap is not a cap on the extended final model. In general, `L_final` is the largest whole-cycle window at most 720, not necessarily exactly 720.

The extended model's validation metrics are recorded for diagnostics and post-run comparison only. They do not change the selected period, λ, centering, or τ, and do not decide whether to retain the extension. There is no second full-window search and no comparison of test scores to choose a model.

## Freeze, test, and aggregation

Freeze final configurations, coefficients, the training scaler, and supporting artifact hashes before loading numeric test values. In an `all` run, all requested datasets are selected and trained before any requested dataset is tested. The test stage requires the freeze manifest and verifies data, source, and model identity. A matching saved test report can be reused without fitting again.

Test every rolling origin `t = val_end, ..., test_end - H`. Its input is exactly `[t - L_final, t)`, and its targets are `[t, t + H)`. Later test origins may use earlier observed test values as causal history. Within a forecast horizon, recursion uses predicted values and never future ground truth. Neither weights nor standardization statistics are updated during test.

Each task's MSE and MAE average over all valid origins, steps, and channels. The reported default mean then weights each of the 32 dataset/horizon tasks equally; it does not pool all datasets' elements. Parameter and lookback averages use the same task weighting.

`selected.json` retains the original search winner. `frozen.json` and `h*_config.json` record the final extended models. `selection_*` output fields refer to the original candidate; unprefixed model fields and `validation_mse` refer to the final model. Search normalizers and the shared selection score retain their original-grid meaning.

## Resume and evidence boundaries

`--resume` permits reuse of matching, hash-verified search artifacts at completed-period boundaries. An incomplete period is recomputed. Final frozen models and matching completed test reports are verified before reuse. Start a new output directory when changing data, source code, or the search specification. A resumed run is not evidence of a fresh uncached search.

The reference CSV and expected-configuration tables are used only for documentation and comparison after testing. They must never supply fitted parameters, preselected configurations, normalization constants, or validation scores to the running pipeline.

The separation of train, validation, and test describes the runtime computation. During method development, test feedback was considered when choosing among validation-selection schemes. This limitation remains part of the evidence and precludes describing the overall development process as completely test-unseen.
