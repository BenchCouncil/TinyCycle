# TinyCycle-PCA on PEMS

Train and test TinyCycle-PCA from the original PEMS files to reproduce the PCA channel-modeling results in the paper. The experiment uses the hyperparameters in [settings.py](tinycycle_pca/settings.py). Each run estimates the normalization statistics, base TinyCycle weights, boundary gains, PCA loadings, and subspace weights from the training data.

## Running the experiment

Use Python 3.13 and run the following commands from this directory. Set `--data-dir` to the directory containing `PEMS03.npz`, `PEMS04.npz`, `PEMS07.npz`, and `PEMS08.npz`. You can use the project's PEMS data files directly.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tinycycle_pca --data-dir /path/to/pems --output-dir runs/fixed
```

Add `--datasets PEMS08` to run only PEMS08. The run writes dataset-average MSE and MAE to `summary.csv` and per-horizon metrics to `per_task.csv`. Each dataset subdirectory contains the fitted `weights.npz` and the experiment settings.

## Training and evaluation protocol

- Split the data chronologically into 60% training, 20% validation, and 20% test. Estimate each channel's mean and standard deviation from the training split.
- Use a lookback of 720, a period of 1, and a PCA rank of 4. Fit PCA on standardized flow. The base weight regularization is 0.003 for PEMS07, and the PCA subspace weight shrinkage is 0.001 for PEMS04. Each setting is zero for the other datasets.
- Fit the base TinyCycle weights on the training split. Fit a separate boundary gain for each horizon using 512 evenly spaced training forecast origins, then fit the PCA subspace weights. The channel mapping for lag `i` is `A_i = w_i I + (v_i - w_i) U Uᵀ`.
- Start rolling forecasts 96 steps into the test split, with the first target index equal to the validation split's end index plus 96. Evaluate horizons of 12, 24, 48, and 96 steps on the standardized flow scale.

## Files

- [settings.py](tinycycle_pca/settings.py) defines the fixed hyperparameters and data file specifications.
- [model.py](tinycycle_pca/model.py) implements weight fitting, PCA, and same-phase recurrence.
- [experiment.py](tinycycle_pca/experiment.py) handles data loading, training, and rolling evaluation.
- [__main__.py](tinycycle_pca/__main__.py) provides the command-line entry point.
