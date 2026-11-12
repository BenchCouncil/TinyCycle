"""Train TinyCycle-PCA from raw PEMS data and evaluate its rolling forecasts."""

import csv
import hashlib
import json
from dataclasses import asdict

import numpy as np

from .model import (
    channel_basis,
    check_recurrence,
    fit_base,
    fit_subspace,
    forecast_kernel,
    predict_components,
)
from .settings import BOUNDARY_FIT_ORIGINS, DATASETS, HORIZONS, MODEL, PCA_RANK, SPLIT_OFFSET

MAX_HORIZON = max(HORIZONS)


def load_data(data_dir, name, dataset):
    path = data_dir / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Pass the directory containing the PEMS NPZ files.")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    if checksum != dataset.md5:
        raise ValueError(f"{name} does not match the original PEMS archive")
    with np.load(path, allow_pickle=False) as archive:
        raw = np.ascontiguousarray(archive["data"][:, :, 0])
    if raw.shape != dataset.shape or not np.isfinite(raw).all():
        raise ValueError(f"Unexpected shape or nonfinite observations for {name}")
    return raw


def batches(origins, width):
    size = max(1, min(64, 1_200_000 // (width * MODEL.lookback)))
    return np.array_split(origins, (len(origins) + size - 1) // size)


def train_scaler(raw_train):
    train = raw_train.astype(np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def fit_boundary(train, base):
    """Fit one output-side gain per horizon using training windows only."""
    origins = np.unique(np.linspace(
        MODEL.lookback, len(train) - MAX_HORIZON, BOUNDARY_FIT_ORIGINS
    ).round().astype(int))
    offsets = np.arange(-MODEL.lookback, 0)
    leads = np.arange(MAX_HORIZON)
    kernel = forecast_kernel(MODEL, base, MAX_HORIZON)
    numerator, denominator = np.zeros((2, len(HORIZONS)))
    for at in batches(origins, train.shape[1]):
        history = train[at[:, None] + offsets]
        prediction, feature = predict_components(history, MODEL, kernel, MAX_HORIZON)
        error = prediction - train[at[:, None] + leads]
        for i, h in enumerate(HORIZONS):
            e, f = error[:, :h], feature[:, :h]
            numerator[i] -= np.einsum("thc,thc->", e, f)
            denominator[i] += np.einsum("thc,thc->", f, f)
    return np.clip(numerator / np.maximum(denominator, 1e-30), 0.0, 1.0)


def train_model(train, dataset):
    train64 = train.astype(np.float64)
    base = fit_base(train64, MODEL, dataset.base_regularization)
    gains = fit_boundary(train, base)
    loadings = channel_basis(train64, PCA_RANK)
    subspace = fit_subspace(train64 @ loadings, MODEL, base, dataset.pca_ridge)
    return dict(base=base, gains=gains, loadings=loadings, subspace=subspace)


def score_model(data, first, base, gains, loadings, subspace):
    """Evaluate TinyCycle-PCA at every valid test origin on all channels."""
    origins = np.arange(first, len(data) - min(HORIZONS) + 1)
    factors = data.astype(np.float64) @ loadings
    base_kernel = forecast_kernel(MODEL, base, MAX_HORIZON)
    update_kernel = forecast_kernel(MODEL, subspace, MAX_HORIZON) - base_kernel
    offsets = np.arange(-MODEL.lookback, 0)
    leads = np.arange(MAX_HORIZON)
    errors = np.zeros((len(HORIZONS), 2))
    counts = np.zeros(len(HORIZONS), dtype=np.int64)
    for at in batches(origins, data.shape[1]):
        history = at[:, None] + offsets
        prediction, boundary = predict_components(data[history], MODEL, base_kernel, MAX_HORIZON)
        update, _ = predict_components(factors[history], MODEL, update_kernel, MAX_HORIZON)
        target = data[np.minimum(at[:, None] + leads, len(data) - 1)]
        error = prediction - target + update @ loadings.T
        for i, h in enumerate(HORIZONS):
            n = np.searchsorted(at, len(data) - h, side="right")
            residual = error[:n, :h] + gains[i] * boundary[:n, :h]
            errors[i] += np.einsum("thc,thc->", residual, residual), np.abs(residual).sum()
            counts[i] += residual.size
    np.testing.assert_array_equal(
        counts, [(len(data) - h - first + 1) * h * data.shape[1] for h in HORIZONS]
    )
    metrics = errors / counts[:, None]
    if not np.isfinite(metrics).all():
        raise ValueError("Nonfinite test metrics")
    return metrics


def run_dataset(name, data_dir, output_dir):
    dataset = DATASETS[name]
    raw = load_data(data_dir, name, dataset)
    train_end, val_end = int(len(raw) * 0.6), int(len(raw) * 0.8)
    mean, scale = train_scaler(raw[:train_end])
    train = ((raw[:train_end].astype(np.float64) - mean) / scale).astype(np.float32)
    print(f"{name}: training scaler, original TinyCycle and PCA weights", flush=True)
    fitted = train_model(train, dataset)
    check_recurrence(
        train[-MODEL.lookback:][None], MODEL, fitted["base"], fitted["loadings"],
        fitted["subspace"], fitted["gains"][-1], MAX_HORIZON,
    )
    del train
    data = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
    first = val_end + SPLIT_OFFSET
    print(f"{name}: testing {len(data) - first} time steps", flush=True)
    metrics = score_model(data, first, **fitted)

    folder = output_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "weights.npz", **fitted, mean=mean, scale=scale)
    settings = dict(
        model=asdict(MODEL), pca_rank=PCA_RANK, pca_source="levels",
        base_regularization=dataset.base_regularization, pca_ridge=dataset.pca_ridge,
        bounds=[train_end, val_end, len(data)], horizons=HORIZONS,
    )
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    rows = [
        dict(dataset=name, horizon=h, test_mse=mse, test_mae=mae,
             test_first_origin=first, test_origins=len(data) - h - first + 1)
        for h, (mse, mae) in zip(HORIZONS, metrics, strict=True)
    ]
    write_csv(folder / "results.csv", rows)
    print(f"{name}: MSE={metrics[:, 0].mean():.6f}, MAE={metrics[:, 1].mean():.6f}", flush=True)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir, rows, names):
    write_csv(output_dir / "per_task.csv", rows)
    summary = []
    for name in (*names, "Average"):
        selected = rows if name == "Average" else [row for row in rows if row["dataset"] == name]
        summary.append(dict(dataset=name, **{
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in ("test_mse", "test_mae")
        }))
    write_csv(output_dir / "summary.csv", summary)
    print("\n| Dataset | MSE | MAE |\n|---|---:|---:|")
    for row in summary:
        print(f"| {row['dataset']} | {row['test_mse']:.3f} | {row['test_mae']:.3f} |")
