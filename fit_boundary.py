"""Fit the boundary gain only from explicitly bounded training statistics.

Validation scoring consumes an already frozen coefficient. No gain is fitted
or optimized using validation or test targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class _TrainingBounds:
    train_end: int
    data_rows: int
    origin_min: int
    origin_max: int
    origins_count: int
    horizon: int
    lookback: int
    channels: int
    count: int


def mark_train_stats(stats, *, train_end: int, data_rows: int, origins):
    """Stamp statistics built from ``data[:train_end]`` and these train origins.

    Example::

        train = data[:train_end]
        stats = build_stats(train, train_origins, H, P, Q, boundary_taus=TAUS)
        mark_train_stats(stats, train_end=len(train), data_rows=len(train),
                         origins=train_origins)

    Indices are relative to this training prefix. Validation/test arrays cannot
    be passed as additional rows: ``data_rows`` must equal ``train_end``. The
    caller remains responsible for passing the actual train prefix to the
    statistics builder; a provenance label cannot inspect its discarded data.
    """
    if getattr(stats, "split", "train") != "train":
        raise ValueError("cannot relabel non-training statistics as train")
    raw_origins = _numpy(origins)
    if raw_origins.ndim != 1 or raw_origins.size == 0:
        raise ValueError("training origins must be a nonempty vector")
    if not np.issubdtype(raw_origins.dtype, np.integer):
        raise ValueError("training origins must have integer dtype")
    origins = raw_origins.astype(np.int64, copy=False)
    if int(train_end) != train_end or int(data_rows) != data_rows:
        raise ValueError("train_end and data_rows must be integer row counts")
    bounds = _TrainingBounds(
        int(train_end), int(data_rows), int(origins.min()), int(origins.max()),
        int(len(origins)), int(stats.H), int(stats.lookback),
        int(stats.channels), int(stats.count),
    )
    _validate_bounds(stats, bounds)
    existing = getattr(stats, "_boundary_training_bounds", None)
    if existing is not None and existing != bounds:
        raise ValueError("training statistics already have different provenance")
    stats.split = "train"
    stats._boundary_training_bounds = bounds
    return stats


def _validate_bounds(stats, bounds):
    if not isinstance(bounds, _TrainingBounds):
        raise ValueError("training statistics must be stamped by mark_train_stats")
    if bounds.train_end != bounds.data_rows or bounds.train_end <= 0:
        raise ValueError("gain fitting requires an already sliced training prefix")
    if bounds.origin_min < bounds.lookback:
        raise ValueError("a training origin lacks its complete input window")
    if bounds.origin_max + bounds.horizon > bounds.train_end:
        raise ValueError("a training forecast target crosses train_end")
    if bounds.origin_max < bounds.origin_min or bounds.origins_count < 1:
        raise ValueError("invalid training origin bounds")
    expected = (bounds.horizon, bounds.lookback, bounds.channels,
                bounds.origins_count, bounds.count)
    current = (int(stats.H), int(stats.lookback), int(stats.channels),
               int(stats.origins_count), int(stats.count))
    if expected != current:
        raise ValueError("statistics dimensions changed after training provenance was set")
    if bounds.count != bounds.origins_count * bounds.horizon * bounds.channels:
        raise ValueError("training target count does not match complete forecast windows")


def _boundary_terms(stats, kernel, tau):
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("boundary tau must be finite and positive")
    kernel = np.asarray(_numpy(kernel), dtype=np.float64)
    if kernel.shape == (int(stats.K), int(stats.Q)):
        kernel = np.pad(kernel, ((0, 0), (0, 1)))
    if kernel.shape != stats.cross.shape or not np.isfinite(kernel).all():
        raise ValueError("kernel has an invalid shape or nonfinite coefficients")
    if tau not in stats.boundary:
        raise ValueError("statistics do not include the requested boundary tau")
    boundary = stats.boundary[tau]
    norm = float(boundary["norm"])
    if not np.isfinite(norm) or norm < 0:
        raise ValueError("invalid boundary feature squared norm")
    residual_dot = float(np.sum(kernel * boundary["cross"]) - boundary["y_cross"])
    if not np.isfinite(residual_dot):
        raise ValueError("invalid boundary residual cross-product")
    return residual_dot, norm


def fit_gain_train(train_stats, kernel, tau) -> float:
    """Fit one scalar on verified train windows, constrained to [0, 1]."""
    if getattr(train_stats, "split", None) != "train":
        raise ValueError("boundary gain fitting accepts explicitly marked train statistics only")
    _validate_bounds(train_stats, getattr(train_stats, "_boundary_training_bounds", None))
    residual_dot, norm = _boundary_terms(train_stats, kernel, tau)
    if norm <= 1e-30:
        return 0.0
    return float(np.clip(-residual_dot / norm, 0.0, 1.0))


def score_fixed_gain(eval_stats, kernel, base_mse, tau, gain) -> float:
    """Evaluate the supplied frozen gain; never estimate or clip it on test."""
    gain = float(gain)
    if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("gain must already be fitted and lie in [0, 1]")
    base_mse = float(base_mse)
    if np.isnan(base_mse) or base_mse < 0:
        raise ValueError("base_mse must be a nonnegative base-kernel MSE")
    if not np.isfinite(base_mse):
        return float("inf")
    residual_dot, norm = _boundary_terms(eval_stats, kernel, tau)
    count = int(eval_stats.count)
    if count <= 0:
        raise ValueError("evaluation statistics must contain target elements")
    sse = base_mse * count + 2 * gain * residual_dot + gain * gain * norm
    if not np.isfinite(sse):
        return float("inf")
    tolerance = 1e-8 * max(1.0, float(eval_stats.yy))
    if sse < -tolerance:
        raise ArithmeticError("negative fixed-gain SSE: check base_mse and statistics")
    return max(0.0, float(sse)) / count

