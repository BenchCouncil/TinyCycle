"""Exact CPU sufficient statistics for shared seasonal forecast kernels.

All recorded metrics cover every supplied origin, horizon step and channel.
The initial-cycle axis is oldest to newest; AR coefficients use lag 1 first.
The final feature is the mean of the complete ``lookback`` input window.
``gram_full`` sums one complete P-phase block; ``gram_tail`` sums H % P
phases. ``cross[k]`` already uses the appropriate width for future cycle k.
Statistics contain only the interval explicitly supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class KernelStats:
    gram_full: np.ndarray
    gram_tail: np.ndarray
    cross: np.ndarray
    yy: float
    count: int
    P: int
    Q: int
    H: int
    K: int
    boundary: dict[float, dict[str, Any]]
    origins_count: int
    channels: int
    lookback: int


def _validate(data, origins, H, P, Qmax, lookback):
    data = _numpy(data)
    origins = _numpy(origins).astype(np.int64, copy=False)
    H, P, lookback = int(H), int(P), int(lookback)
    Q = lookback // P if Qmax is None else int(Qmax)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("data must have shape [time, channel]")
    if origins.ndim != 1 or origins.size == 0:
        raise ValueError("origins must be a nonempty vector")
    if H < 1 or P < 1 or Q < 1 or Q * P > lookback:
        raise ValueError("invalid horizon, period, order, or lookback")
    if origins.min() < lookback or origins.max() + H > len(data):
        raise ValueError("origin outside complete input / target bounds")
    if not np.isfinite(data).all():
        raise ValueError("data contains non-finite entries")
    return data, origins, H, P, Q, lookback


def _new_stats(data, origins, H, P, Q, lookback, taus):
    K, D = (H + P - 1) // P, Q + 1
    return KernelStats(
        np.zeros((D, D)), np.zeros((D, D)), np.zeros((K, D)),
        0.0, int(len(origins) * H * data.shape[1]), P, Q, H, K,
        {float(t): {"cross": np.zeros((K, D)), "y_cross": 0.0,
                    "norm": 0.0} for t in taus},
        len(origins), data.shape[1], lookback,
    )


def _boundary_residual(data, origins, P, Q):
    mean = np.zeros((len(origins), data.shape[1]), dtype=np.float64)
    for age in range(Q):
        mean += data[origins - 1 - age * P]
    return np.asarray(data[origins - 1], dtype=np.float64) - mean / Q


def _window_correlation_sums(z, origins, shifts, width):
    """Sum z over width-phase windows; optimize a consecutive origin run."""
    shifts = np.asarray(shifts, dtype=np.int64)
    prefix = np.concatenate(([0.0], np.cumsum(z, dtype=np.float64)))
    if len(origins) == 1 or np.all(np.diff(origins) == 1):
        second = np.concatenate(([0.0], np.cumsum(prefix, dtype=np.float64)))
        low = origins[0] + shifts
        high = origins[-1] + shifts + 1
        return (second[high + width] - second[low + width]
                - second[high] + second[low])
    out = np.empty(len(shifts), dtype=np.float64)
    # Irregular (usually sampled training) origins: bounded temporary memory.
    chunk = max(1, 1_000_000 // len(origins))
    for start in range(0, len(shifts), chunk):
        indices = shifts[start:start + chunk, None] + origins[None, :]
        out[start:start + chunk] = np.sum(
            prefix[indices + width] - prefix[indices], axis=1)
    return out


def _discounted_suffix(data, a):
    """Stable reverse exponential filter; no growing exponential powers."""
    try:
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -a], data[::-1], axis=0)[::-1]
    except ImportError:
        out = np.empty(data.shape, dtype=np.float64)
        state = np.zeros(data.shape[1], dtype=np.float64)
        for i in range(len(data) - 1, -1, -1):
            state = data[i] + a * state
            out[i] = state
        return out


def build_stats(data, origins, H, P, Qmax=None,
                boundary_taus: Iterable[float] = (1, 3, 7, 14, 28), *,
                lookback=720, method="prefix", batch_size=32):
    """Build double-precision full-split quadratics without candidate forecasts.

    ``method='batch'`` is a straightforward independent implementation useful
    for checks and a fallback. The default uses lagged time correlations to
    avoid repeatedly forming equivalent input windows and Q-by-Q products.
    Boundary residuals always use all Q cycles, including zero AR taps.
    """
    data, origins, H, P, Q, lookback = _validate(
        data, origins, H, P, Qmax, lookback)
    # Nothing outside this interval can contribute. In particular, a test
    # split near the end of a long series need not rescan its training prefix.
    lower, upper = int(origins.min()) - lookback, int(origins.max()) + H
    data, origins = data[lower:upper], origins - lower
    taus = tuple(float(t) for t in boundary_taus)
    if any(t <= 0 for t in taus):
        raise ValueError("boundary taus must be positive")
    if method == "batch":
        return _build_batch(data, origins, H, P, Q, lookback, taus, batch_size)
    if method != "prefix":
        raise ValueError("method must be 'prefix' or 'batch'")
    stats = _new_stats(data, origins, H, P, Q, lookback, taus)
    K, R = stats.K, H % P
    # A matrix entry is a sum of channel dot-products at one temporal lag.
    # Each distinct lag is computed once; prefix sums supply shifted windows.
    for lag in range(Q + K):
        shift = lag * P
        if shift >= len(data):
            break
        left = data if shift == 0 else data[:-shift]
        right = data if shift == 0 else data[shift:]
        z = np.einsum("tc,tc->t", left, right, dtype=np.float64,
                      optimize=False)
        ii, jj, kinds = [], [], []
        # Symmetric history Gram, including its diagonal.
        if lag < Q:
            for i in range(Q - lag):
                ii.append(i); jj.append(i + lag); kinds.append(0)
        # Input/target cross-products.
        for k in range(K):
            i = Q + k - lag
            if 0 <= i < Q:
                ii.append(i); jj.append(Q + k); kinds.append(1)
        if lag == 0:
            for k in range(K):
                ii.append(Q + k); jj.append(Q + k); kinds.append(2)
        if not ii:
            continue
        ii, jj, kinds = map(np.asarray, (ii, jj, kinds))
        widths = np.where((jj == Q + K - 1) & (kinds != 0) & (R > 0), R, P)
        vals = np.empty(len(ii))
        for width in np.unique(widths):
            mask = widths == width
            vals[mask] = _window_correlation_sums(
                z, origins, (ii[mask] - Q) * P, int(width))
        gm = kinds == 0
        stats.gram_full[ii[gm], jj[gm]] = vals[gm]
        stats.gram_full[jj[gm], ii[gm]] = vals[gm]
        if R and np.any(gm):
            tail = _window_correlation_sums(z, origins, (ii[gm] - Q) * P, R)
            stats.gram_tail[ii[gm], jj[gm]] = tail
            stats.gram_tail[jj[gm], ii[gm]] = tail
        cm = kinds == 1
        stats.cross[jj[cm] - Q, ii[cm]] = vals[cm]
        stats.yy += float(vals[kinds == 2].sum())

    # Global input mean is shared across the phases but depends on origin.
    prefix = np.zeros((len(data) + 1, data.shape[1]), dtype=np.float64)
    np.cumsum(data, axis=0, dtype=np.float64, out=prefix[1:])
    g = (prefix[origins] - prefix[origins - lookback]) / lookback
    gnorm = np.einsum("oc,oc->", g, g)
    stats.gram_full[Q, Q] = P * gnorm
    stats.gram_tail[Q, Q] = R * gnorm
    for i in range(Q + K):
        positions = origins + (i - Q) * P
        width = R if i == Q + K - 1 and R else P
        summed = prefix[positions + width] - prefix[positions]
        value = np.einsum("oc,oc->", g, summed)
        if i < Q:
            stats.gram_full[i, Q] = stats.gram_full[Q, i] = value
            if R:
                summed = prefix[positions + R] - prefix[positions]
                value = np.einsum("oc,oc->", g, summed)
                stats.gram_tail[i, Q] = stats.gram_tail[Q, i] = value
        else:
            stats.cross[i - Q, Q] = value
    del prefix

    if taus:
        residual = _boundary_residual(data, origins, P, Q)
        residual_norm = float(np.einsum("oc,oc->", residual, residual))
        gr = float(np.einsum("oc,oc->", residual, g))
        for tau in taus:
            a = np.exp(-1.0 / (tau * P))
            suffix = np.zeros((len(data) + 1, data.shape[1]), dtype=np.float64)
            suffix[:-1] = _discounted_suffix(data, a)
            boundary = stats.boundary[tau]
            history_full, history_tail = np.zeros(Q + 1), np.zeros(Q + 1)
            phase_sum = a * (-np.expm1(P * np.log(a))) / (1 - a)
            history_full[Q] = phase_sum * gr
            if R:
                history_tail[Q] = a * (-np.expm1(R * np.log(a))) / (1 - a) * gr
            for i in range(Q + K):
                positions = origins + (i - Q) * P
                width = R if i == Q + K - 1 and R else P
                summed = a * (suffix[positions] - a ** width * suffix[positions + width])
                value = np.einsum("oc,oc->", residual, summed)
                if i < Q:
                    history_full[i] = value
                    if R:
                        summed = a * (suffix[positions] - a ** R * suffix[positions + R])
                        history_tail[i] = np.einsum("oc,oc->", residual, summed)
                else:
                    boundary["y_cross"] += float(a ** ((i - Q) * P) * value)
            for k in range(K):
                raw = history_tail if k == K - 1 and R else history_full
                boundary["cross"][k] = a ** (k * P) * raw
            boundary["norm"] = residual_norm * float(
                np.exp(-2 * np.arange(1, H + 1, dtype=float) / (tau * P)).sum())
            del suffix
    return stats


def _build_batch(data, origins, H, P, Q, lookback, taus, batch_size):
    stats = _new_stats(data, origins, H, P, Q, lookback, taus)
    R = H % P
    for start in range(0, len(origins), batch_size):
        selected = origins[start:start + batch_size]
        hist = np.asarray(data[selected[:, None] - lookback + np.arange(lookback)], dtype=np.float64)
        g = hist.mean(axis=1)
        initial = hist[:, -Q * P:].reshape(len(selected), Q, P, data.shape[1])
        x = np.concatenate((initial, np.broadcast_to(g[:, None, None, :],
             (len(selected), 1, P, data.shape[1]))), axis=1)
        flat = x.transpose(0, 2, 3, 1).reshape(-1, Q + 1)
        stats.gram_full += np.einsum("ni,nj->ij", flat, flat, optimize=False)
        if R:
            flat_tail = x[:, :, :R].transpose(0, 2, 3, 1).reshape(-1, Q + 1)
            stats.gram_tail += np.einsum("ni,nj->ij", flat_tail, flat_tail, optimize=False)
        residual = _boundary_residual(data, selected, P, Q) if taus else None
        for k in range(stats.K):
            width = min(P, H - k * P)
            y = np.asarray(data[selected[:, None] + k * P + np.arange(width)], dtype=np.float64)
            stats.cross[k] += np.einsum("bqpc,bpc->q", x[:, :, :width], y)
            stats.yy += float(np.square(y).sum())
            for tau in taus:
                decay = np.exp(-(k * P + np.arange(1, width + 1)) / (tau * P))
                b = residual[:, None, :] * decay[None, :, None]
                stats.boundary[tau]["cross"][k] += np.einsum("bqpc,bpc->q", x[:, :, :width], b)
                stats.boundary[tau]["y_cross"] += float(np.sum(y * b))
                stats.boundary[tau]["norm"] += float(np.square(b).sum())
    return stats


def ar_kernel(weights, H, P, Q=None):
    """Expand lag-1-first AR coefficients into [future_cycle, old_to_new]."""
    weights = np.asarray(_numpy(weights), dtype=np.float64)
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("weights must be a nonempty vector")
    Q = weights.size if Q is None else int(Q)
    if weights.size > Q:
        raise ValueError("coefficient vector exceeds history order")
    weights = np.pad(weights, (0, Q - weights.size))
    state = list(np.eye(Q))
    future = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range((int(H) + int(P) - 1) // int(P)):
            new = weights @ np.stack(state[-Q:][::-1])
            future.append(new)
            state.append(new)
    return np.stack(future)


def score_kernel(stats, kernel):
    """Evaluate a frozen base kernel; this function never optimizes coefficients."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.shape != stats.cross.shape:
        raise ValueError('Kernel shape does not match statistics')
    if not np.isfinite(kernel).all():
        return {'mse': float('inf')}
    full = stats.K - int(bool(stats.H % stats.P))
    with np.errstate(over='ignore', invalid='ignore'):
        sse = stats.yy - 2 * np.sum(kernel * stats.cross)
        if full:
            sse += np.einsum('ki,ij,kj->', kernel[:full], stats.gram_full, kernel[:full])
        if stats.H % stats.P:
            sse += kernel[-1] @ stats.gram_tail @ kernel[-1]
    if not np.isfinite(sse):
        return {'mse': float('inf')}
    if sse < -1e-8 * max(1., stats.yy):
        raise ArithmeticError('Negative SSE: inspect statistics or numerical conditioning')
    return {'mse': max(0., float(sse)) / stats.count}
