"""Exact shared scoring statistics for every contiguous cycle order.

History matrices are oldest-to-newest. Every order q uses exactly the most
recent q * P input values, including the mean and boundary residual. All
observations supplied by the caller are scored; no origins are sampled here.
This module never fits any forecast coefficient or boundary gain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kernel_scoring import KernelStats, _discounted_suffix, _validate, build_stats


@dataclass
class AllOrderStats:
    raw: KernelStats
    ss: np.ndarray
    st: np.ndarray
    su: np.ndarray
    ee: np.ndarray
    es: np.ndarray
    boundary: dict[float, dict[str, Any]]


@np.errstate(divide='ignore', over='ignore', invalid='ignore')
def build_all_stats(data, origins, H, P, Qmax, taus=(1, 3, 7, 14, 28), *,
                    batch_size=32, max_elements=3_000_000, method="prefix"):
    """Pool all-q moments once, with bounded raw-window temporary storage.

    Returns AllOrderStats; call stats_for_q(result, q) for ordinary KernelStats.
    ``method`` selects the independently checked raw phase-statistics builder.
    Prefix sums and stable reverse exponential filters supply cycle sums and
    weighted sums without expanding complete input/output windows. Dense
    matrix products pool every q simultaneously. All accumulation is float64.
    """
    P, Qmax = int(P), int(Qmax)
    data, origins, H, P, Qmax, lookback = _validate(
        data, origins, H, P, Qmax, P * Qmax)
    if lookback > 720:
        raise ValueError("maximum contiguous lookback exceeds 720")
    taus = tuple(dict.fromkeys(float(t) for t in taus))
    if any(not np.isfinite(t) or t <= 0 for t in taus):
        raise ValueError("boundary time constants must be finite and positive")
    if int(batch_size) < 1 or int(max_elements) < 1:
        raise ValueError("batch_size and max_elements must be positive")
    raw = build_stats(data, origins, H, P, Qmax, boundary_taus=(),
                      lookback=lookback, method=method, batch_size=batch_size)
    lower, upper = int(origins.min()) - lookback, int(origins.max()) + H
    data, origins = data[lower:upper], origins - lower
    Q, K, R, C = raw.Q, raw.K, H % P, data.shape[1]
    result = AllOrderStats(
        raw=raw, ss=np.zeros((Q, Q)), st=np.zeros((Q, Q)),
        su=np.zeros((Q, K)), ee=np.zeros((Q, Q)), es=np.zeros((Q, Q)),
        boundary={t: {"ev": np.zeros((Q, Q)),
                      "evt": np.zeros((Q, Q)), "ey": np.zeros(Q)} for t in taus},
    )
    history_offsets = (np.arange(Q) - Q) * P
    future_offsets = np.arange(K) * P
    future_widths = np.minimum(P, H - future_offsets)
    batch_size = max(1, min(int(batch_size), int(max_elements) // (C * (3 * Q + K))))
    prefix = np.zeros((len(data) + 1, C), dtype=np.float64)
    np.cumsum(data, axis=0, dtype=np.float64, out=prefix[1:])

    def flatten(array):
        return array.transpose(0, 2, 1).reshape(-1, array.shape[1])

    for start in range(0, len(origins), batch_size):
        at = origins[start:start + batch_size]
        positions = at[:, None] + history_offsets
        target_positions = at[:, None] + future_offsets
        # Flatten in the common (origin, channel) order, preserving cycles.
        sums = flatten(prefix[positions + P] - prefix[positions])
        endpoints = flatten(np.asarray(data[positions + P - 1], dtype=np.float64))
        target_sums = flatten(prefix[target_positions + future_widths] - prefix[target_positions])
        result.ss += sums.T @ sums
        result.su += sums.T @ target_sums
        if R:
            tail_sums = flatten(prefix[positions + R] - prefix[positions])
            result.st += sums.T @ tail_sums
        if taus:
            result.ee += endpoints.T @ endpoints
            result.es += endpoints.T @ sums
    del prefix
    # Keep only one entire filtered time series in memory at a time.
    for tau in taus:
        a = np.exp(-1.0 / (tau * P))
        suffix = np.zeros((len(data) + 1, C), dtype=np.float64)
        suffix[:-1] = _discounted_suffix(data, a)
        b = result.boundary[tau]
        for start in range(0, len(origins), batch_size):
            at = origins[start:start + batch_size]
            positions = at[:, None] + history_offsets
            endpoints = flatten(np.asarray(data[positions + P - 1], dtype=np.float64))
            weighted = flatten(a * (suffix[positions] - a ** P * suffix[positions + P]))
            b["ev"] += endpoints.T @ weighted
            if R:
                weighted_tail = flatten(a * (suffix[positions] - a ** R * suffix[positions + R]))
                b["evt"] += endpoints.T @ weighted_tail
            weighted_target = (a * (suffix[at] - a ** H * suffix[at + H])).reshape(-1)
            b["ey"] += endpoints.T @ weighted_target
        del suffix
    # Some Apple Accelerate builds leave spurious floating-point status flags
    # after BLAS calls. Ignore flags locally, but validate every actual output.
    pooled = [result.ss, result.st, result.su, result.ee, result.es,
              raw.gram_full, raw.gram_tail, raw.cross, np.asarray(raw.yy)]
    pooled.extend(value for boundary in result.boundary.values() for value in boundary.values())
    if any(not np.isfinite(value).all() for value in pooled):
        raise ArithmeticError("nonfinite pooled all-order statistics")
    return result


@np.errstate(divide='ignore', over='ignore', invalid='ignore')
def stats_for_q(all_stats: AllOrderStats, q: int) -> KernelStats:
    """Recover exact moments for lookback=q*P from shared pooled moments.

    The returned object is independent of all_stats and may be stamped as
    train/validation by the caller. Its optional provenance is intentionally unset.
    """
    raw = all_stats.raw
    if int(q) != q or not 1 <= int(q) <= raw.Q:
        raise ValueError("q must be an integer in [1,Qmax]")
    q = int(q)
    Q, P, H, K, R = raw.Q, raw.P, raw.H, raw.K, raw.H % raw.P
    selected = slice(Q - q, Q)
    g = np.zeros(Q)
    g[selected] = 1.0 / (q * P)
    r = np.zeros(Q)
    r[selected] = -1.0 / q
    r[-1] += 1.0
    stats = KernelStats(
        gram_full=np.zeros((q + 1, q + 1)),
        gram_tail=np.zeros((q + 1, q + 1)), cross=np.zeros((K, q + 1)),
        yy=raw.yy, count=raw.count, P=P, Q=q, H=H, K=K,
        boundary={}, origins_count=raw.origins_count, channels=raw.channels,
        lookback=q * P,
    )
    stats.gram_full[:q, :q] = raw.gram_full[selected, selected]
    stats.gram_tail[:q, :q] = raw.gram_tail[selected, selected]
    stats.cross[:, :q] = raw.cross[:, selected]
    sg = all_stats.ss @ g
    gnorm = float(g @ sg)
    stats.gram_full[:q, q] = stats.gram_full[q, :q] = sg[selected]
    stats.gram_full[q, q] = P * gnorm
    if R:
        tg = all_stats.st.T @ g
        stats.gram_tail[:q, q] = stats.gram_tail[q, :q] = tg[selected]
        stats.gram_tail[q, q] = R * gnorm
    stats.cross[:, q] = all_stats.su.T @ g
    rnorm = max(0.0, float(r @ all_stats.ee @ r))
    rg = float(r @ all_stats.es @ g)
    for tau, shared in all_stats.boundary.items():
        a = np.exp(-1.0 / (tau * P))
        phase_sum = np.exp(-np.arange(1, P + 1, dtype=np.float64) / (tau * P)).sum()
        full = np.r_[(r @ shared["ev"])[selected], phase_sum * rg]
        if R:
            tail_sum = np.exp(-np.arange(1, R + 1, dtype=np.float64) / (tau * P)).sum()
            tail = np.r_[(r @ shared["evt"])[selected], tail_sum * rg]
        cross = np.empty((K, q + 1))
        for k in range(K):
            cross[k] = a ** (k * P) * (tail if k == K - 1 and R else full)
        stats.boundary[tau] = {
            "cross": cross, "y_cross": float(r @ shared["ey"]),
            "norm": rnorm * float(np.exp(-2 * np.arange(1, H + 1, dtype=np.float64) / (tau * P)).sum()),
        }
    arrays = [stats.gram_full, stats.gram_tail, stats.cross, np.asarray(stats.yy)]
    arrays += [value for boundary in stats.boundary.values() for value in boundary.values()]
    if any(not np.isfinite(value).all() for value in arrays):
        raise ArithmeticError('Nonfinite per-order statistics')
    return stats
