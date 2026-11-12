"""Phase-aligned forecasting and fitting in the PCA channel subspace."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import eigh, solve


@dataclass(frozen=True)
class Config:
    period: int
    order: int
    center: bool
    tau: float

    @property
    def lookback(self):
        return self.period * self.order


def channel_basis(train, rank):
    values = train.copy()
    values -= values.mean(axis=0)
    covariance = values.T @ values
    width = covariance.shape[0]
    _, axes = eigh(covariance, subset_by_index=(width - rank, width - 1))
    return np.ascontiguousarray(axes[:, ::-1])


def lag_gram(factors, length):
    """Cross-products over all one-step training targets starting at length."""
    data = np.ascontiguousarray(factors.T)
    nfft = next_fast_len(2 * data.shape[1] - 1)
    corr = np.zeros(length + 1)
    blocks = np.array_split(data, (len(data) + 31) // 32)
    for block in blocks:
        power = np.abs(rfft(block, n=nfft, axis=1)) ** 2
        corr += irfft(power.sum(axis=0), n=nfft)[:length + 1]

    # Adjust the FFT autocorrelations for the fixed set of training targets.
    tail = data[:, -length:]
    for lag in range(length):
        corr[lag] -= np.einsum("ct,ct->", tail[:, :length-lag], tail[:, lag:])
    gram = np.empty((length + 1, length + 1))
    gram[0, :] = gram[:, 0] = corr
    delta = tail.T @ tail - data[:, :length].T @ data[:, :length]
    for i in range(length):
        gram[i+1, i+1:] = gram[i, i:length] + delta[i, i:]
        gram[i+1:, i+1] = gram[i+1, i+1:]

    offsets = np.array([0, 1, length // 2, length - 1, length])
    direct = np.zeros((len(offsets), len(offsets)))
    for block in blocks:
        samples = np.stack([block[:, k:k + data.shape[1] - length].ravel() for k in offsets])
        direct += samples @ samples.T
    np.testing.assert_allclose(gram[np.ix_(offsets, offsets)], direct,
                               rtol=1e-10, atol=1e-6)
    return gram


def fit_base(train, config, regularization):
    """Fit TinyCycle's original shared weights with the age-weighted penalty."""
    scale = train.std(axis=0, ddof=1).clip(1e-4)
    gram = lag_gram(train / scale, config.lookback)
    lags = config.lookback - config.period * np.arange(1, config.order + 1)
    matrix = gram[np.ix_(lags, lags)]
    target = gram[lags, -1]
    if regularization and config.order > 1:
        age = (np.arange(1, config.order + 1, dtype=np.float64) / config.order) ** 2
        edges = np.outer(age, age)
        np.fill_diagonal(edges, 0.0)
        penalty = np.diag(edges.sum(axis=1)) - edges
        penalty *= np.diag(matrix).mean() / np.diag(penalty).mean()
        matrix += regularization * penalty
    scale = np.diag(matrix).max()
    weights = solve(matrix / scale, target / scale, assume_a="pos")
    np.testing.assert_allclose(matrix @ weights, target, rtol=1e-9, atol=1e-7)
    return weights


def fit_subspace(factors, config, base, ridge):
    """Fit the subspace weights v with ridge shrinkage toward the frozen w."""
    gram = lag_gram(factors, config.lookback)
    lag = config.lookback - config.period * np.arange(1, config.order + 1)
    matrix = gram[np.ix_(lag, lag)]
    residual = gram[lag, -1] - matrix @ base
    eigenvalues, eigenvectors = eigh(matrix, driver="evd")
    scale = np.trace(matrix) / config.order
    if scale <= 0:
        raise ValueError("The training factors have no variation")
    projected = eigenvectors.T @ residual
    return base + eigenvectors @ (
        projected / np.maximum(eigenvalues + ridge * scale, scale * 1e-12)
    )


def forecast_kernel(config, weights, horizon):
    """Unroll the same-phase recurrence into weights on the observed window."""
    weights = weights[::-1]
    cycles = (horizon + config.period - 1) // config.period
    kernel = np.zeros((cycles, config.order + 1))
    kernel[0, :-1] = weights
    for i in range(1, cycles):
        previous = kernel[i - 1, :-1]
        kernel[i, :-1] = previous[-1] * weights
        kernel[i, 1:-1] += previous[:-1]
    if config.center:
        kernel[:, -1] = 1 - kernel[:, :-1].sum(axis=1)
    return kernel


def predict_components(history, config, kernel, horizon):
    """Return the base forecast and the boundary feature before scaling by g."""
    batch, _, width = history.shape
    past = history.reshape(batch, config.order, config.period, width)
    x = np.ascontiguousarray(past.transpose(0, 2, 3, 1))
    prediction = (x @ np.ascontiguousarray(kernel[:, :-1].T)).transpose(0, 3, 1, 2)
    prediction = prediction.reshape(batch, -1, width)[:, :horizon]
    if config.center:
        response = np.repeat(kernel[:, -1], config.period)[:horizon]
        prediction += history.mean(axis=1, dtype=np.float64)[:, None, :] * response[None, :, None]
    residual = history[:, -1] - past[:, :, -1].mean(axis=1, dtype=np.float64)
    decay = np.exp(-np.arange(1, horizon + 1) / (config.period * config.tau))
    return prediction, residual[:, None, :] * decay[None, :, None]


def predict(history, config, base, loadings, subspace, gain, horizon):
    """Forecast a batch of standardized histories shaped (batch, H, channels)."""
    base_kernel = forecast_kernel(config, base, horizon)
    subspace_kernel = forecast_kernel(config, subspace, horizon)
    baseline, boundary = predict_components(history, config, base_kernel, horizon)
    update, _ = predict_components(history @ loadings, config, subspace_kernel - base_kernel, horizon)
    return baseline + update @ loadings.T + gain * boundary


def check_recurrence(history, config, base, loadings, subspace, gain, horizon):
    """Compare the optimized forecast against direct channel-matrix recursion."""
    centered = history.astype(np.float64)
    mean = centered.mean(axis=1, keepdims=True) if config.center else 0.0
    centered = centered - mean
    past = list(centered.reshape(len(history), config.order, config.period, -1).transpose(1, 0, 2, 3))
    projection = loadings @ loadings.T
    for _ in range((horizon + config.period - 1) // config.period):
        lags = np.stack(past[-config.order:][::-1])
        standard = np.einsum("i,ibpc->bpc", base, lags)
        projected = np.einsum("i,ibpc->bpc", subspace - base, lags) @ projection
        past.append(standard + projected)
    direct = np.concatenate(past[config.order:], axis=1)[:, :horizon] + mean
    last_phase = history[:, config.period - 1::config.period]
    deviation = history[:, -1] - last_phase.mean(axis=1, dtype=np.float64)
    decay = np.exp(-np.arange(1, horizon + 1) / (config.period * config.tau))
    direct += gain * deviation[:, None, :] * decay[None, :, None]
    np.testing.assert_allclose(
        predict(history, config, base, loadings, subspace, gain, horizon), direct,
        rtol=2e-9, atol=2e-9,
    )
