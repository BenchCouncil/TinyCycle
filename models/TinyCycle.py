"""TinyCycle with consecutive cycle weights and age-weighted sharing in training."""

import math

import torch
from torch import nn


class Model(nn.Module):
    """One recurrent architecture: Q cycle weights plus one boundary gain."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len, self.pred_len = configs.seq_len, configs.pred_len
        self.period = configs.period
        if type(self.period) is not int or not 1 <= self.period <= 720:
            raise ValueError("period must be an integer in [1, 720]")
        if (type(self.seq_len) is not int or not self.period <= self.seq_len <= 720
                or self.seq_len % self.period):
            raise ValueError("seq_len must equal period * Q and be <=720")
        if type(self.pred_len) is not int or self.pred_len < 1:
            raise ValueError("pred_len must be a positive integer")
        self.order = self.seq_len // self.period
        self.use_revin = bool(configs.revin)
        self.tau_cycles = configs.tau_cycles
        if not math.isfinite(self.tau_cycles) or self.tau_cycles <= 0:
            raise ValueError("tau_cycles must be finite and positive")
        self.theta = nn.Parameter(torch.zeros(self.order, dtype=torch.float64))
        self.boundary_gain = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def cycle_weights(self):
        """Coefficients for the nearest Q cycles: lags P, 2P, ..., QP."""
        return self.theta

    def forecast_kernel(self):
        weights = self.cycle_weights()
        state = list(torch.eye(self.order, dtype=weights.dtype, device=weights.device).unbind(0))
        future = []
        for _ in range((self.pred_len + self.period - 1) // self.period):
            lagged = torch.stack(list(reversed(state[-self.order:])))
            new = torch.einsum("o,oi->i", weights, lagged)
            future.append(new)
            state.append(new)
        return torch.stack(future)

    def boundary_features(self, history):
        history = history[:, -self.seq_len:].double()
        past = history.reshape(history.shape[0], self.order, self.period, history.shape[2])
        residual = history[:, -1] - past[:, :, -1, :].mean(1)
        decay = torch.exp(
            -(torch.arange(self.pred_len, dtype=history.dtype, device=history.device) + 1)
            / (self.tau_cycles * self.period)
        )
        return residual[:, None, :] * decay[None, :, None]

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] < self.seq_len:
            raise ValueError("x must have shape [batch, history >= seq_len, channels]")
        # Mean and boundary corrections use this same bounded input window.
        history = x[:, -self.seq_len:].double()
        batch, _, channels = history.shape
        kernel = self.forecast_kernel()
        past = history.reshape(batch, self.order, self.period, channels)
        prediction = torch.einsum("kq,bqpc->bkpc", kernel, past).reshape(
            batch, -1, channels
        )[:, :self.pred_len]
        if self.use_revin:
            sums = kernel.sum(-1).repeat_interleave(self.period)[:self.pred_len]
            prediction = prediction + (1 - sums)[None, :, None] * history.mean(1)[:, None, :]
        return prediction + self.boundary_gain * self.boundary_features(history)
