"""Same-phase autoregression around the iTransformer backbone."""
import math
from types import SimpleNamespace

import torch
from torch import nn

from model import iTransformer


class Model(nn.Module):
    """Run the official inverted forecaster as a shared same-phase transition."""

    def __init__(self, config):
        super().__init__()
        self.lookback, self.period = config['lookback'], config['period']
        width, depth = config['width'], config['depth']
        self.register_buffer('decay', torch.exp(-torch.arange(1, 97).float() / (0.5 * self.period)))
        self.gain = nn.Parameter(torch.tensor(0.5))
        self.backbone = iTransformer.Model(SimpleNamespace(
            seq_len=self.lookback // self.period, pred_len=1, d_model=width,
            d_ff=2 * width, e_layers=depth, n_heads=4, dropout=0.,
            embed='timeF', freq='h', class_strategy='projection',
            output_attention=False, use_norm=False, fused_phase_encoder=True,
        ))
        nn.init.zeros_(self.backbone.projector.weight)
        nn.init.zeros_(self.backbone.projector.bias)

    def increment(self, lags):
        return self.backbone(lags.transpose(1, 2), None, None, None).float().squeeze(1)

    def phase_predict(self, state):
        b, phases, n, q = state.shape
        packed = state.reshape(b * phases, n, q)
        step = max(8, min(256, 90000 // n))
        prediction = torch.cat([self.increment(x) for x in packed.split(step)])
        return state[..., 0] + prediction.reshape(b, phases, n)

    def fit_phase(self, lags, center):
        x = lags - center.unsqueeze(-1)
        return lags[..., 0] + self.increment(x)

    def forward(self, history, horizon=96):
        b, _, n = history.shape
        p, q = self.period, self.lookback // self.period
        center = history.mean(1, keepdim=True)
        state = (history - center).reshape(b, q, p, n)
        boundary = state[:, -1, -1] - state[:, :, -1].mean(1)
        correction = self.gain * self.decay[:horizon][None, :, None] * boundary[:, None, :]
        lagged = state.flip(1).permute(0, 2, 3, 1)[:, :min(p, horizon)]
        output = []
        for _ in range(math.ceil(horizon / p)):
            new = self.phase_predict(lagged)
            output.append(new)
            lagged = torch.cat((new[..., None], lagged[..., :-1]), -1)
        return torch.cat(output, 1)[:, :horizon] + center + correction
