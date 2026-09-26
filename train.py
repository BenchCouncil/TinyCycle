"""Fit age-weighted cycle sharing and boundary correction on training data only."""

import math

import torch


def sample_times(start, stop, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError("fit_samples must be a positive integer")
    if stop <= start:
        raise ValueError("training split has no eligible fitting origins")
    times = torch.arange(start, stop)
    if len(times) > limit:
        indices = torch.linspace(0, len(times) - 1, limit).round().long().unique()
        times = times[indices]
    return times


def solve_weights(gram, cross):
    """Numerically checked, unconstrained least squares on a small system."""
    system = (gram + gram.T) * 0.5
    scale = system.diagonal().abs().max().clamp_min(1e-12)
    system, target = system / scale, cross / scale
    rtol = torch.finfo(torch.float64).eps * len(system) * 10
    spectrum = torch.linalg.eigvalsh(system).abs()
    if spectrum.min() <= spectrum.max() * rtol:
        return torch.linalg.pinv(system, rtol=rtol, hermitian=True) @ target
    solution, info = torch.linalg.solve_ex(system, target)
    residual = torch.linalg.vector_norm(system @ solution - target)
    if info or not torch.isfinite(solution).all() or residual > 1e-8 * max(
        1.0, float(torch.linalg.vector_norm(target))
    ):
        solution = torch.linalg.pinv(system, rtol=rtol, hermitian=True) @ target
    return solution


def sharing_penalty(gram):
    """Older weights share more strongly; the shared mean is eliminated analytically."""
    q = len(gram)
    if q == 1:
        return torch.zeros_like(gram)
    age = torch.arange(1, q + 1, dtype=torch.float64, device=gram.device) / q
    a = age.square()
    edges = a[:, None] * a[None, :]
    edges.fill_diagonal_(0)
    matrix = torch.diag(edges.sum(1)) - edges
    return matrix / matrix.diagonal().mean().clamp_min(1e-12)


@torch.no_grad()
def fit_model(model, train, args):
    if train.ndim != 2 or train.shape[1] < 1:
        raise ValueError("train must have shape [time, channels]")
    if not math.isfinite(args.regularization) or args.regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if train.device != model.theta.device:
        raise ValueError("training data and model must be on the same device")
    order, device = model.order, train.device
    # Preserve the original selection grid's origins; an extended final window
    # must also fit fully before the first target (especially for P=4 or P=8).
    start = max(model.seq_len, model.period * min(30, 720 // model.period))
    times = sample_times(start, len(train), args.fit_samples).to(device)
    scale = train.std(0, unbiased=True).clamp_min(1e-4).double()
    gram = torch.zeros(order, order, dtype=torch.float64, device=device)
    cross = torch.zeros(order, dtype=torch.float64, device=device)
    lags = model.period * torch.arange(1, order + 1, device=device)
    batch = max(1, min(256, 2_000_000 // (train.shape[1] * order)))
    for selected in times.split(batch):
        x = train[selected[:, None] - lags].permute(0, 2, 1).double()
        x.div_(scale[None, :, None])
        y = train[selected].double() / scale[None, :]
        x, y = x.reshape(-1, order), y.reshape(-1)
        gram.add_(x.T @ x)
        cross.add_(x.T @ y)
    penalty_scale = gram.diagonal().mean().clamp_min(1e-12)
    regularized = gram + args.regularization * penalty_scale * sharing_penalty(gram)
    model.theta.copy_(solve_weights(regularized, cross))

    model.boundary_gain.zero_()
    times = sample_times(start, len(train) - model.pred_len + 1, args.fit_samples).to(device)
    history_offsets = torch.arange(-model.seq_len, 0, device=device)
    future_offsets = torch.arange(model.pred_len, device=device)
    batch = max(1, min(args.batch_size, 2_000_000 // ((model.seq_len + 2 * model.pred_len) * train.shape[1])))
    numerator = denominator = 0.0
    for selected in times.split(batch):
        history = train[selected[:, None] + history_offsets]
        target = train[selected[:, None] + future_offsets]
        feature = model.boundary_features(history)
        numerator += ((target - model(history)) * feature).sum().item()
        denominator += feature.square().sum().item()
    gain = numerator / denominator if denominator > 1e-30 else 0.0
    model.boundary_gain.fill_(min(1.0, max(0.0, gain)))
    return model
