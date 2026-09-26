"""Direct, batched rolling-origin MSE/MAE with an immutable model."""
import torch


@torch.no_grad()
def evaluate(model, data, first_origin, target_end, batch_size=64):
    if first_origin < model.seq_len or target_end > len(data):
        raise ValueError('Evaluation interval lacks its complete history or exceeds the supplied prefix')
    if first_origin + model.pred_len > target_end:
        raise ValueError('Split has no complete forecast window')
    origins = torch.arange(first_origin, target_end - model.pred_len + 1)
    past, future = torch.arange(-model.seq_len, 0), torch.arange(model.pred_len)
    batch_size = max(1, min(batch_size, 2_000_000 // ((model.seq_len + 2 * model.pred_len) * data.shape[1])))
    sse = sae = 0.
    count = 0
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval()
    for at in origins.split(batch_size):
        error = model(data[at[:, None] + past]) - data[at[:, None] + future]
        if not torch.isfinite(error).all():
            raise ArithmeticError('Nonfinite forecast error')
        sse += error.square().sum().item()
        sae += error.abs().sum().item()
        count += error.numel()
    for k, value in model.state_dict().items():
        if not torch.equal(value, before[k]):
            raise RuntimeError('Model changed during evaluation')
    return {'mse': sse / count, 'mae': sae / count, 'origins': len(origins), 'elements': count,
            'first_origin': int(origins[0]), 'last_origin': int(origins[-1]),
            'target_end_exclusive': int(origins[-1]) + model.pred_len}
