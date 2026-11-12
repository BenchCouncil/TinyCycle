"""Search TinyCycle-plug on PEMS validation data, then test the selected models."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from data_provider.pems_phase import DATASETS, input_at, load_data
from model.TinyCyclePhase import Model

HORIZONS = (12, 24, 48, 96)
TAUS = (3, 6, 12, 24, 48, 96, 144, 288)
STAGES = ((192, 1., 64), (640, .65, 160), (1600, .30, 320))
LOOKBACK = 720
PERIODS = (4, 8, 12, 18, 24, 48, 72, 144)
WIDTHS = (32, 64, 128, 256)
DEPTHS = (1, 2, 3, 4)
RATES = (.0001, .0003, .001, .003)
DECAYS = (.0001, .001)
SEED = 2023


def trial_configs():
    for k, period in enumerate(PERIODS):
        for i, width in enumerate(WIDTHS):
            for j, depth in enumerate(DEPTHS):
                if period < 24:
                    if (k + i + j) % 2:
                        continue
                    rate = RATES[(k + i + 2*j) % len(RATES)]
                    decay = DECAYS[(k + i + j//2) % len(DECAYS)]
                else:
                    index = k - 4
                    rate = RATES[(i + j + index) % len(RATES)]
                    decay = DECAYS[(i + j + index//2) % len(DECAYS)]
                yield dict(lookback=LOOKBACK, period=period, width=width, depth=depth,
                           lr=rate, weight_decay=decay)


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


@torch.inference_mode()
def calibrate(model, data, first, stop, origins):
    model.eval()
    model.gain.zero_()
    n = data.shape[-1]
    lead = torch.arange(96, device=data.device)
    tau = torch.tensor(TAUS, device=data.device)
    decay = torch.exp(-(lead[None, :] + 1) / tau[:, None])
    base = torch.linspace(first, stop - 96, origins, device=data.device).long()
    error_sums = torch.zeros(96, 3, dtype=torch.float64, device=data.device)
    for ids in base.split(12):
        history = input_at(data, ids, model.lookback)
        edge = history[:, -1] - history[:, model.period - 1::model.period].mean(1)
        with amp():
            err = model(history).float() - data[ids[:, None] + lead]
        error_sums[:, 0] += err.square().sum((0, 2))
        error_sums[:, 1] += (err * edge[:, None]).sum((0, 2))
        error_sums[:, 2] += edge.square().sum()
    result = {}
    for h in HORIZONS:
        err, cross, norm = error_sums[:h].T
        weighted_cross = decay[:, :h].double() @ cross
        weighted_norm = decay[:, :h].double().square() @ norm
        gain = (-weighted_cross / weighted_norm.clamp_min(1e-12)).clamp(0, 1.5)
        score = (err.sum() + 2*gain*weighted_cross + gain.square()*weighted_norm) / (len(base)*h*n)
        candidates = list(zip(TAUS, gain.tolist(), score.tolist()))
        if not all(math.isfinite(mse) for _, _, mse in candidates):
            raise FloatingPointError(f'Nonfinite validation MSE for H={h}')
        selected = min(candidates, key=lambda item: item[2])
        result[h] = dict(zip(('tau', 'gain', 'mse'), selected))
    return result


def fit(config, data, a, b, folder, stage, previous=None):
    stop, factor, validation_origins = STAGES[stage]
    torch.manual_seed(SEED)
    rng = torch.Generator(device=data.device).manual_seed(SEED)
    model = Model(config).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], fused=True)
    start = 0
    if previous is not None:
        start = previous['steps']
        model.load_state_dict(torch.load(folder / previous['checkpoint'], map_location='cpu', weights_only=True))
        resume = torch.load(folder / f'optimizer{start}.pt', map_location='cpu', weights_only=True)
        opt.load_state_dict(resume['optimizer'])
        rng.set_state(resume['rng'])
    opt.param_groups[0]['lr'] = config['lr'] * factor
    lags = config['period'] * torch.arange(1, config['lookback']//config['period']+1, device=data.device)
    prefix = F.pad(data[:a].cumsum(0), (0, 0, 1, 0))
    batch = 96 if data.shape[-1] <= 400 else 64
    total = torch.zeros((), device=data.device)
    model.train()
    for _ in range(start, stop):
        ids = torch.randint(config['lookback'], a, (batch,), generator=rng, device=data.device)
        x = data[ids[:, None]-lags].transpose(1, 2)
        center = (prefix[ids]-prefix[ids-config['lookback']])/config['lookback']
        y = data[ids]
        opt.zero_grad(set_to_none=True)
        with amp():
            loss = (model.fit_phase(x, center)-y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        opt.step()
        total += loss.detach()
    train_mse = float(total/(stop-start))
    if not math.isfinite(train_mse):
        raise FloatingPointError('Nonfinite training MSE')
    validation = calibrate(model, data, a+96, b, validation_origins)
    checkpoint = folder / f'step{stop}.pt'
    torch.save(model.state_dict(), checkpoint)
    if stage < len(STAGES) - 1:
        torch.save({'optimizer': opt.state_dict(), 'rng': rng.get_state()}, folder / f'optimizer{stop}.pt')
    return dict(config=config, steps=stop, train_mse=train_mse, validation=validation,
                validation_origins=validation_origins, checkpoint=checkpoint.name)


@torch.inference_mode()
def test_model(model, data, first, horizons, calibration):
    model.eval()
    model.gain.zero_()
    starts = {h: torch.arange(first, len(data)-h+1, device=data.device) for h in horizons}
    lead = torch.arange(max(horizons), device=data.device)
    weights = {h: calibration[h]['gain'] * torch.exp(-(lead[:h] + 1) / calibration[h]['tau']) for h in horizons}
    totals = torch.zeros(len(horizons), 2, device=data.device, dtype=torch.float64)
    for offset in range(0, len(starts[min(horizons)]), 12):
        ids = starts[min(horizons)][offset:offset+12]
        active = [h for h in horizons if offset < len(starts[h])]
        history = input_at(data, ids, model.lookback)
        edge = history[:, -1] - history[:, model.period-1::model.period].mean(1)
        with amp():
            forecast = model(history, max(active)).float()
        for j, h in enumerate(horizons):
            count = min(len(ids), len(starts[h])-offset)
            if count <= 0:
                continue
            error = forecast[:count, :h] + edge[:count, None] * weights[h][None, :, None]
            error -= data[ids[:count, None] + lead[:h]]
            totals[j] += torch.stack((error.square().sum(), error.abs().sum()))
    scores = {}
    for h, (mse, mae) in zip(horizons, totals.tolist()):
        divisor = len(starts[h]) * h * data.shape[-1]
        scores[h] = dict(mse=mse/divisor, mae=mae/divisor, origins=len(starts[h]))
    return scores


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rank(records, horizon=None):
    def score(record):
        validation = record['validation']
        if horizon is not None:
            return validation[horizon]['mse']
        return mean(value['mse'] for value in validation.values())

    return sorted(records, key=score)


def promote(records, keep_each_period=False):
    selected = {record['trial']: record for record in rank(records)[:2]}
    for h in HORIZONS:
        selected.update((record['trial'], record) for record in rank(records, h)[:2])
    if keep_each_period:
        for period in PERIODS:
            pool = [record for record in records if record['config']['period'] == period]
            selected.update((record['trial'], record) for record in rank(pool)[:1])
    return list(selected.values())


def search_dataset(name, data, train_end, val_end, folder):
    folder.mkdir()
    trials = [dict(config=config, trial=f'trial{index:03d}')
              for index, config in enumerate(trial_configs(), 1)]
    for stage, (steps, _, _) in enumerate(STAGES):
        records = []
        for trial in trials:
            trial_folder = folder / trial['trial']
            trial_folder.mkdir(exist_ok=True)
            try:
                record = fit(trial['config'], data, train_end, val_end, trial_folder, stage,
                             previous=trial if stage else None)
            except FloatingPointError as error:
                write_json(trial_folder / f'step{steps}.json',
                           dict(config=trial['config'], steps=steps, error=str(error)))
                print(f'SKIP {name} {trial["trial"]} step={steps}: {error}', flush=True)
                continue
            record['trial'] = trial['trial']
            write_json(trial_folder / f'step{steps}.json', record)
            records.append(record)
            print(f'VAL {name} {trial["trial"]} step={steps} {record["validation"]}', flush=True)
        if stage < len(STAGES) - 1:
            if not records:
                raise RuntimeError(f'{name}: no finite validation candidates at step {steps}')
            trials = promote(records, keep_each_period=(stage == 0))
            print(f'PROMOTE {name} step={steps}: {len(trials)}/{len(records)}', flush=True)

    # Compare the two final budgets on the same validation origins.
    finalists = records
    for trial in trials:
        model = Model(trial['config']).to(data.device)
        checkpoint = folder / trial['trial'] / trial['checkpoint']
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        validation_origins = STAGES[-1][2]
        try:
            validation = calibrate(model, data, train_end+96, val_end, validation_origins)
        except FloatingPointError as error:
            print(f'SKIP {name} {trial["trial"]} final validation: {error}', flush=True)
            continue
        finalists.append(dict(trial, validation=validation, validation_origins=validation_origins))
    if not finalists:
        raise RuntimeError(f'{name}: no finite finalists on the final validation origins')

    write_json(folder / 'final_validation.json', finalists)
    selected = {h: rank(finalists, h)[0] for h in HORIZONS}
    write_json(folder / 'selected.json', selected)
    return selected


def summarize(rows, output):
    order = {name: index for index, name in enumerate(DATASETS)}
    rows.sort(key=lambda row: (order[row['dataset']], row['horizon']))
    with (output/'per_task.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Dataset | Test MSE | Test MAE |', '|---|---:|---:|']
    for name in (*DATASETS, 'Average'):
        subset = rows if name == 'Average' else [row for row in rows if row['dataset'] == name]
        values = [mean(row[key] for row in subset) for key in ('mse', 'mae')]
        lines.append('| ' + ' | '.join((name, *(f'{v:.4f}' for v in values))) + ' |')
    summary = '\n'.join(lines) + '\n'
    (output/'comparison.md').write_text(summary)
    print(summary, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root_path', type=Path, required=True, help='Directory containing the four original PEMS .npz files')
    parser.add_argument('--output', type=Path, required=True, help='New output directory for this run')
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('A CUDA GPU supporting BF16 is required')
    for name in DATASETS:
        if not (args.root_path / f'{name}.npz').is_file():
            parser.error(f'Missing dataset: {args.root_path / (name + ".npz")}')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    print(f'{torch.cuda.get_device_name(0)} | torch={torch.__version__} | seed={SEED}', flush=True)
    fitted = {}
    for name in DATASETS:
        data, a, b = load_data(name, args.root_path)
        fitted[name] = search_dataset(name, data, a, b, args.output / name)
        del data
    rows = []
    for name, selected in fitted.items():
        data, _, b = load_data(name, args.root_path, test=True)
        groups = {}
        for h, record in selected.items():
            checkpoint = args.output / name / record['trial'] / record['checkpoint']
            groups.setdefault(checkpoint, []).append(h)
        for checkpoint, horizons in groups.items():
            record = selected[horizons[0]]
            config = record['config']
            model = Model(config).to(data.device)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            calibration = {h: selected[h]['validation'][h] for h in horizons}
            metrics = test_model(model, data, b+96, horizons, calibration)
            parameters = sum(p.numel() for p in model.parameters())
            for h in horizons:
                score = metrics[h]
                if not all(math.isfinite(score[k]) for k in ('mse', 'mae')):
                    raise RuntimeError(f'Nonfinite test metrics for {name}, H={h}')
                row = dict(dataset=name, horizon=h, **config, steps=record['steps'], parameters=parameters,
                           tau=calibration[h]['tau'], gain=calibration[h]['gain'],
                           validation_mse=calibration[h]['mse'], **score)
                rows.append(row)
                print(f'TEST {name} H={h} {score}', flush=True)
        del data
    summarize(rows, args.output)
