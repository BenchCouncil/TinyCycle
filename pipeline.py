"""Train selected configurations, freeze artifacts, then evaluate test exactly once."""
import time
from types import SimpleNamespace

import numpy as np
import torch

from data import development_data, test_data
from config import FINAL_LOOKBACK_RULE, final_configuration
from evaluate import evaluate
from io_utils import read_json, save_json, sha256, verify_files, write_csv
from models.TinyCycle import Model
from search import source_hashes
from train import fit_model
from selection import RULE, SHARED_FIELDS


def verify_search(folder):
    path = folder / 'search_finished.json'
    if not path.exists():
        raise RuntimeError(f'Run the search stage first: {folder}')
    verify_files(folder, read_json(path)['artifacts'])
    if source_hashes() != read_json(folder / 'source_hashes.json'):
        raise RuntimeError('Program changed since search; start a new output directory')


def freeze_selected(dataset, args):
    folder = args.output_dir / dataset
    verify_search(folder)
    if (folder / 'frozen.json').exists():
        verify_files(folder, read_json(folder / 'frozen.json')['artifacts'])
        print(f'Train {dataset}: existing frozen checkpoints verified', flush=True)
        return
    identity = read_json(folder / 'dataset.json')
    grid = read_json(folder / 'search_spec.json')
    if grid.get('final_lookback_rule') != FINAL_LOOKBACK_RULE:
        raise RuntimeError('Expected the paper lookback-extension protocol')
    selected = read_json(folder / 'selected.json')
    if selected['selection_rule'] != RULE:
        raise RuntimeError('Expected horizon-shared validation selection')
    for row in selected['winners'].values():
        if {k: row[k] for k in SHARED_FIELDS} != selected['shared_configuration']:
            raise RuntimeError('Selected hyperparameters must be identical across horizons')
    development, scaler, actual_identity = development_data(identity)
    if actual_identity != identity or scaler != read_json(folder / 'scaler.json'):
        raise RuntimeError('Train/validation data or training scaler changed after search')
    train = development[:identity['train_end']].clone()
    metrics, fitted_files = [], []
    shared_theta = None
    for horizon in grid['horizons']:
        row = selected['winners'][str(horizon)]
        config = {'seq_len': row['lookback'], 'pred_len': horizon, 'period': row['period'],
                  'regularization': row['regularization'], 'revin': row['revin'],
                  'tau_cycles': row['tau_cycles'], 'fit_samples': grid['fit_samples'],
                  'batch_size': grid['batch_size']}
        tick = time.perf_counter()
        model = fit_model(Model(SimpleNamespace(**config)), train, SimpleNamespace(**config))
        selection_fit_seconds = time.perf_counter() - tick
        li = grid['regularizations'].index(row['regularization'])
        with np.load(folder / f'p{row["period"]}_train_weights.npz') as cache:
            expected = cache[f'{row["cycles"]}_{li}']
        theta_delta = float(np.max(np.abs(model.theta.detach().numpy() - expected)))
        gain_delta = abs(float(model.boundary_gain.detach()) - row['boundary_gain'])
        if theta_delta > 1e-7 or gain_delta > 1e-7:
            raise ArithmeticError('Independent final training disagrees with train-only search fit')
        validation = evaluate(model, development, identity['train_end'], identity['val_end'], args.batch_size)
        delta = abs(validation['mse'] - row['validation_mse'])
        if delta > 1e-7:
            raise ArithmeticError('Direct validation MSE disagrees with exact search statistics')
        # The selected short-window candidate is verified above. Table 1 then
        # changes only Q, deterministically, and refits both coefficients on
        # train. The new validation score is diagnostic and never ranks models.
        extended = final_configuration(row, horizon, grid)
        if extended['seq_len'] == config['seq_len']:
            elapsed = selection_fit_seconds
        else:
            tick = time.perf_counter()
            model = fit_model(Model(SimpleNamespace(**extended)), train, SimpleNamespace(**extended))
            elapsed = time.perf_counter() - tick
            validation = evaluate(model, development, identity['train_end'], identity['val_end'], args.batch_size)
        config = extended
        if shared_theta is None:
            shared_theta = model.theta.detach().clone()
        elif not torch.equal(shared_theta, model.theta.detach()):
            raise RuntimeError('Final cycle weights differ across shared horizons')
        ckpt, settings = folder / f'h{horizon}.pt', folder / f'h{horizon}_config.json'
        temporary = ckpt.with_suffix('.pt.tmp')
        torch.save(model.state_dict(), temporary)
        temporary.replace(ckpt)
        save_json(settings, config)
        fitted_files.extend([ckpt.name, settings.name])
        metrics.append({'dataset': dataset, 'horizon': horizon, 'parameters': model.order + 1,
                        'lookback': model.seq_len, 'cycles': model.order,
                        'selection_cycles': row['cycles'], 'selection_lookback': row['lookback'],
                        'selection_validation_mse': row['validation_mse'],
                        'validation_mse': validation['mse'], 'validation_mae': validation['mae'],
                        'validation_origins': validation['origins'], 'validation_elements': validation['elements'],
                        'train_seconds': elapsed, 'selection_theta_delta': theta_delta,
                        'selection_gain_delta': gain_delta, 'selection_validation_mse_delta': delta,
                        'first_train_target': max(model.seq_len, model.period * min(30, 720 // model.period))})
    write_csv(folder / 'validation_metrics.csv', metrics)
    files = ['selected.json', 'search_spec.json', 'source_hashes.json', 'dataset.json', 'scaler.json',
             'search_finished.json', 'validation_metrics.csv'] + fitted_files
    final_shared = dict(selected['shared_configuration'])
    final_shared['cycles'] = 720 // final_shared['period']
    save_json(folder / 'frozen.json', {'dataset': dataset, 'fit_split': 'train',
              'selection_split': 'validation', 'frozen_before_test': True,
              'selection_rule': RULE, 'selection_configuration': selected['shared_configuration'],
              'shared_configuration': final_shared, 'final_lookback_rule': FINAL_LOOKBACK_RULE,
              'shared_selection_score': selected['shared_selection_score'],
              'validation_normalizers': selected['validation_normalizers'],
              'horizons': grid['horizons'], 'artifacts': {name: sha256(folder / name) for name in files},
              'checks': metrics})
    print(f'Train {dataset}: all {len(metrics)} final models trained and frozen before test', flush=True)


def test_selected(dataset, args):
    folder = args.output_dir / dataset
    frozen_path = folder / 'frozen.json'
    if not frozen_path.exists():
        raise RuntimeError(f'Run search then train before test: {folder}')
    freeze_hash = sha256(frozen_path)
    frozen = read_json(frozen_path)
    verify_files(folder, frozen['artifacts'])
    if source_hashes() != read_json(folder / 'source_hashes.json'):
        raise RuntimeError('Program changed since search; refusing evaluation with stale artifacts')
    identity = read_json(folder / 'dataset.json')
    selected = read_json(folder / 'selected.json')
    scaler = read_json(folder / 'scaler.json')
    # This is the first stage that loads numeric test values.
    data, test_hash = test_data(identity, scaler)
    result_path = folder / 'test_metrics.json'
    if result_path.exists():
        saved = read_json(result_path)
        if saved['frozen_sha256'] != freeze_hash or saved['test_data_sha256'] != test_hash:
            raise RuntimeError('Data/model differs from an existing test report; use a new output directory')
        print(f'Test {dataset}: returning existing report; no new fit or evaluation', flush=True)
        return saved['rows']
    results = []
    for horizon in frozen['horizons']:
        config = read_json(folder / f'h{horizon}_config.json')
        model = Model(SimpleNamespace(**config))
        model.load_state_dict(torch.load(folder / f'h{horizon}.pt', map_location='cpu', weights_only=True))
        tick = time.perf_counter()
        measured = evaluate(model, data, identity['val_end'], identity['test_end'], args.batch_size)
        row = selected['winners'][str(horizon)]
        diagnostic = next(r for r in frozen['checks'] if r['horizon'] == horizon)
        result = {key: row[key] for key in ['horizon', 'period', 'regularization', 'revin', 'tau_cycles']}
        result.update(cycles=model.order, parameters=model.order + 1, lookback=model.seq_len,
                      validation_mse=diagnostic['validation_mse'],
                      selection_cycles=row['cycles'], selection_lookback=row['lookback'],
                      selection_validation_mse=row['validation_mse'])
        result = {'dataset': dataset, **result, 'test_mse': measured['mse'], 'test_mae': measured['mae'],
                  'test_origins': measured['origins'], 'test_elements': measured['elements'],
                  'shared_selection_score': selected['shared_selection_score'],
                  'validation_normalizer': selected['validation_normalizers'][str(horizon)],
                  'test_seconds': time.perf_counter() - tick}
        results.append(result)
        print(f'Test {dataset} L={horizon}: MSE={measured["mse"]:.9f}, MAE={measured["mae"]:.9f}', flush=True)
    verify_files(folder, frozen['artifacts'])
    if sha256(frozen_path) != freeze_hash:
        raise RuntimeError('Freeze manifest changed during test')
    write_csv(folder / 'test_metrics.csv', results)
    save_json(result_path, {'frozen_sha256': freeze_hash, 'test_data_sha256': test_hash,
                           'test_candidate_search': False, 'model_updated_on_test': False,
                           'selection_split': 'validation', 'fit_split': 'train', 'rows': results})
    return results
