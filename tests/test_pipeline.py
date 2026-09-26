"""Regression checks for split boundaries, exact scoring, and test isolation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from all_order_stats import build_all_stats, stats_for_q
from config import final_configuration, parse_args, search_spec
import data as data_module
from data import describe_dataset, development_data, fit_scaler
from evaluate import evaluate
from fit_boundary import fit_gain_train, mark_train_stats, score_fixed_gain
from io_utils import read_json, save_json
from kernel_scoring import score_kernel
from models.TinyCycle import Model
from pipeline import freeze_selected, test_selected
from search import kernel_for, search_dataset


class NumericsTests(unittest.TestCase):
    def test_default_search_count(self):
        grid = search_spec(parse_args([]))
        self.assertEqual(grid['candidates_per_horizon'], 15040)
        self.assertEqual(sum(len(v) for v in grid['cycles'].values()), 94)
        self.assertEqual(grid['selection_metric'], 'equal_mean_of_per_horizon_normalized_validation_mse')

    def test_exact_mse_matches_direct_model(self):
        data = torch.tensor(np.random.default_rng(3).normal(size=(180, 3)), dtype=torch.float32)
        for period, horizon in [(4, 13), (7, 14), (17, 9)]:
            origins = np.arange(80, 101)
            shared = build_all_stats(data, origins, horizon, period, 4, [1, 7])
            for n in [1, 2, 4]:
                for center in [0, 1]:
                    cfg = SimpleNamespace(seq_len=n*period, pred_len=horizon, period=period,
                                          revin=center, tau_cycles=7)
                    model = Model(cfg)
                    weights = np.arange(1, n+1, dtype=float)
                    weights = weights / weights.sum() * .85
                    with torch.no_grad():
                        model.theta.copy_(torch.tensor(weights))
                        model.boundary_gain.fill_(.31)
                    stats = stats_for_q(shared, n)
                    kernel = kernel_for(weights, horizon, period, center)
                    base = score_kernel(stats, kernel)['mse']
                    fast = score_fixed_gain(stats, kernel, base, 7, .31)
                    slow = evaluate(model, data, 80, 100 + horizon)['mse']
                    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10)

    def test_gain_fit_rejects_validation_statistics_and_crossing_targets(self):
        data = np.random.default_rng(4).normal(size=(200, 2)).astype(np.float32)
        origins = np.arange(60, 81)
        stats = stats_for_q(build_all_stats(data, origins, 13, 4, 3, [1]), 3)
        kernel = kernel_for([.5, .2, .1], 13, 4, 0)
        with self.assertRaises(ValueError):
            fit_gain_train(stats, kernel, 1)
        stats.split = 'validation'
        with self.assertRaises(ValueError):
            mark_train_stats(stats, train_end=200, data_rows=200, origins=origins)
        crossing = stats_for_q(build_all_stats(data, origins, 13, 4, 3, [1]), 3)
        with self.assertRaises(ValueError):
            mark_train_stats(crossing, train_end=90, data_rows=90, origins=origins)

    def test_model_only_reads_selected_window_and_enforces_720_limit(self):
        cfg = SimpleNamespace(seq_len=12, pred_len=13, period=4, revin=1, tau_cycles=7)
        model = Model(cfg)
        with torch.no_grad():
            model.theta.fill_(.2)
            model.boundary_gain.fill_(.4)
        a = torch.randn(2, 40, 3)
        b = a.clone()
        b[:, :-12] += 10000
        torch.testing.assert_close(model(a), model(b), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4)
        cfg.seq_len = 720
        self.assertEqual(Model(cfg).order, 180)
        cfg.seq_len = 724
        with self.assertRaises(ValueError):
            Model(cfg)

    def test_paper_extension_preserves_validation_selected_hyperparameters(self):
        for period, cycles, lookback in [(24, 30, 720), (96, 7, 672), (4, 180, 720), (8, 90, 720)]:
            selected = dict(period=period, cycles=1, lookback=period,
                            regularization=.1, revin=1, tau_cycles=7)
            final = final_configuration(selected, 96, dict(fit_samples=4096, batch_size=64))
            self.assertEqual(final['seq_len'], lookback)
            self.assertEqual(final['seq_len'] // period, cycles)
            for key in ['period', 'regularization', 'revin', 'tau_cycles']:
                self.assertEqual(final[key], selected[key])
            self.assertEqual(selected['cycles'], 1)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(42)
        t = np.arange(1200)
        self.values = np.column_stack([np.sin(t / 9) + rng.normal(0, .1, len(t)),
                                       np.cos(t / 17) + t / 2000,
                                       np.sin(t / 31) + rng.normal(0, .2, len(t))])
        self.write_data('a.csv', self.values)

    def tearDown(self):
        self.tmp.cleanup()

    def write_data(self, name, values):
        frame = pd.DataFrame(values, columns=['a', 'b', 'OT'])
        frame.insert(0, 'date', pd.date_range('2020-01-01', periods=len(frame), freq='h'))
        frame.to_csv(self.root / name, index=False)

    def args(self, file='a.csv', output='run'):
        return parse_args(['--datasets', 'Synthetic', '--data-root', str(self.root), '--data-path', file,
                           '--train-end', '800', '--val-end', '1000', '--test-end', '1200',
                           '--output-dir', str(self.root/output), '--horizons', '13', '25',
                           '--periods', '4', '8', '--cycles', '1', '3', '--regularizations', '0', '.1',
                           '--taus', '1', '7', '--fit-samples', '64', '--batch-size', '16', '--threads', '1'])

    def fit(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            search_dataset('Synthetic', args)
            freeze_selected('Synthetic', args)

    def test_test_perturbation_cannot_change_selection_scaler_or_weights(self):
        changed = self.values.copy()
        changed[1000:] = changed[1000:] * 100 + 500
        self.write_data('b.csv', changed)
        a, b = self.args(), self.args('b.csv', 'changed')
        self.fit(a)
        self.fit(b)
        fa, fb = a.output_dir/'Synthetic', b.output_dir/'Synthetic'
        self.assertEqual(read_json(fa/'selected.json'), read_json(fb/'selected.json'))
        self.assertEqual(read_json(fa/'scaler.json'), read_json(fb/'scaler.json'))
        for h in [13, 25]:
            sa = torch.load(fa/f'h{h}.pt', weights_only=True)
            sb = torch.load(fb/f'h{h}.pt', weights_only=True)
            for key in sa:
                torch.testing.assert_close(sa[key], sb[key], atol=0, rtol=0)
        with contextlib.redirect_stdout(io.StringIO()):
            ra, rb = test_selected('Synthetic', a), test_selected('Synthetic', b)
        self.assertGreater(abs(ra[0]['test_mse'] - rb[0]['test_mse']), 1)

    def test_search_and_training_do_not_load_test_values(self):
        poisoned = self.values.copy()
        poisoned[1000:] = np.nan
        self.write_data('poisoned.csv', poisoned)
        args = self.args('poisoned.csv')
        original = data_module.read_values
        loaded = []
        def guarded(spec, end):
            self.assertLessEqual(end, spec['val_end'])
            loaded.append(end)
            return original(spec, end)
        with patch('data.read_values', side_effect=guarded):
            self.fit(args)
        self.assertEqual(loaded, [1000, 1000])
        self.assertTrue((args.output_dir/'Synthetic/frozen.json').is_file())
        with self.assertRaises(ValueError):
            test_selected('Synthetic', args)

    def test_validation_does_not_fit_standardization(self):
        changed = self.values.copy()
        changed[800:1000] += 999
        self.write_data('changed_val.csv', changed)
        scalers = []
        for name in ['a.csv', 'changed_val.csv']:
            spec = describe_dataset('Synthetic', self.root/name, (800, 1000, 1200))
            _, scaler, _ = development_data(spec)
            scalers.append(scaler)
        self.assertEqual(scalers[0], scalers[1])
        self.assertEqual(scalers[0]['n_samples'], 800)

    def test_duplicate_timestamp_preserves_benchmark_row_order(self):
        path = self.root/'a.csv'
        frame = pd.read_csv(path)
        frame.loc[101, 'date'] = frame.loc[100, 'date']
        frame.to_csv(path, index=False)
        spec = describe_dataset('Synthetic', path, (800, 1000, 1200))
        raw, columns = data_module.read_values(spec, 1000)
        self.assertEqual(len(raw), 1000)
        np.testing.assert_array_equal(raw, frame[columns].to_numpy(dtype=np.float32)[:1000])

    def test_frozen_artifact_tamper_is_rejected(self):
        args = self.args()
        with self.assertRaises(RuntimeError):
            test_selected('Synthetic', args)
        self.fit(args)
        p = args.output_dir/'Synthetic/selected.json'
        selected = read_json(p)
        selected['winners']['13']['regularization'] = 123
        save_json(p, selected)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            test_selected('Synthetic', args)

    def test_resume_and_training_boundaries(self):
        args = self.args()
        self.fit(args)
        args.resume = True
        self.fit(args)
        folder = args.output_dir/'Synthetic'
        selected = read_json(folder/'selected.json')
        self.assertEqual(selected['shared_candidates'], 32)
        for row in selected['winners'].values():
            self.assertEqual({k: row[k] for k in selected['shared_fields']}, selected['shared_configuration'])
        states = [torch.load(folder/f'h{h}.pt', weights_only=True) for h in [13, 25]]
        torch.testing.assert_close(states[0]['theta'], states[1]['theta'], rtol=0, atol=0)
        frozen = read_json(folder/'frozen.json')
        self.assertEqual(frozen['selection_configuration'], selected['shared_configuration'])
        self.assertEqual(frozen['shared_configuration']['cycles'], 720 // selected['shared_configuration']['period'])
        for h in [13, 25]:
            self.assertEqual(read_json(folder/f'h{h}_config.json')['seq_len'], 720)
        for row in frozen['checks']:
            self.assertEqual(row['first_train_target'], 720)
            self.assertEqual(row['selection_validation_mse'], selected['winners'][str(row['horizon'])]['validation_mse'])
        for period in [4, 8]:
            record = read_json(folder/f'p{period}_complete.json')
            self.assertEqual(record['rows'], 32)
            for bounds in record['bounds']:
                self.assertLessEqual(bounds['gain_fit_last_target_exclusive'], 800)
                self.assertEqual(bounds['validation_first_origin'], 800)
                self.assertEqual(bounds['validation_target_end_exclusive'], 1000)
        with contextlib.redirect_stdout(io.StringIO()):
            first = test_selected('Synthetic', args)
            second = test_selected('Synthetic', args)
        self.assertEqual(first, second)
        for row in first:
            self.assertEqual(row['lookback'], 720)
            self.assertEqual(row['cycles'], 720 // row['period'])
            self.assertEqual(row['parameters'], row['cycles'] + 1)
        args.taus = [1]
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            search_dataset('Synthetic', args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
