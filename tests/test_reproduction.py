"""A partial or corrupt metric table must never pass release verification."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_results', ROOT / 'reproduction/compare_results.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ReproductionComparisonTests(unittest.TestCase):
    def compare(self, frame):
        with tempfile.TemporaryDirectory() as temporary:
            frame.to_csv(Path(temporary) / 'summary.csv', index=False)
            with patch.object(sys, 'argv', ['compare_results.py', '--results', temporary]):
                with contextlib.redirect_stdout(io.StringIO()):
                    comparison.main()

    def test_complete_reference_passes(self):
        self.compare(pd.read_csv(ROOT / 'reproduction/reference_32_tasks.csv'))

    def test_missing_or_infinite_metric_is_rejected(self):
        for metric in ['validation_mse', 'test_mse', 'test_mae']:
            for invalid in [np.nan, np.inf, -np.inf]:
                with self.subTest(metric=metric, invalid=invalid):
                    frame = pd.read_csv(ROOT / 'reproduction/reference_32_tasks.csv')
                    frame.loc[0, metric] = invalid
                    with self.assertRaisesRegex(SystemExit, 'missing or nonfinite'):
                        self.compare(frame)

    def test_incomplete_task_coverage_is_rejected(self):
        frame = pd.read_csv(ROOT / 'reproduction/reference_32_tasks.csv')
        with self.assertRaises(SystemExit):
            self.compare(frame.iloc[:-1])


if __name__ == '__main__':
    unittest.main()
