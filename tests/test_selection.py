"""Independent examples distinguish horizon-shared normalized selection."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selection import select_shared


def row(n, h, mse, gain=0):
    return dict(period=4, cycles=n, regularization=0., revin=0, tau_cycles=1.,
                horizon=h, validation_mse=mse, parameters=n+1, lookback=4*n,
                boundary_gain=gain)


class SharedSelectionTests(unittest.TestCase):
    def test_normalizes_each_horizon_before_averaging(self):
        # Raw mean prefers n=2 (4 vs 5.5); normalized mean prefers n=1
        # (1.75 vs 2.5), because the second horizon has a different error scale.
        rows = [row(1, 96, 1, .1), row(1, 720, 10, .8),
                row(2, 96, 4, .2), row(2, 720, 4, .7)]
        selected = select_shared(rows, [96, 720], 2)
        self.assertEqual(selected['shared_configuration']['cycles'], 1)
        self.assertEqual(selected['validation_normalizers'], {'96': 1., '720': 4.})
        self.assertEqual(selected['shared_selection_score'], 1.75)
        self.assertEqual(selected['winners']['96']['boundary_gain'], .1)
        self.assertEqual(selected['winners']['720']['boundary_gain'], .8)

    def test_rejects_missing_duplicate_and_test_columns(self):
        rows = [row(1, 96, 1), row(1, 720, 2)]
        with self.assertRaises(ValueError):
            select_shared(rows[:1], [96, 720])
        with self.assertRaises(ValueError):
            select_shared(rows + rows[:1], [96, 720])
        with self.assertRaises(ValueError):
            select_shared([{**rows[0], 'test_mse': 0}, rows[1]], [96, 720])

    def test_tie_break_uses_fewer_parameters(self):
        rows = [row(2, 96, 1), row(2, 720, 2), row(1, 96, 1), row(1, 720, 2)]
        selected = select_shared(rows, [96, 720])
        self.assertEqual(selected['shared_configuration']['cycles'], 1)

    def test_zero_error_and_nonfinite_candidates(self):
        rows = [row(1, 96, 0), row(1, 720, 0), row(2, 96, 0), row(2, 720, float('inf'))]
        selected = select_shared(rows, [96, 720])
        self.assertEqual(selected['shared_selection_score'], 0)
        self.assertEqual(selected['eligible_shared_candidates'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
