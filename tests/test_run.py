"""The CLI checks the whole requested group before starting any test pass."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import save_json
from run import main


class TestStagePreflightTests(unittest.TestCase):
    def test_missing_later_freeze_prevents_all_test_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_json(root / 'ETTh1/frozen.json', {'artifacts': {}})
            with patch('pipeline.test_selected') as evaluate:
                with self.assertRaisesRegex(RuntimeError, 'every requested dataset'):
                    main(['--stage', 'test', '--datasets', 'ETTh1', 'ETTh2', '--output-dir', temporary])
                evaluate.assert_not_called()

    def test_corrupt_later_freeze_prevents_all_test_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_json(root / 'ETTh1/frozen.json', {'artifacts': {}})
            save_json(root / 'ETTh2/frozen.json', {'artifacts': {'missing.pt': 'invalid'}})
            with patch('pipeline.test_selected') as evaluate:
                with self.assertRaisesRegex(RuntimeError, 'missing or changed'):
                    main(['--stage', 'test', '--datasets', 'ETTh1', 'ETTh2', '--output-dir', temporary])
                evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
