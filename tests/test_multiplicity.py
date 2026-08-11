from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from model_utils import bh_adjust, holm_adjust  # noqa: E402


class MultiplicityTests(unittest.TestCase):
    def test_holm_adjustment(self) -> None:
        observed = holm_adjust(np.array([0.01, 0.04, 0.03, 0.20]))
        expected = np.array([0.04, 0.09, 0.09, 0.20])
        np.testing.assert_allclose(observed, expected)

    def test_bh_adjustment(self) -> None:
        observed = bh_adjust(np.array([0.01, 0.04, 0.03, 0.20]))
        expected = np.array([0.04, 0.05333333333333334, 0.05333333333333334, 0.20])
        np.testing.assert_allclose(observed, expected)

    def test_adjusted_values_preserve_input_order(self) -> None:
        p = np.array([0.50, 0.001, 0.05])
        self.assertLess(holm_adjust(p)[1], holm_adjust(p)[2])
        self.assertLess(bh_adjust(p)[1], bh_adjust(p)[2])


if __name__ == "__main__":
    unittest.main()

