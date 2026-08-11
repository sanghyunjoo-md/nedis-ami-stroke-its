"""Data-free checks for the public extension configuration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "analysis_extension" / "config" / "extension_config_v1.0.json"


class ExtensionConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_expected_complete_week_counts(self) -> None:
        study = self.config["study"]
        self.assertEqual(study["expected_pre_weeks"], 59)
        self.assertEqual(study["expected_post_weeks"], 44)

    def test_multiplicity_domain_sizes(self) -> None:
        multiplicity = self.config["multiplicity"]
        self.assertEqual(multiplicity["Access"], 6)
        self.assertEqual(multiplicity["Presenting severity"], 18)
        self.assertEqual(multiplicity["Care pathways"], 24)
        self.assertEqual(multiplicity["Disease heterogeneity"], 16)

    def test_eight_extension_outcomes(self) -> None:
        self.assertEqual(len(self.config["outcomes"]), 8)


if __name__ == "__main__":
    unittest.main()
