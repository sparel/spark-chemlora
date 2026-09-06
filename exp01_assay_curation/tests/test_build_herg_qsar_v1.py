import importlib.util
from pathlib import Path
import unittest

import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_herg_qsar_v1.py"
spec = importlib.util.spec_from_file_location("build_herg_qsar_v1", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def row(**overrides):
    base = {
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": 100.0,
        "standard_units": "nM",
        "data_validity_comment": None,
    }
    base.update(overrides)
    return pd.Series(base)


class TestQsarAdmission(unittest.TestCase):
    def test_exact_ic50_nm_is_admitted_and_converted(self):
        eligible, reason, value_nm, pic50 = module.classify_row(row())
        self.assertTrue(eligible)
        self.assertEqual(reason, "")
        self.assertEqual(value_nm, 100.0)
        self.assertEqual(pic50, 7.0)

    def test_censored_ic50_is_not_coerced_to_exact(self):
        eligible, reason, value_nm, pic50 = module.classify_row(row(standard_relation=">"))
        self.assertFalse(eligible)
        self.assertEqual(reason, "nonexact_or_missing_relation")
        self.assertIsNone(value_nm)
        self.assertIsNone(pic50)

    def test_flagged_ic50_is_excluded(self):
        eligible, reason, _, _ = module.classify_row(row(data_validity_comment="Outside typical range"))
        self.assertFalse(eligible)
        self.assertEqual(reason, "data_validity_flag")

    def test_dimensionless_pic50_is_excluded_by_ic50_only_policy(self):
        eligible, reason, value_nm, pic50 = module.classify_row(
            row(standard_type="pIC50", standard_value=5.31, standard_units=None)
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "endpoint_not_ic50")
        self.assertIsNone(value_nm)
        self.assertIsNone(pic50)


if __name__ == "__main__":
    unittest.main()
