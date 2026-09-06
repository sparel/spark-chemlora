import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_herg_dataset.py"
spec = importlib.util.spec_from_file_location("build_herg_dataset", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class TestHergCurationContract(unittest.TestCase):
    def test_prompt_declares_the_flattened_output_schema(self):
        text = module.prompt({})
        self.assertIn(module.OUTPUT_SCHEMA, text)
        for key in ("target_name", "endpoint", "qsar_usable", "exclude_reason", "curation_notes"):
            self.assertIn('"' + key + '"', module.OUTPUT_SCHEMA)

    def test_pic50_is_excluded_from_strict_numerical_qsar(self):
        usable, reasons = module.qsar_admission("pIC50", "=", 5.3, None, None, "CC")
        self.assertFalse(usable)
        self.assertEqual(reasons, ["endpoint_not_ic50"])

    def test_missing_structure_excludes_an_otherwise_valid_ic50(self):
        usable, reasons = module.qsar_admission("IC50", "=", 50.0, "nM", None, None)
        self.assertFalse(usable)
        self.assertEqual(reasons, ["missing_canonical_smiles"])


if __name__ == "__main__":
    unittest.main()
