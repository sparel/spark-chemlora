import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).parents[2] / "common" / "scripts" / "evaluate_json_outputs.py"
spec = importlib.util.spec_from_file_location("evaluate_json_outputs", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class TestReferenceExtraction(unittest.TestCase):
    def test_extracts_assistant_json_from_chat_messages(self):
        record = {"id": "x", "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "record"},
            {"role": "assistant", "content": '{"endpoint":"IC50","units":"nM"}'},
        ]}
        self.assertEqual(module.reference_value(record), '{"endpoint":"IC50","units":"nM"}')


if __name__ == "__main__":
    unittest.main()
