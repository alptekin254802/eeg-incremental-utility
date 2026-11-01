"""Guard the scalar license expected by Zenodo's CFF import path."""
from pathlib import Path
import re
import unittest


class ZenodoCitationCompatibility(unittest.TestCase):
    def test_software_license_is_one_spdx_string(self):
        citation = Path(__file__).resolve().parents[1] / "CITATION.cff"
        text = citation.read_text(encoding="utf-8")
        licenses = re.findall(r'^license:[ \t]*([^\r\n]*)', text, re.MULTILINE)
        self.assertEqual(len(licenses), 1)
        self.assertIn(licenses[0].strip(), ('MIT', '"MIT"', "'MIT'"),
                      "Zenodo requires one software license string; lists can pass CFF validation but fail import.")
        self.assertIn("CC BY 4.0", text,
                      "The metadata must retain the separate documentation and derived-content license scope.")


if __name__ == "__main__":
    unittest.main()
