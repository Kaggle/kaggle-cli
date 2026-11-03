import importlib
import sys
import unittest
from unittest.mock import patch

from kaggle.models.upload_file import UploadFile


class TestUploadFile(unittest.TestCase):
    def test_to_dict_returns_every_attribute(self):
        upload_file = UploadFile(token="token-123", description="A file", columns=[])

        self.assertEqual(upload_file.to_dict(), {"token": "token-123", "description": "A file", "columns": []})

    def test_imports_without_six(self):
        # six is not a declared dependency, so it may be missing from an install.
        with patch.dict(sys.modules, {"six": None}):
            sys.modules.pop("kaggle.models.upload_file", None)
            module = importlib.import_module("kaggle.models.upload_file")

        self.assertEqual(module.UploadFile(token="token-123").to_dict()["token"], "token-123")


if __name__ == "__main__":
    unittest.main()
