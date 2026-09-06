# coding=utf-8
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../../src")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestKernelsOutputPaths(unittest.TestCase):
    """Tests for where kernels_output writes server-named output files.

    The file names come from the notebook's output listing, so they are not trusted:
    a name that resolves outside the requested directory must be refused, while the
    nested layout of legitimate output (`sub/dir/result.csv`) must be preserved.
    """

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.download_needed = MagicMock(return_value=True)  # type: ignore[method-assign]

    def _download(self, file_names, target_dir):
        """Runs kernels_output against a fake listing of `file_names`; keeps the patched requests.get in self.mock_get."""
        response = MagicMock()
        response.files = [
            MagicMock(file_name=name, url=f"https://example.com/{i}") for i, name in enumerate(file_names)
        ]
        response.next_page_token = ""
        response.log = None
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = response

        with (
            patch.object(KaggleApi, "build_kaggle_client") as mock_client,
            patch("kaggle.api.kaggle_api_extended.requests.get") as mock_get,
        ):
            self.mock_get = mock_get
            mock_get.return_value = MagicMock(content=b"payload")
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            outfiles, _ = self.api.kernels_output("owner/kernel-slug", target_dir, quiet=True)
        return outfiles

    def test_traversal_file_name_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            target_dir = os.path.join(root, "downloads")
            with self.assertRaises(ValueError) as ctx:
                self._download(["../outside.txt"], target_dir)
            self.assertIn("../outside.txt", str(ctx.exception))
            self.assertFalse(os.path.exists(os.path.join(root, "outside.txt")))
            self.mock_get.assert_not_called()

    def test_absolute_file_name_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            target_dir = os.path.join(root, "downloads")
            escaped = os.path.join(root, "escaped.txt")
            with self.assertRaises(ValueError):
                self._download([escaped], target_dir)
            self.assertFalse(os.path.exists(escaped))
            self.mock_get.assert_not_called()

    def test_nested_file_name_keeps_its_directory(self):
        with tempfile.TemporaryDirectory() as root:
            target_dir = os.path.join(root, "downloads")
            outfiles = self._download(["result.csv", "sub/dir/result.csv"], target_dir)
            expected = [os.path.join(target_dir, "result.csv"), os.path.join(target_dir, "sub", "dir", "result.csv")]
            self.assertEqual([os.path.normpath(p) for p in outfiles], expected)
            for path in expected:
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), b"payload")


if __name__ == "__main__":
    unittest.main()
