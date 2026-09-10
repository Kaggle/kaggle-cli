# coding=utf-8
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from kaggle.api.kaggle_api_extended import KaggleApi


class TestKernelsPushRetiredAccelerator(unittest.TestCase):
    """Tests the warning shown when a push requests an accelerator the server substitutes."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.valid_push_language_types = ["python", "r", "julia", "rmarkdown"]
        self.api.valid_push_kernel_types = ["script", "notebook"]
        self.api.valid_push_pinning_types = ["original", "latest"]
        self.api.KERNEL_METADATA_FILE = "kernel-metadata.json"

    @staticmethod
    def _mock_client(mock_client):
        mock_kaggle = MagicMock()
        response = MagicMock()
        response.error = None
        response.invalidTags = []
        response.invalidDatasetSources = []
        response.invalidCompetitionSources = []
        response.invalidKernelSources = []
        response.versionNumber = 3
        response.url = "https://www.kaggle.com/code/testuser/test-kernel"
        mock_kaggle.kernels.kernels_api_client.save_kernel.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    def _kernel_folder(self, machine_shape=None):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, True)
        metadata = {
            "id": "testuser/test-kernel",
            "title": "Test Kernel Title",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
        }
        if machine_shape is not None:
            metadata["machine_shape"] = machine_shape
        with open(os.path.join(tmpdir, self.api.KERNEL_METADATA_FILE), "w", encoding="utf-8") as meta:
            json.dump(metadata, meta)
        with open(os.path.join(tmpdir, "script.py"), "w", encoding="utf-8") as code:
            code.write("print('hello')\n")
        return tmpdir

    @staticmethod
    def _sent_request(mock_kaggle):
        mock_kaggle.kernels.kernels_api_client.save_kernel.assert_called_once()
        return mock_kaggle.kernels.kernels_api_client.save_kernel.call_args[0][0]

    def _push(self, mock_client, acc=None, machine_shape=None):
        """Pushes a kernel and returns (sent request, stderr text)."""
        mock_kaggle = self._mock_client(mock_client)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.api.kernels_push(self._kernel_folder(machine_shape), acc=acc)
        return self._sent_request(mock_kaggle), stderr.getvalue()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_accelerator_flag_warns(self, mock_client):
        _, stderr = self._push(mock_client, acc="NvidiaTeslaP100")

        self.assertIn("is retired", stderr)
        self.assertIn("NvidiaTeslaT4", stderr)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_machine_shape_warns(self, mock_client):
        # The metadata file is the other way a machine shape reaches the request.
        _, stderr = self._push(mock_client, machine_shape="NvidiaTeslaP100")

        self.assertIn("is retired", stderr)
        self.assertIn("NvidiaTeslaT4", stderr)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_match_is_case_insensitive(self, mock_client):
        _, stderr = self._push(mock_client, acc="nvidiateslap100")

        self.assertIn("is retired", stderr)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_surrounding_whitespace_is_matched_and_not_displayed(self, mock_client):
        # Stray whitespace in kernel-metadata.json still matches, and quoting the raw value would show it.
        _, stderr = self._push(mock_client, machine_shape="  NvidiaTeslaP100  ")

        self.assertIn("'NvidiaTeslaP100' is retired", stderr)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_warning_carries_the_shared_icon_prefix(self, mock_client):
        # Other warnings in the CLI lead with this, so the visual style should match.
        _, stderr = self._push(mock_client, acc="NvidiaTeslaP100")

        self.assertIn("⚠ Warning:", stderr)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_each_retired_shape_names_its_replacement(self, mock_client):
        for acc, replacement in (
            ("NvidiaTeslaP100", "NvidiaTeslaT4"),
            ("TpuV38", "TpuV5E8"),
            ("Tpu1VmV38", "TpuV5E8"),
            ("TpuV232", "CPU"),
            ("TpuV2256", "CPU"),
        ):
            with self.subTest(acc=acc):
                _, stderr = self._push(mock_client, acc=acc)
                self.assertIn("is retired", stderr)
                self.assertIn(replacement, stderr)
                mock_client.reset_mock()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_accelerators_that_are_not_retired_are_quiet(self, mock_client):
        # Flag-gated shapes such as A100 can be swapped for the default too, but per-user flags, admin
        # status and competition allowlists decide that and the client cannot see any of it. Category names
        # such as "Gpu" ask for a kind of accelerator rather than a specific chip, so whatever the server
        # picks is what was requested; `--accelerator gpu` is a documented spelling.
        for acc in ("NvidiaTeslaT4", "TpuV5E8", "NvidiaTeslaA100", "Gpu", "gpu"):
            with self.subTest(acc=acc):
                _, stderr = self._push(mock_client, acc=acc)
                self.assertEqual(stderr, "")
                mock_client.reset_mock()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_warning_does_not_change_the_request(self, mock_client):
        # The server decides the real accelerator, so the warning must stay advisory.
        request, _ = self._push(mock_client, acc="NvidiaTeslaP100")

        self.assertEqual(request.machine_shape, "NvidiaTeslaP100")


if __name__ == "__main__":
    unittest.main()
