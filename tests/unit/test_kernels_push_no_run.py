# coding=utf-8
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

from kaggle.api.kaggle_api_extended import KaggleApi


class _PushTestCase(unittest.TestCase):
    """Shared setup for pushing a minimal, valid kernel."""

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

    def _kernel_folder(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, True)
        metadata = {
            "id": "testuser/test-kernel",
            "title": "Test Kernel Title",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
        }
        with open(os.path.join(tmpdir, self.api.KERNEL_METADATA_FILE), "w", encoding="utf-8") as meta:
            json.dump(metadata, meta)
        with open(os.path.join(tmpdir, "script.py"), "w", encoding="utf-8") as code:
            code.write("print('hello')\n")
        return tmpdir

    @staticmethod
    def _sent_request(mock_kaggle):
        mock_kaggle.kernels.kernels_api_client.save_kernel.assert_called_once()
        return mock_kaggle.kernels.kernels_api_client.save_kernel.call_args[0][0]


class TestKernelsPushNoRun(_PushTestCase):
    """Tests that --no-run maps to the QUICK_SAVE execution type."""

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_no_run_requests_a_quick_save(self, mock_client):
        mock_kaggle = self._mock_client(mock_client)

        self.api.kernels_push(self._kernel_folder(), no_run=True)

        self.assertEqual(self._sent_request(mock_kaggle).kernel_execution_type, KernelExecutionType.QUICK_SAVE)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_default_push_leaves_the_execution_type_unspecified(self, mock_client):
        # Not setting the field is what makes the server save and run, so the default must not change.
        mock_kaggle = self._mock_client(mock_client)

        self.api.kernels_push(self._kernel_folder())

        request = self._sent_request(mock_kaggle)
        self.assertNotEqual(request.kernel_execution_type, KernelExecutionType.QUICK_SAVE)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_accelerator_and_timeout_still_apply_with_no_run(self, mock_client):
        mock_kaggle = self._mock_client(mock_client)

        self.api.kernels_push(self._kernel_folder(), timeout=600, acc="NvidiaTeslaT4", no_run=True)

        request = self._sent_request(mock_kaggle)
        self.assertEqual(request.kernel_execution_type, KernelExecutionType.QUICK_SAVE)
        self.assertEqual(request.machine_shape, "NvidiaTeslaT4")
        self.assertEqual(request.session_timeout_seconds, 600)


class TestKernelsPushCliNoRun(_PushTestCase):
    """Tests the wrapper forwarding and the message it prints."""

    @patch.object(KaggleApi, "kernels_push")
    def test_cli_forwards_no_run(self, mock_push):
        mock_push.return_value = MagicMock(error=None, versionNumber=3, url="u")

        with redirect_stdout(io.StringIO()):
            self.api.kernels_push_cli("folder", None, None, True)

        self.assertIs(mock_push.call_args[0][3], True)

    @patch.object(KaggleApi, "kernels_push")
    def test_cli_defaults_to_running(self, mock_push):
        mock_push.return_value = MagicMock(error=None, versionNumber=3, url="u")

        with redirect_stdout(io.StringIO()):
            self.api.kernels_push_cli("folder", None, None)

        self.assertIs(mock_push.call_args[0][3], False)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_no_run_does_not_point_at_run_progress(self, mock_client):
        self._mock_client(mock_client)
        folder = self._kernel_folder()

        output = io.StringIO()
        with redirect_stdout(output):
            self.api.kernels_push_cli(folder, None, None, True)

        printed = output.getvalue()
        self.assertIn("successfully saved without running", printed)
        self.assertNotIn("check progress", printed)
        self.assertIn("https://www.kaggle.com/code/testuser/test-kernel", printed)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_default_message_is_unchanged(self, mock_client):
        self._mock_client(mock_client)
        folder = self._kernel_folder()

        output = io.StringIO()
        with redirect_stdout(output):
            self.api.kernels_push_cli(folder, None, None)

        self.assertIn(
            "Kernel version 3 successfully pushed.  Please check progress at "
            "https://www.kaggle.com/code/testuser/test-kernel",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
