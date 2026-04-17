# coding=utf-8
import unittest
from unittest.mock import patch, MagicMock, call
import io
import sys

sys.path.insert(0, "..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus


class TestKernelsLogs(unittest.TestCase):
    """Tests for the kernels_logs and kernels_logs_cli methods."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_returns_log_string(self, mock_validate, mock_client):
        """Test that kernels_logs returns the log string from the API response."""
        mock_response = MagicMock()
        mock_response.log = "Line 1\nLine 2\nLine 3"
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.kernels_logs("owner/kernel-slug")
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_returns_empty_string_when_no_log(self, mock_validate, mock_client):
        """Test that kernels_logs returns empty string when log is None."""
        mock_response = MagicMock()
        mock_response.log = None
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.kernels_logs("owner/kernel-slug")
        self.assertEqual(result, "")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_logs_raises_when_kernel_none(self, mock_client):
        """Test that kernels_logs raises ValueError when kernel is None."""
        with self.assertRaises(ValueError):
            self.api.kernels_logs(None)

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "get_config_value", return_value="defaultuser")
    def test_kernels_logs_uses_default_user_for_bare_slug(self, mock_config, mock_client):
        """Test that a bare kernel slug uses the default username."""
        mock_response = MagicMock()
        mock_response.log = "some log"
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.kernels_logs("my-kernel")
        self.assertEqual(result, "some log")

        # Verify the request used the default user
        call_args = mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "defaultuser")
        self.assertEqual(request.kernel_slug, "my-kernel")

    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_oneshot(self, mock_logs):
        """Test one-shot mode prints log to stdout."""
        mock_logs.return_value = "Line 1\nLine 2\nDone"

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug")
        finally:
            sys.stdout = sys.__stdout__

        self.assertEqual(captured.getvalue(), "Line 1\nLine 2\nDone\n")

    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_uses_kernel_opt(self, mock_logs):
        """Test that kernel_opt is used when kernel is None."""
        mock_logs.return_value = "log output"

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli(None, kernel_opt="owner/kernel-slug")
        finally:
            sys.stdout = sys.__stdout__

        mock_logs.assert_called_once_with("owner/kernel-slug")

    @patch("time.sleep")
    @patch.object(KaggleApi, "kernels_status")
    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_follow_mode(self, mock_logs, mock_status, mock_sleep):
        """Test follow mode polls and prints new lines, stops on terminal status."""
        # First poll: kernel is running, returns some log lines
        # Second poll: kernel is complete, returns more log lines
        mock_logs.side_effect = [
            "Line 1\nLine 2",
            "Line 1\nLine 2\nLine 3\nLine 4",
            "Line 1\nLine 2\nLine 3\nLine 4",  # final fetch after terminal status
        ]

        status_running = MagicMock()
        status_running.status = KernelWorkerStatus.RUNNING
        status_complete = MagicMock()
        status_complete.status = KernelWorkerStatus.COMPLETE
        mock_status.side_effect = [status_running, status_complete]

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug", follow=True, interval=1)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # First poll prints "Line 1\nLine 2"
        # Second poll prints "Line 3\nLine 4"
        self.assertIn("Line 1", output)
        self.assertIn("Line 2", output)
        self.assertIn("Line 3", output)
        self.assertIn("Line 4", output)

        # Verify sleep was called with the right interval
        mock_sleep.assert_called_with(1)

    @patch("time.sleep")
    @patch.object(KaggleApi, "kernels_status")
    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_follow_stops_on_error(self, mock_logs, mock_status, mock_sleep):
        """Test follow mode stops when kernel status is ERROR."""
        mock_logs.side_effect = [
            "Line 1",
            "Line 1",  # final fetch
        ]

        status_error = MagicMock()
        status_error.status = KernelWorkerStatus.ERROR
        mock_status.return_value = status_error

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug", follow=True, interval=1)
        finally:
            sys.stdout = sys.__stdout__

        # Should only poll once before stopping
        self.assertEqual(mock_status.call_count, 1)

    @patch("time.sleep")
    @patch.object(KaggleApi, "kernels_status")
    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_follow_stops_on_cancel(self, mock_logs, mock_status, mock_sleep):
        """Test follow mode stops when kernel status is CANCEL_ACKNOWLEDGED."""
        mock_logs.side_effect = [
            "Cancelled",
            "Cancelled",  # final fetch
        ]

        status_cancel = MagicMock()
        status_cancel.status = KernelWorkerStatus.CANCEL_ACKNOWLEDGED
        mock_status.return_value = status_cancel

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug", follow=True, interval=1)
        finally:
            sys.stdout = sys.__stdout__

        self.assertEqual(mock_status.call_count, 1)

    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_empty_log(self, mock_logs):
        """Test one-shot mode with empty log."""
        mock_logs.return_value = ""

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug")
        finally:
            sys.stdout = sys.__stdout__

        self.assertEqual(captured.getvalue(), "\n")


if __name__ == "__main__":
    unittest.main()
