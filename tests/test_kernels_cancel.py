# coding=utf-8
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiCancelKernelSessionRequest,
    ApiCancelKernelSessionResponse,
)


class TestKernelsCancel(unittest.TestCase):
    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.args = []
        self.api.config_values = {"username": "testuser", "key": "testkey"}
        self.api.already_printed_version_warning = True

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_cancel_successful(self, mock_build_client):
        """Test that a successful cancel prints a success message."""
        mock_kaggle = MagicMock()
        mock_build_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build_client.return_value.__exit__ = MagicMock(return_value=False)

        # Mock the raw HTTP call to get session status with session ID
        mock_http_client = MagicMock()
        mock_kaggle.http_client.return_value = mock_http_client

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"status": "running", "kernelSessionId": 12345}
        mock_http_response.raise_for_status = MagicMock()
        mock_http_client._session.send.return_value = mock_http_response
        mock_http_client._session.merge_environment_settings.return_value = {}
        mock_http_client._prepare_request.return_value = MagicMock(url="http://test")

        # Mock the cancel response
        cancel_response = ApiCancelKernelSessionResponse()
        mock_kaggle.kernels.kernels_api_client.cancel_kernel_session.return_value = cancel_response

        result = self.api.kernels_cancel("owner/kernel-slug")

        # Verify cancel was called with the correct session ID
        call_args = mock_kaggle.kernels.kernels_api_client.cancel_kernel_session.call_args
        cancel_request = call_args[0][0]
        self.assertEqual(cancel_request.kernel_session_id, 12345)
        self.assertEqual(result.error_message, "")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_cancel_with_error_response(self, mock_build_client):
        """Test that cancel handles an error response from the API."""
        mock_kaggle = MagicMock()
        mock_build_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build_client.return_value.__exit__ = MagicMock(return_value=False)

        mock_http_client = MagicMock()
        mock_kaggle.http_client.return_value = mock_http_client

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"status": "complete", "kernelSessionId": 99999}
        mock_http_response.raise_for_status = MagicMock()
        mock_http_client._session.send.return_value = mock_http_response
        mock_http_client._session.merge_environment_settings.return_value = {}
        mock_http_client._prepare_request.return_value = MagicMock(url="http://test")

        cancel_response = ApiCancelKernelSessionResponse()
        cancel_response.error_message = "Session is not running"
        mock_kaggle.kernels.kernels_api_client.cancel_kernel_session.return_value = cancel_response

        result = self.api.kernels_cancel("owner/kernel-slug")
        self.assertEqual(result.error_message, "Session is not running")

    def test_cancel_none_kernel_raises(self):
        """Test that passing None raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_cancel(None)
        self.assertIn("A kernel must be specified", str(ctx.exception))

    def test_cancel_invalid_format_raises(self):
        """Test that a kernel slug that is too short raises ValueError."""
        with self.assertRaises(ValueError):
            self.api.kernels_cancel("owner/ab")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_cancel_no_session_id_raises(self, mock_build_client):
        """Test that missing session ID in status response raises ValueError."""
        mock_kaggle = MagicMock()
        mock_build_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build_client.return_value.__exit__ = MagicMock(return_value=False)

        mock_http_client = MagicMock()
        mock_kaggle.http_client.return_value = mock_http_client

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"status": "complete"}
        mock_http_response.raise_for_status = MagicMock()
        mock_http_client._session.send.return_value = mock_http_response
        mock_http_client._session.merge_environment_settings.return_value = {}
        mock_http_client._prepare_request.return_value = MagicMock(url="http://test")

        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_cancel("owner/kernel-slug")
        self.assertIn("No active session found", str(ctx.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_cancel_without_owner_uses_config_user(self, mock_build_client):
        """Test that a kernel slug without owner uses the configured username."""
        mock_kaggle = MagicMock()
        mock_build_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build_client.return_value.__exit__ = MagicMock(return_value=False)

        mock_http_client = MagicMock()
        mock_kaggle.http_client.return_value = mock_http_client

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"status": "running", "kernelSessionId": 100}
        mock_http_response.raise_for_status = MagicMock()
        mock_http_client._session.send.return_value = mock_http_response
        mock_http_client._session.merge_environment_settings.return_value = {}
        mock_http_client._prepare_request.return_value = MagicMock(url="http://test")

        cancel_response = ApiCancelKernelSessionResponse()
        mock_kaggle.kernels.kernels_api_client.cancel_kernel_session.return_value = cancel_response

        self.api.kernels_cancel("my-kernel")

        # Verify the status request used the configured username
        prepare_call = mock_http_client._prepare_request.call_args
        status_request = prepare_call[0][2]
        self.assertEqual(status_request.user_name, "testuser")
        self.assertEqual(status_request.kernel_slug, "my-kernel")

    @patch("builtins.print")
    @patch.object(KaggleApi, "kernels_cancel")
    def test_cancel_cli_success(self, mock_cancel, mock_print):
        """Test CLI wrapper prints success message."""
        response = ApiCancelKernelSessionResponse()
        mock_cancel.return_value = response

        self.api.kernels_cancel_cli("owner/kernel-slug")

        mock_cancel.assert_called_once_with("owner/kernel-slug")
        mock_print.assert_called_once_with("Kernel session for 'owner/kernel-slug' was cancelled successfully.")

    @patch("builtins.print")
    @patch.object(KaggleApi, "kernels_cancel")
    def test_cancel_cli_error(self, mock_cancel, mock_print):
        """Test CLI wrapper prints error message."""
        response = ApiCancelKernelSessionResponse()
        response.error_message = "Cannot cancel completed session"
        mock_cancel.return_value = response

        self.api.kernels_cancel_cli("owner/kernel-slug")

        mock_print.assert_called_once_with("Cancel failed: Cannot cancel completed session")

    @patch("builtins.print")
    @patch.object(KaggleApi, "kernels_cancel")
    def test_cancel_cli_uses_kernel_opt(self, mock_cancel, mock_print):
        """Test CLI wrapper falls back to kernel_opt argument."""
        response = ApiCancelKernelSessionResponse()
        mock_cancel.return_value = response

        self.api.kernels_cancel_cli(None, kernel_opt="owner/my-kernel")

        mock_cancel.assert_called_once_with("owner/my-kernel")


if __name__ == "__main__":
    unittest.main()
