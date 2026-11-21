# coding=utf-8
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    api.config_values = {"username": "owner"}
    return api


def _mock_client(mock_build, error_message=None):
    mock_kaggle = MagicMock()
    mock_kaggle.kernels.kernels_api_client.cancel_kernel_session.return_value.error_message = error_message
    mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
    mock_build.return_value.__exit__ = MagicMock(return_value=False)
    return mock_kaggle


class TestKernelsCancel(unittest.TestCase):

    def setUp(self):
        self.api = _make_api()

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_cancel_success(self, mock_build, mock_confirmation, mock_print):
        mock_kaggle = _mock_client(mock_build)

        self.api.kernels_cancel(123456789)

        mock_confirmation.assert_called_once_with("cancel the kernel session: 123456789")
        req = mock_kaggle.kernels.kernels_api_client.cancel_kernel_session.call_args[0][0]
        self.assertEqual(req.kernel_session_id, 123456789)
        mock_print.assert_any_call("Kernel session 123456789 cancelled")

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_cancel_no_confirm_skips_prompt(self, mock_build, mock_confirmation, mock_print):
        mock_kaggle = _mock_client(mock_build)

        self.api.kernels_cancel(42, no_confirm=True)

        mock_confirmation.assert_not_called()
        mock_kaggle.kernels.kernels_api_client.cancel_kernel_session.assert_called_once()

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_cancel_declined(self, mock_build, mock_confirmation, mock_print):
        self.api.kernels_cancel(42)

        mock_print.assert_any_call("Cancel aborted")
        mock_build.assert_not_called()

    @patch("builtins.print")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_cancel_error_message_raises(self, mock_build, mock_print):
        _mock_client(mock_build, error_message="Session already finished")

        with self.assertRaisesRegex(ValueError, "Session already finished"):
            self.api.kernels_cancel(42, no_confirm=True)
        mock_print.assert_not_called()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_cancel_rejects_missing_or_non_positive_id(self, mock_build):
        for session_id in (None, 0, -5):
            with self.subTest(session_id=session_id):
                with self.assertRaises(ValueError):
                    self.api.kernels_cancel(session_id, no_confirm=True)
        mock_build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
