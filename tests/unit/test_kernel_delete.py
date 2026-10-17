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


class TestKernelsDelete(unittest.TestCase):
    """Tests for kernels_delete() version rejection (#1202)."""

    def setUp(self):
        self.api = _make_api()

    def test_kernels_delete_with_version_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_delete("owner/kernel-slug/3", no_confirm=True)
        self.assertIn("version", str(ctx.exception).lower())
        self.assertIn("owner/kernel-slug", str(ctx.exception))

    def test_kernels_delete_with_v_prefixed_version_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_delete("owner/kernel-slug/v3", no_confirm=True)
        self.assertIn("v3", str(ctx.exception))

    def test_kernels_delete_with_version_no_confirm_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_delete("owner/kernel-slug/3")
        self.assertIn("version", str(ctx.exception).lower())

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_success(self, mock_build, mock_confirmation, mock_print):
        mock_kaggle = MagicMock()
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.kernels_delete("owner/kernel-slug", no_confirm=False)

        mock_kaggle.kernels.kernels_api_client.delete_kernel.assert_called_once()
        req = mock_kaggle.kernels.kernels_api_client.delete_kernel.call_args[0][0]
        self.assertEqual(req.user_name, "owner")
        self.assertEqual(req.kernel_slug, "kernel-slug")

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_trailing_slash_accepted(self, mock_build, mock_confirmation, mock_print):
        mock_kaggle = MagicMock()
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.kernels_delete("owner/kernel-slug/", no_confirm=False)

        mock_kaggle.kernels.kernels_api_client.delete_kernel.assert_called_once()

    def test_kernels_delete_none_raises_error(self):
        with self.assertRaises(ValueError):
            self.api.kernels_delete(None)


if __name__ == "__main__":
    unittest.main()
