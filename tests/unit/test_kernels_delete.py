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
    """Tests for kernels_delete_cli() and kernels_delete()."""

    def setUp(self):
        self.api = _make_api()

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "confirmation")
    def test_kernels_delete_with_version_raises_error(self, mock_confirmation, mock_build):
        """When a version is included in the kernel identifier, raise ValueError instead of an unpacking error."""
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_delete_cli("owner/kernel-slug/3")
        self.assertIn("version", str(ctx.exception).lower())
        self.assertIn("kaggle kernels delete owner/kernel-slug", str(ctx.exception))
        mock_confirmation.assert_not_called()
        mock_build.assert_not_called()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_with_version_no_confirm_raises_error(self, mock_build):
        """Even with no_confirm=True (-y), a versioned kernel identifier must raise ValueError and delete nothing."""
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_delete_cli("owner/kernel-slug/3", no_confirm=True)
        self.assertIn("version", str(ctx.exception).lower())
        mock_build.assert_not_called()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_with_too_many_parts_raises_error(self, mock_build):
        """An identifier with more than three parts gets the standard format error."""
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_delete_cli("owner/kernel-slug/3/4", no_confirm=True)
        self.assertIn("{username}/{kernel-slug}", str(ctx.exception))
        mock_build.assert_not_called()

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_success(self, mock_build, mock_confirmation, mock_print):
        """A plain owner/kernel-slug identifier deletes the kernel."""
        mock_kaggle = MagicMock()
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.kernels_delete_cli("owner/kernel-slug")

        mock_confirmation.assert_called_once_with("delete the kernel: owner/kernel-slug")
        request = mock_kaggle.kernels.kernels_api_client.delete_kernel.call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "kernel-slug")
        mock_print.assert_any_call("Kernel owner/kernel-slug deleted successfully")

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_cancelled(self, mock_build, mock_confirmation, mock_print):
        """When confirmation is declined, nothing is deleted."""
        self.api.kernels_delete_cli("owner/kernel-slug")

        mock_print.assert_any_call("Deletion cancelled")
        mock_build.assert_not_called()

    def test_kernels_delete_none_raises_error(self):
        """When kernel is None, raises ValueError."""
        with self.assertRaises(ValueError):
            self.api.kernels_delete_cli(None)

    def test_kernels_delete_without_owner_raises_error(self):
        """A bare slug is rejected; delete requires an explicit owner."""
        with self.assertRaises(ValueError):
            self.api.kernels_delete_cli("kernel-slug")


if __name__ == "__main__":
    unittest.main()
