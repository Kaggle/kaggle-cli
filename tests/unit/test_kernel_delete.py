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

    def test_kernels_delete_with_whitespace_version_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.kernels_delete("owner/kernel-slug/   ", no_confirm=True)
        self.assertIn("version", str(ctx.exception).lower())

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_cancelled(self, mock_build, mock_confirmation, mock_print):
        self.api.kernels_delete("owner/kernel-slug", no_confirm=False)

        mock_confirmation.assert_called_once_with("delete the kernel: owner/kernel-slug")
        mock_print.assert_any_call("Deletion cancelled")
        mock_build.assert_not_called()

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_confirmation_prompt_uses_owner_and_slug(self, mock_build, mock_confirmation, mock_print):
        mock_kaggle = MagicMock()
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.kernels_delete("owner/kernel-slug/", no_confirm=False)

        mock_confirmation.assert_called_once_with("delete the kernel: owner/kernel-slug")
        mock_print.assert_any_call("Kernel owner/kernel-slug deleted successfully")

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_delete_slug_only_resolves_configured_owner(self, mock_build, mock_confirmation, mock_print):
        mock_kaggle = MagicMock()
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.kernels_delete("kernel-slug", no_confirm=False)

        mock_confirmation.assert_called_once_with("delete the kernel: owner/kernel-slug")
        req = mock_kaggle.kernels.kernels_api_client.delete_kernel.call_args[0][0]
        self.assertEqual(req.user_name, "owner")
        self.assertEqual(req.kernel_slug, "kernel-slug")
        mock_print.assert_any_call("Kernel owner/kernel-slug deleted successfully")

    @patch.object(KaggleApi, "confirmation")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_short_bare_slug_rejected_before_confirmation(self, mock_build, mock_confirmation):
        for no_confirm in (False, True):
            with self.subTest(no_confirm=no_confirm):
                with self.assertRaisesRegex(ValueError, "at least five characters"):
                    self.api.kernels_delete("tiny", no_confirm=no_confirm)
        mock_confirmation.assert_not_called()
        mock_build.assert_not_called()

    @patch.object(KaggleApi, "confirmation")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_bare_slug_without_username_rejected(self, mock_build, mock_confirmation):
        for username in (None, "", "   "):
            self.api.config_values = {"username": username}
            for no_confirm in (False, True):
                with self.subTest(username=username, no_confirm=no_confirm):
                    with self.assertRaisesRegex(ValueError, "A kernel owner is required"):
                        self.api.kernels_delete("kernel-slug", no_confirm=no_confirm)
        mock_confirmation.assert_not_called()
        mock_build.assert_not_called()

    @patch.object(KaggleApi, "confirmation")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_whitespace_owner_rejected_before_confirmation(self, mock_build, mock_confirmation):
        for no_confirm in (False, True):
            with self.subTest(no_confirm=no_confirm):
                with self.assertRaisesRegex(ValueError, "A kernel owner is required"):
                    self.api.kernels_delete("  /kernel-slug", no_confirm=no_confirm)
        mock_confirmation.assert_not_called()
        mock_build.assert_not_called()

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_explicit_owner_without_configured_username(self, mock_build, mock_confirmation, mock_print):
        self.api.config_values = {}
        self.api.kernels_delete("owner/kernel-slug", no_confirm=True)
        mock_confirmation.assert_not_called()
        client = mock_build.return_value.__enter__.return_value
        client.kernels.kernels_api_client.delete_kernel.assert_called_once()
        request = client.kernels.kernels_api_client.delete_kernel.call_args.args[0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "kernel-slug")
        mock_print.assert_any_call("Kernel owner/kernel-slug deleted successfully")


if __name__ == "__main__":
    unittest.main()
