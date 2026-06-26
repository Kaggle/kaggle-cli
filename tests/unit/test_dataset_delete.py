# coding=utf-8
import unittest
from unittest.mock import MagicMock, patch
import sys
from io import StringIO

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    api.config_values = {"username": "owner"}
    return api


class TestDatasetDelete(unittest.TestCase):
    """Tests for dataset_delete_cli() and dataset_delete()."""

    def setUp(self):
        self.api = _make_api()

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=False)
    def test_dataset_delete_cli_cancelled(self, mock_confirmation, mock_print):
        """When confirmation is cancelled (returns False), print 'Deletion cancelled' and no success message."""
        self.api.dataset_delete_cli("owner/dataset-slug")

        # Verify confirmation was called
        mock_confirmation.assert_called_once_with("delete the dataset: owner/dataset-slug")

        # Verify that print("Deletion cancelled") was called
        mock_print.assert_any_call("Deletion cancelled")

        # Verify that the success message was NOT printed
        for call_args in mock_print.call_args_list:
            printed_str = call_args[0][0]
            if "deleted successfully" in printed_str:
                self.fail("Success message was printed on cancellation!")

    @patch("builtins.print")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_delete_cli_success(self, mock_build, mock_confirmation, mock_print):
        """When confirmation is approved (returns True), call backend and print success message."""
        mock_kaggle = MagicMock()
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_delete_cli("owner/dataset-slug")

        mock_confirmation.assert_called_once_with("delete the dataset: owner/dataset-slug")

        # Verify backend client delete_dataset was called
        mock_kaggle.datasets.dataset_api_client.delete_dataset.assert_called_once()

        # Verify success message was printed
        mock_print.assert_any_call('Dataset "owner/dataset-slug" deleted successfully.')


if __name__ == "__main__":
    unittest.main()
