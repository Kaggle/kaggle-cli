# coding=utf-8
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestCompetitionDeletePage(unittest.TestCase):
    """Tests for competition_delete_page and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def _patch_client(self, mock_client):
        mock_kaggle = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_delete_no_confirm_sends_request(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        result = self.api.competition_delete_page(
            competition_name="my-comp",
            page_name="rules",
            no_confirm=True,
        )

        self.assertTrue(result)
        request = mock_kaggle.competitions.competition_api_client.delete_competition_page.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(request.page_name, "rules")

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "confirmation", return_value=True)
    def test_delete_with_confirmation_yes_sends_request(self, mock_confirm, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        result = self.api.competition_delete_page(competition_name="my-comp", page_name="rules")

        self.assertTrue(result)
        mock_confirm.assert_called_once()
        self.assertIn("rules", mock_confirm.call_args[0][0])
        self.assertIn("my-comp", mock_confirm.call_args[0][0])
        mock_kaggle.competitions.competition_api_client.delete_competition_page.assert_called_once()

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "confirmation", return_value=False)
    def test_delete_cancelled_skips_request(self, mock_confirm, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            result = self.api.competition_delete_page(competition_name="my-comp", page_name="rules")
        finally:
            sys.stdout = sys.__stdout__

        self.assertFalse(result)
        self.assertIn("Deletion cancelled", captured.getvalue())
        mock_kaggle.competitions.competition_api_client.delete_competition_page.assert_not_called()

    @patch.object(KaggleApi, "competition_delete_page", return_value=True)
    def test_cli_forwards_yes_flag(self, mock_delete):
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.competition_delete_page_cli(
                competition="my-comp",
                page_name="rules",
                no_confirm=True,
            )
        finally:
            sys.stdout = sys.__stdout__

        mock_delete.assert_called_once_with("my-comp", "rules", no_confirm=True)
        self.assertIn('Page "rules" deleted', captured.getvalue())

    @patch.object(KaggleApi, "competition_delete_page", return_value=False)
    def test_cli_cancelled_prints_no_success(self, mock_delete):
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.competition_delete_page_cli(competition="my-comp", page_name="rules")
        finally:
            sys.stdout = sys.__stdout__

        self.assertNotIn("deleted", captured.getvalue())

    @patch.object(KaggleApi, "competition_delete_page", return_value=True)
    def test_cli_uses_competition_opt_flag(self, mock_delete):
        self.api.competition_delete_page_cli(
            competition_opt="my-comp",
            page_name="rules",
            no_confirm=True,
        )
        self.assertEqual(mock_delete.call_args[0][0], "my-comp")

    def test_cli_missing_competition_raises(self):
        self.api.config_values = {}
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_delete_page_cli(page_name="rules")
        self.assertIn("No competition specified", str(ctx.exception))

    def test_cli_missing_page_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_delete_page_cli(competition="my-comp")
        self.assertIn("--page-name is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
