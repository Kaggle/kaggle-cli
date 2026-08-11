# coding=utf-8
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestCompetitionAddHost(unittest.TestCase):
    """Tests for competition_add_host and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}

    def _patch_client(self, mock_client):
        mock_kaggle = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    def _add_host_call(self, mock_kaggle):
        return mock_kaggle.competitions.competition_api_client.add_competition_host

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_add_host_builds_request(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        result = self.api.competition_add_host("my-comp", "alice", no_confirm=True)

        request = self._add_host_call(mock_kaggle).call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(request.user_name, "alice")
        self.assertTrue(result)

    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_add_host_prompts_when_not_confirmed(self, mock_client, mock_confirm):
        mock_kaggle = self._patch_client(mock_client)

        result = self.api.competition_add_host("my-comp", "alice")

        mock_confirm.assert_called_once()
        self._add_host_call(mock_kaggle).assert_called_once()
        self.assertTrue(result)

    @patch.object(KaggleApi, "confirmation", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_add_host_prompt_names_user_and_competition(self, mock_client, mock_confirm):
        """The prompt must name the user being granted access and the competition."""
        self._patch_client(mock_client)

        self.api.competition_add_host("my-comp", "alice")

        action = mock_confirm.call_args[0][0]
        self.assertEqual(action, "add 'alice' as a host of competition 'my-comp'")

    @patch.object(KaggleApi, "confirmation", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_add_host_declined_makes_no_request(self, mock_client, mock_confirm):
        """Declining the prompt must not reach the API."""
        mock_kaggle = self._patch_client(mock_client)

        with redirect_stdout(io.StringIO()) as out:
            result = self.api.competition_add_host("my-comp", "alice")

        self.assertFalse(result)
        self._add_host_call(mock_kaggle).assert_not_called()
        self.assertIn("Add host cancelled", out.getvalue())

    @patch.object(KaggleApi, "confirmation")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_add_host_no_confirm_skips_prompt(self, mock_client, mock_confirm):
        self._patch_client(mock_client)

        self.api.competition_add_host("my-comp", "alice", no_confirm=True)

        mock_confirm.assert_not_called()

    @patch.object(KaggleApi, "competition_add_host", return_value=True)
    def test_cli_uses_positional_competition(self, mock_add):
        with redirect_stdout(io.StringIO()) as out:
            self.api.competition_add_host_cli(competition="my-comp", user_name="alice", no_confirm=True)

        mock_add.assert_called_once_with("my-comp", "alice", no_confirm=True)
        self.assertIn("alice", out.getvalue())
        self.assertIn("my-comp", out.getvalue())

    @patch.object(KaggleApi, "competition_add_host", return_value=True)
    def test_cli_uses_competition_opt(self, mock_add):
        with redirect_stdout(io.StringIO()):
            self.api.competition_add_host_cli(competition_opt="my-comp", user_name="alice", no_confirm=True)

        mock_add.assert_called_once_with("my-comp", "alice", no_confirm=True)

    @patch.object(KaggleApi, "competition_add_host", return_value=True)
    def test_cli_falls_back_to_configured_competition(self, mock_add):
        self.api.config_values = {self.api.CONFIG_NAME_COMPETITION: "configured-comp"}

        with redirect_stdout(io.StringIO()) as out:
            self.api.competition_add_host_cli(user_name="alice", no_confirm=True)

        mock_add.assert_called_once_with("configured-comp", "alice", no_confirm=True)
        self.assertIn("Using competition: configured-comp", out.getvalue())

    @patch.object(KaggleApi, "competition_add_host", return_value=True)
    def test_cli_quiet_suppresses_using_competition(self, mock_add):
        self.api.config_values = {self.api.CONFIG_NAME_COMPETITION: "configured-comp"}

        with redirect_stdout(io.StringIO()) as out:
            self.api.competition_add_host_cli(user_name="alice", no_confirm=True, quiet=True)

        self.assertNotIn("Using competition", out.getvalue())

    def test_cli_without_competition_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_add_host_cli(user_name="alice")
        self.assertIn("No competition specified", str(ctx.exception))

    def test_cli_without_user_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_add_host_cli(competition="my-comp")
        self.assertIn("--user is required", str(ctx.exception))

    @patch.object(KaggleApi, "competition_add_host", return_value=False)
    def test_cli_cancelled_prints_no_success_message(self, mock_add):
        """A declined add must not report success."""
        with redirect_stdout(io.StringIO()) as out:
            self.api.competition_add_host_cli(competition="my-comp", user_name="alice")

        self.assertNotIn("added as a host", out.getvalue())


if __name__ == "__main__":
    unittest.main()
