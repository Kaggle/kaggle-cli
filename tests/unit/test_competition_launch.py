# coding=utf-8
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestCompetitionLaunch(unittest.TestCase):
    """Tests for competition_launch and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def _patch_client(self, mock_client):
        mock_kaggle = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_launch_now_sends_request_without_future_time(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_launch(competition_name="my-comp")

        call = mock_kaggle.competitions.competition_api_client.launch_competition.call_args
        request = call[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertIsNone(request.future_time)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_launch_with_future_time(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)
        when = datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        self.api.competition_launch(competition_name="my-comp", future_time=when)

        request = mock_kaggle.competitions.competition_api_client.launch_competition.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(request.future_time, when)

    @patch.object(KaggleApi, "competition_launch")
    def test_cli_launch_now_passes_no_future_time(self, mock_launch):
        self.api.competition_launch_cli(competition="my-comp")

        mock_launch.assert_called_once_with(competition_name="my-comp", future_time=None)

    @patch.object(KaggleApi, "competition_launch")
    def test_cli_parses_zulu_iso_string(self, mock_launch):
        self.api.competition_launch_cli(competition="my-comp", at="2027-01-01T00:00:00Z")

        kwargs = mock_launch.call_args.kwargs
        self.assertEqual(kwargs["competition_name"], "my-comp")
        self.assertEqual(
            kwargs["future_time"],
            datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )

    @patch.object(KaggleApi, "competition_launch")
    def test_cli_parses_explicit_offset(self, mock_launch):
        self.api.competition_launch_cli(competition="my-comp", at="2027-01-01T00:00:00+00:00")

        self.assertEqual(
            mock_launch.call_args.kwargs["future_time"],
            datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )

    @patch.object(KaggleApi, "competition_launch")
    def test_cli_naive_string_assumed_utc(self, mock_launch):
        self.api.competition_launch_cli(competition="my-comp", at="2027-01-01T00:00:00")

        self.assertEqual(
            mock_launch.call_args.kwargs["future_time"],
            datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )

    def test_cli_bad_at_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_launch_cli(competition="my-comp", at="not-a-date")
        self.assertIn("Invalid --at value", str(ctx.exception))

    def test_cli_missing_competition_raises(self):
        self.api.config_values = {}
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_launch_cli()
        self.assertIn("No competition specified", str(ctx.exception))

    @patch.object(KaggleApi, "competition_launch")
    def test_cli_uses_competition_opt_flag(self, mock_launch):
        self.api.competition_launch_cli(competition_opt="my-comp")

        self.assertEqual(mock_launch.call_args.kwargs["competition_name"], "my-comp")


if __name__ == "__main__":
    unittest.main()
