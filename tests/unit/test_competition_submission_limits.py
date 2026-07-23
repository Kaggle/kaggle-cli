# coding=utf-8
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiSubmissionLimits


def _make_limits(**overrides) -> ApiSubmissionLimits:
    limits = ApiSubmissionLimits()
    for name, value in overrides.items():
        setattr(limits, name, value)
    return limits


class TestCompetitionSubmissionLimits(unittest.TestCase):
    """Tests for competition_get_submission_limits and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}

    def _patch_client(self, mock_client, limits=None):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.get_submission_limits.return_value = (
            limits if limits is not None else _make_limits()
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_builds_request_with_competition_name(self, mock_client):
        mock_kaggle = self._patch_client(mock_client, _make_limits(num_allowed_now=3))

        result = self.api.competition_get_submission_limits("my-comp")

        request = mock_kaggle.competitions.competition_api_client.get_submission_limits.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(result.num_allowed_now, 3)

    @patch.object(KaggleApi, "competition_get_submission_limits")
    def test_cli_positional_competition(self, mock_get):
        mock_get.return_value = _make_limits()
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_submission_limits_cli(competition="my-comp")
        mock_get.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_get_submission_limits")
    def test_cli_dash_c_option(self, mock_get):
        mock_get.return_value = _make_limits()
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_submission_limits_cli(competition_opt="my-comp")
        mock_get.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_get_submission_limits")
    def test_cli_falls_back_to_config(self, mock_get):
        mock_get.return_value = _make_limits()
        self.api.config_values = {self.api.CONFIG_NAME_COMPETITION: "from-config"}
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_submission_limits_cli()
        mock_get.assert_called_once_with("from-config")

    def test_cli_missing_competition_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_get_submission_limits_cli()
        self.assertIn("No competition specified", str(ctx.exception))

    @patch.object(KaggleApi, "competition_get_submission_limits")
    def test_cli_json_output_emits_valid_json(self, mock_get):
        mock_get.return_value = _make_limits(num_today=3, num_total=47, num_allowed_now=2, limited_by_total=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_submission_limits_cli(competition="my-comp", json_output=True)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["numToday"], 3)
        self.assertEqual(payload["numTotal"], 47)
        self.assertEqual(payload["numAllowedNow"], 2)
        self.assertTrue(payload["limitedByTotal"])

    @patch.object(KaggleApi, "competition_get_submission_limits")
    def test_cli_human_view_prints_all_labels(self, mock_get):
        mock_get.return_value = _make_limits(num_today=3, num_total=47, num_allowed_now=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_submission_limits_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("Submissions today: 3", out)
        self.assertIn("Lifetime submissions: 47", out)
        self.assertIn("Remaining today: 2", out)
        self.assertNotIn("lifetime cap", out)

    @patch.object(KaggleApi, "competition_get_submission_limits")
    def test_cli_human_view_appends_lifetime_cap_suffix(self, mock_get):
        mock_get.return_value = _make_limits(num_today=1, num_total=999, num_allowed_now=1, limited_by_total=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_submission_limits_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("Remaining today: 1 (limited by lifetime cap)", out)


if __name__ == "__main__":
    unittest.main()
