# coding=utf-8
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.host_service import CompetitionSettings


def _make_settings(**overrides) -> CompetitionSettings:
    settings = CompetitionSettings()
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


class TestCompetitionGetSettings(unittest.TestCase):
    """Tests for competition_get_settings and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}

    def _patch_client(self, mock_client, settings=None):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.get_competition_settings.return_value = (
            settings if settings is not None else _make_settings()
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_get_settings_builds_request_with_competition_name(self, mock_client):
        mock_kaggle = self._patch_client(mock_client, _make_settings(title="Amazing Comp"))

        result = self.api.competition_get_settings("my-comp")

        request = mock_kaggle.competitions.competition_api_client.get_competition_settings.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(result.title, "Amazing Comp")

    @patch.object(KaggleApi, "competition_get_settings")
    def test_cli_positional_competition(self, mock_get):
        mock_get.return_value = _make_settings(title="T")
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_settings_cli(competition="my-comp")
        mock_get.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_get_settings")
    def test_cli_dash_c_option(self, mock_get):
        mock_get.return_value = _make_settings()
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_settings_cli(competition_opt="my-comp")
        mock_get.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_get_settings")
    def test_cli_falls_back_to_config(self, mock_get):
        mock_get.return_value = _make_settings()
        self.api.config_values = {self.api.CONFIG_NAME_COMPETITION: "from-config"}
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_settings_cli()
        mock_get.assert_called_once_with("from-config")

    def test_cli_missing_competition_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_get_settings_cli()
        self.assertIn("No competition specified", str(ctx.exception))

    @patch.object(KaggleApi, "competition_get_settings")
    def test_cli_json_output_emits_valid_json(self, mock_get):
        mock_get.return_value = _make_settings(
            title="Amazing Comp",
            brief_description="do the thing",
            rules_required=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_settings_cli(competition="my-comp", json_output=True)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["title"], "Amazing Comp")
        self.assertEqual(payload["briefDescription"], "do the thing")
        self.assertTrue(payload["rulesRequired"])

    @patch.object(KaggleApi, "competition_get_settings")
    def test_cli_human_readable_groups_and_skips_defaults(self, mock_get):
        mock_get.return_value = _make_settings(
            title="Amazing Comp",
            rules_required=True,
            max_gpu_runtime_minutes=240,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_settings_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("[General]", out)
        self.assertIn("title: Amazing Comp", out)
        self.assertIn("rules_required: true", out)
        self.assertIn("[Code Competition]", out)
        self.assertIn("max_gpu_runtime_minutes: 240", out)
        # Defaults (unset strings, false bools, zero ints) should be skipped.
        self.assertNotIn("brief_description", out)
        self.assertNotIn("has_scripts", out)
        self.assertNotIn("[Access & Teams]", out)

    @patch.object(KaggleApi, "competition_get_settings")
    def test_cli_human_readable_all_defaults_shows_placeholder(self, mock_get):
        mock_get.return_value = _make_settings()
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_settings_cli(competition="my-comp")
        self.assertIn("(no settings set)", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
