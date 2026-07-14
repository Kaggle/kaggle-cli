# coding=utf-8
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.host_service import CompetitionSettings


class TestCompetitionUpdateSettings(unittest.TestCase):
    """Tests for competition_update_settings and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}
        self._tmp_files = []

    def tearDown(self):
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _write_tmp(self, suffix: str, contents: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
        f.write(contents)
        f.close()
        self._tmp_files.append(f.name)
        return f.name

    def _patch_client(self, mock_client, returned=None):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.update_competition_settings.return_value = (
            returned if returned is not None else CompetitionSettings()
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_builds_field_mask_from_supplied_keys_only(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_settings(
            "my-comp",
            {"title": "New Title", "rules_required": True},
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_settings.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(sorted(request.update_mask.paths), ["rules_required", "title"])
        self.assertEqual(request.settings.title, "New Title")
        self.assertTrue(request.settings.rules_required)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_accepts_camel_case_keys(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_settings(
            "my-comp",
            {"briefDescription": "sub", "maxGpuRuntimeMinutes": 240},
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_settings.call_args[0][0]
        # FieldMask paths always use snake_case (SDK canonical form).
        self.assertEqual(sorted(request.update_mask.paths), ["brief_description", "max_gpu_runtime_minutes"])
        self.assertEqual(request.settings.brief_description, "sub")
        self.assertEqual(request.settings.max_gpu_runtime_minutes, 240)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_coerces_iso_datetime_string(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_settings(
            "my-comp",
            {"team_file_deadline": "2026-08-01T10:30:00Z"},
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_settings.call_args[0][0]
        self.assertEqual(
            request.settings.team_file_deadline,
            datetime(2026, 8, 1, 10, 30, 0, tzinfo=timezone.utc),
        )

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_coerces_enum_from_full_name(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_settings(
            "my-comp",
            {"host_segment": "HOST_SEGMENT_FEATURED"},
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_settings.call_args[0][0]
        self.assertEqual(request.settings.host_segment.name, "HOST_SEGMENT_FEATURED")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_coerces_enum_from_short_name(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_settings(
            "my-comp",
            {"publicly_cloneable": "WITH_PRIVATE_SOLUTION_FILE"},
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_settings.call_args[0][0]
        self.assertEqual(request.settings.publicly_cloneable.name, "PUBLICLY_CLONEABLE_WITH_PRIVATE_SOLUTION_FILE")

    def test_update_empty_updates_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings("my-comp", {})
        self.assertIn("No settings to update", str(ctx.exception))

    def test_update_unknown_field_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings("my-comp", {"bogus_field": 1})
        self.assertIn("Unknown competition setting", str(ctx.exception))

    def test_update_wrong_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings("my-comp", {"title": 42})
        self.assertIn("expects a string", str(ctx.exception))

    def test_update_bad_enum_lists_allowed(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings("my-comp", {"host_segment": "NOPE"})
        message = str(ctx.exception)
        self.assertIn("not a valid HostSegment", message)
        self.assertIn("HOST_SEGMENT_FEATURED", message)

    def test_update_bad_datetime_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings("my-comp", {"team_file_deadline": "yesterday"})
        self.assertIn("ISO-8601 datetime", str(ctx.exception))

    # --- CLI wrapper --------------------------------------------------------

    @patch.object(KaggleApi, "competition_update_settings")
    def test_cli_loads_json_file(self, mock_update):
        mock_update.return_value = CompetitionSettings()
        path = self._write_tmp(".json", json.dumps({"title": "X", "rules_required": True}))

        with redirect_stdout(io.StringIO()):
            self.api.competition_update_settings_cli(
                competition="my-comp",
                file_path=path,
                quiet=True,
            )

        mock_update.assert_called_once()
        called_updates = mock_update.call_args[0][1]
        self.assertEqual(called_updates, {"title": "X", "rules_required": True})

    @patch.object(KaggleApi, "competition_update_settings")
    def test_cli_loads_yaml_file(self, mock_update):
        mock_update.return_value = CompetitionSettings()
        path = self._write_tmp(".yaml", "title: X\nrules_required: true\n")

        with redirect_stdout(io.StringIO()):
            self.api.competition_update_settings_cli(
                competition="my-comp",
                file_path=path,
                quiet=True,
            )

        called_updates = mock_update.call_args[0][1]
        self.assertEqual(called_updates, {"title": "X", "rules_required": True})

    def test_cli_missing_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings_cli(
                competition="my-comp",
                file_path="/tmp/does-not-exist-9999.json",
            )
        self.assertIn("Settings file not found", str(ctx.exception))

    def test_cli_empty_file_raises(self):
        path = self._write_tmp(".json", "")
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings_cli(competition="my-comp", file_path=path)
        # An empty .json file → JSON parse error surfaces via "Failed to parse".
        self.assertIn("Failed to parse", str(ctx.exception))

    def test_cli_top_level_not_mapping_raises(self):
        path = self._write_tmp(".json", "[1, 2, 3]")
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings_cli(competition="my-comp", file_path=path)
        self.assertIn("mapping at the top level", str(ctx.exception))

    def test_cli_missing_competition_raises(self):
        path = self._write_tmp(".json", "{}")
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings_cli(file_path=path)
        self.assertIn("No competition specified", str(ctx.exception))

    def test_cli_missing_file_arg_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_settings_cli(competition="my-comp")
        self.assertIn("--from-file is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
