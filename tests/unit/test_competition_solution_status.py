# coding=utf-8
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiCompetitionSolutionStatus,
    ApiMetricColumn,
    ApiSolutionFileInfo,
)


def _make_status(**overrides) -> ApiCompetitionSolutionStatus:
    status = ApiCompetitionSolutionStatus()
    for name, value in overrides.items():
        setattr(status, name, value)
    return status


class TestCompetitionSolutionStatus(unittest.TestCase):
    """Tests for competition_get_solution_status and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}

    def _patch_client(self, mock_client, status=None):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.get_competition_solution_status.return_value = (
            status if status is not None else _make_status()
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_get_status_builds_request_with_competition_name(self, mock_client):
        mock_kaggle = self._patch_client(mock_client, _make_status(ready=True))

        result = self.api.competition_get_solution_status("my-comp")

        request = mock_kaggle.competitions.competition_api_client.get_competition_solution_status.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertTrue(result.ready)

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_positional_competition(self, mock_get):
        mock_get.return_value = _make_status(ready=True)
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_solution_status_cli(competition="my-comp")
        mock_get.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_dash_c_option(self, mock_get):
        mock_get.return_value = _make_status()
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_solution_status_cli(competition_opt="my-comp")
        mock_get.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_falls_back_to_config(self, mock_get):
        mock_get.return_value = _make_status()
        self.api.config_values = {self.api.CONFIG_NAME_COMPETITION: "from-config"}
        with redirect_stdout(io.StringIO()):
            self.api.competition_get_solution_status_cli()
        mock_get.assert_called_once_with("from-config")

    def test_cli_missing_competition_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_get_solution_status_cli()
        self.assertIn("No competition specified", str(ctx.exception))

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_json_output_emits_valid_json(self, mock_get):
        mock_get.return_value = _make_status(ready=True, setup_error="bad column")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_solution_status_cli(competition="my-comp", json_output=True)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["setupError"], "bad column")

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_human_view_shows_ready_false_by_default(self, mock_get):
        mock_get.return_value = _make_status()
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_solution_status_cli(competition="my-comp")
        self.assertIn("Ready: false", buf.getvalue())

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_human_view_surfaces_setup_error(self, mock_get):
        # Even if the server also reports ready=True, a setup_error must flip
        # the Ready line so polling scripts eyeballing the first line stop.
        mock_get.return_value = _make_status(ready=True, setup_error="missing header row")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_solution_status_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("Ready: false (setup failed)", out)
        self.assertNotIn("Ready: true", out)
        self.assertIn("Setup error: missing header row", out)

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_human_view_prints_kernels_and_row_id(self, mock_get):
        mock_get.return_value = _make_status(
            ready=True,
            kernels_metric=True,
            row_id_column_name="row_id",
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_solution_status_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("Ready: true", out)
        self.assertIn("Kernels metric: true", out)
        self.assertIn("Row ID column: row_id", out)

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_human_view_prints_column_mapping_and_required_columns(self, mock_get):
        col = ApiMetricColumn()
        col.name = "target"
        col.data_type = "Double"
        status = _make_status(required_metric_columns=[col])
        # column_mapping setter in the SDK has a validation bug; assign the
        # private attribute directly to sidestep it.
        object.__setattr__(status, "_column_mapping", {"target": "y"})
        mock_get.return_value = status
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_solution_status_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("Column mapping:", out)
        self.assertIn("target -> y", out)
        self.assertIn("Required columns:", out)
        self.assertIn("target (Double)", out)

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_human_view_prints_solution_info(self, mock_get):
        info = ApiSolutionFileInfo()
        info.file_name = "solution.csv"
        info.file_size_bytes = 1234
        info.upload_date = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
        info.total_rows = 100
        info.public_rows = 30
        info.private_rows = 70
        mock_get.return_value = _make_status(ready=True, solution_info=info)

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_solution_status_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("solution.csv", out)
        self.assertIn("uploaded 2026-07-21T12:00:00+00:00", out)
        self.assertIn("total=100", out)
        self.assertIn("public=30", out)
        self.assertIn("private=70", out)

    @patch.object(KaggleApi, "competition_get_solution_status")
    def test_cli_human_view_skips_bare_solution_file_label(self, mock_get):
        # If the server ever returns solution_info populated only with
        # row-count fields (partial preprocessing state), we should suppress
        # the "Solution file:" header entirely rather than print a bare label.
        info = ApiSolutionFileInfo()
        info.total_rows = 100
        info.public_rows = 30
        info.private_rows = 70
        mock_get.return_value = _make_status(ready=True, solution_info=info)

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_get_solution_status_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertNotIn("Solution file:", out)


if __name__ == "__main__":
    unittest.main()
