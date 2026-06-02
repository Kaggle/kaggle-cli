# coding=utf-8
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "..")

from kaggle.api.kaggle_api_extended import KaggleApi


def _mock_submission(id_, date_iso, public_score):
    sub = MagicMock()
    sub.id = id_
    sub.date_submitted = date_iso
    sub.public_score = public_score
    return sub


def _build_response(submissions):
    response = MagicMock()
    response.submissions = submissions
    return response


class TestTeamPublicSubmissions(unittest.TestCase):
    """Tests for competition_team_submissions and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competition_team_submissions_returns_list(self, mock_client):
        expected = [_mock_submission(1, "2026-01-01T00:00:00Z", "50.5")]
        response = _build_response(expected)
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.list_team_public_submissions.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.competition_team_submissions(team_id=42)

        self.assertEqual(result, expected)
        called_request = mock_kaggle.competitions.competition_api_client.list_team_public_submissions.call_args[0][0]
        self.assertEqual(called_request.team_id, 42)

    @patch.object(KaggleApi, "competition_team_submissions")
    def test_cli_table_output(self, mock_view):
        mock_view.return_value = [
            _mock_submission(11, "2026-01-02T00:00:00Z", "100.0"),
            _mock_submission(22, "2026-01-01T00:00:00Z", "50.0"),
        ]

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.competition_team_submissions_cli(team_id=42)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("11", output)
        self.assertIn("22", output)
        self.assertIn("100.0", output)

    @patch.object(KaggleApi, "competition_team_submissions")
    def test_cli_csv_output(self, mock_view):
        mock_view.return_value = [_mock_submission(11, "2026-01-02T00:00:00Z", "100.0")]

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.competition_team_submissions_cli(team_id=42, csv_display=True)
        finally:
            sys.stdout = sys.__stdout__

        lines = [line for line in captured.getvalue().splitlines() if line]
        self.assertEqual(lines[0], "id,dateSubmitted,publicScore")
        self.assertEqual(lines[1], "11,2026-01-02T00:00:00Z,100.0")

    @patch.object(KaggleApi, "competition_team_submissions")
    def test_cli_empty(self, mock_view):
        mock_view.return_value = []

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.competition_team_submissions_cli(team_id=42)
        finally:
            sys.stdout = sys.__stdout__

        self.assertIn("No submissions found", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
