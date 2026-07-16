# coding=utf-8
import io
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kagglesdk.competitions.types.competition_api_service import (
    ApiCompetition,
    ApiListCompetitionsResponse,
    ApiListCompetitionsRequest,
)
from kagglesdk.competitions.types.competition_enums import CompetitionListTab, HostSegment, CompetitionSortBy

from kaggle.api.kaggle_api_extended import KaggleApi


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    return api


def _make_competition(user_rank=784, user_has_entered=True):
    competition = ApiCompetition()
    competition.ref = "https://www.kaggle.com/competitions/example-comp"
    competition.deadline = datetime(2026, 8, 1, 0, 0, 0)
    competition.category = "Playground"
    competition.reward = "Swag"
    competition.team_count = 1200
    competition.user_has_entered = user_has_entered
    competition.user_rank = user_rank
    return competition


class TestCompetitionsList(unittest.TestCase):
    """Tests for competitions_list_cli userRank output."""

    def setUp(self):
        self.api = _make_api()

    def test_competition_fields_has_user_rank(self):
        self.assertIn("userRank", KaggleApi.competition_fields)

    @patch.object(KaggleApi, "competitions_list")
    def test_competitions_list_cli_csv_includes_user_rank_succeeds(self, mock_list):
        response = ApiListCompetitionsResponse()
        response.competitions = [_make_competition(user_rank=784)]
        mock_list.return_value = response

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.api.competitions_list_cli(group="entered", csv_display=True)

        output = mock_stdout.getvalue()
        self.assertIn("userRank", output)
        self.assertIn("784", output)

    @patch.object(KaggleApi, "competitions_list")
    def test_competitions_list_cli_csv_shows_zero_rank_succeeds(self, mock_list):
        response = ApiListCompetitionsResponse()
        response.competitions = [_make_competition(user_rank=0, user_has_entered=True)]
        mock_list.return_value = response

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.api.competitions_list_cli(group="entered", csv_display=True)

        output = mock_stdout.getvalue()
        self.assertIn("userRank", output)
        self.assertIn(",0", output.splitlines()[-1])

    @patch.object(KaggleApi, "print_results")
    @patch.object(KaggleApi, "competitions_list")
    def test_competitions_list_cli_passes_user_rank_to_print_results_succeeds(self, mock_list, mock_print_results):
        response = ApiListCompetitionsResponse()
        response.competitions = [_make_competition()]
        mock_list.return_value = response

        self.api.competitions_list_cli(group="entered")

        mock_print_results.assert_called_once()
        fields = mock_print_results.call_args[0][1]
        self.assertEqual(fields, KaggleApi.competition_fields)
        self.assertIn("userRank", fields)

    @patch.object(KaggleApi, "competitions_list")
    def test_competitions_list_cli_no_results_prints_message(self, mock_list):
        response = ApiListCompetitionsResponse()
        response.competitions = []
        mock_list.return_value = response

        with patch("builtins.print") as mock_print:
            self.api.competitions_list_cli(group="entered")

        mock_print.assert_called_once_with("No competitions found")

    def test_competitions_list_invalid_group_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competitions_list(group="invalid-group")
        self.assertIn("Invalid group specified", str(context.exception))

    def test_competitions_list_invalid_category_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competitions_list(category="invalid-cat")
        self.assertIn("Invalid category specified", str(context.exception))

    def test_competitions_list_invalid_sort_by_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competitions_list(sort_by="invalid-sort")
        self.assertIn("Invalid sort_by specified", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competitions_list_defaults_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = ApiListCompetitionsResponse()
        mock_kaggle.competitions.competition_api_client.list_competitions.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        response = self.api.competitions_list()

        self.assertEqual(response, mock_response)
        mock_kaggle.competitions.competition_api_client.list_competitions.assert_called_once()
        request = mock_kaggle.competitions.competition_api_client.list_competitions.call_args[0][0]

        self.assertEqual(request.group, CompetitionListTab.COMPETITION_LIST_TAB_EVERYTHING)
        self.assertEqual(request.category, HostSegment.HOST_SEGMENT_UNSPECIFIED)
        self.assertEqual(request.sort_by, CompetitionSortBy.COMPETITION_SORT_BY_BEST)
        self.assertEqual(request.page, 1)
        self.assertEqual(request.search, "")
        self.assertEqual(request.page_size, 20)
        self.assertEqual(request.page_token, "")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competitions_list_custom_filters_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.list_competitions.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.competitions_list(
            group="entered",
            category="featured",
            sort_by="prize",
            page=3,
            search="terms",
            page_size=50,
            page_token="token-abc",
        )

        request = mock_kaggle.competitions.competition_api_client.list_competitions.call_args[0][0]
        self.assertEqual(request.group, CompetitionListTab.COMPETITION_LIST_TAB_ENTERED)
        self.assertEqual(request.category, HostSegment.HOST_SEGMENT_FEATURED)
        self.assertEqual(request.sort_by, CompetitionSortBy.COMPETITION_SORT_BY_PRIZE)
        self.assertEqual(request.page, 3)
        self.assertEqual(request.search, "terms")
        self.assertEqual(request.page_size, 50)
        self.assertEqual(request.page_token, "token-abc")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competitions_list_success_group_all(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.list_competitions.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.competitions_list(group="all")

        request = mock_kaggle.competitions.competition_api_client.list_competitions.call_args[0][0]
        self.assertEqual(request.group, CompetitionListTab.COMPETITION_LIST_TAB_EVERYTHING)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competitions_list_category_all_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.list_competitions.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.competitions_list(category="all")

        request = mock_kaggle.competitions.competition_api_client.list_competitions.call_args[0][0]
        self.assertEqual(request.category, HostSegment.HOST_SEGMENT_UNSPECIFIED)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competitions_list_page_token_precludes_page_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.list_competitions.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.competitions_list(page_token="token-xyz")

        request = mock_kaggle.competitions.competition_api_client.list_competitions.call_args[0][0]
        self.assertEqual(request.page, 0)
        self.assertEqual(request.page_token, "token-xyz")


if __name__ == "__main__":
    unittest.main()
