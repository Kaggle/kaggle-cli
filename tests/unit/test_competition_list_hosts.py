# coding=utf-8
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiCompetitionHost


def _make_host(id_, user_name, display_name="", profile_url=""):
    h = ApiCompetitionHost()
    h.id = id_
    h.user_name = user_name
    h.display_name = display_name
    h.profile_url = profile_url
    return h


class TestCompetitionListHosts(unittest.TestCase):
    """Tests for competition_list_hosts and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}

    def _patch_client(self, mock_client, hosts=None):
        mock_kaggle = MagicMock()
        response = MagicMock()
        response.hosts = hosts if hosts is not None else []
        mock_kaggle.competitions.competition_api_client.list_competition_hosts.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_list_hosts_builds_request_with_competition_name(self, mock_client):
        mock_kaggle = self._patch_client(mock_client, [_make_host(1, "alice")])

        result = self.api.competition_list_hosts("my-comp")

        request = mock_kaggle.competitions.competition_api_client.list_competition_hosts.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].user_name, "alice")

    @patch.object(KaggleApi, "competition_list_hosts")
    def test_cli_positional_competition(self, mock_list):
        mock_list.return_value = [_make_host(1, "alice")]
        with redirect_stdout(io.StringIO()):
            self.api.competition_list_hosts_cli(competition="my-comp")
        mock_list.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_list_hosts")
    def test_cli_dash_c_option(self, mock_list):
        mock_list.return_value = []
        with redirect_stdout(io.StringIO()):
            self.api.competition_list_hosts_cli(competition_opt="my-comp")
        mock_list.assert_called_once_with("my-comp")

    @patch.object(KaggleApi, "competition_list_hosts")
    def test_cli_falls_back_to_config(self, mock_list):
        mock_list.return_value = []
        self.api.config_values = {self.api.CONFIG_NAME_COMPETITION: "from-config"}
        with redirect_stdout(io.StringIO()):
            self.api.competition_list_hosts_cli()
        mock_list.assert_called_once_with("from-config")

    def test_cli_missing_competition_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_list_hosts_cli()
        self.assertIn("No competition specified", str(ctx.exception))

    @patch.object(KaggleApi, "competition_list_hosts")
    def test_cli_empty_prints_no_hosts_found(self, mock_list):
        mock_list.return_value = []
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_list_hosts_cli(competition="my-comp")
        self.assertIn("No hosts found", buf.getvalue())

    @patch.object(KaggleApi, "competition_list_hosts")
    def test_cli_table_output_lists_hosts(self, mock_list):
        mock_list.return_value = [
            _make_host(42, "alice", "Alice A", "https://www.kaggle.com/alice"),
            _make_host(99, "bob", "Bob B", "https://www.kaggle.com/bob"),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_list_hosts_cli(competition="my-comp")
        out = buf.getvalue()
        self.assertIn("userName", out)
        self.assertIn("displayName", out)
        self.assertIn("profileUrl", out)
        self.assertIn("alice", out)
        self.assertIn("bob", out)
        self.assertIn("42", out)

    @patch.object(KaggleApi, "competition_list_hosts")
    def test_cli_csv_output(self, mock_list):
        mock_list.return_value = [_make_host(42, "alice", "Alice A", "https://www.kaggle.com/alice")]
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.api.competition_list_hosts_cli(competition="my-comp", csv_display=True)
        out = buf.getvalue()
        # CSV header line
        self.assertIn("userName,displayName,id,profileUrl", out)
        self.assertIn("alice,Alice A,42,https://www.kaggle.com/alice", out)


if __name__ == "__main__":
    unittest.main()
