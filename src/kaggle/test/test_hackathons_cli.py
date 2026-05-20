"""Tests for ``kaggle hackathons`` CLI commands.

The two CSV-export and resolve-links endpoints are not yet shipped by any
released ``kagglesdk``, so those wrappers still import their request types
lazily. The fixture pins a ``WriteUpsClient`` onto the mocked
``kaggle.discussions`` so the resolve-links code path resolves cleanly.
"""

import argparse
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle import cli as kaggle_cli


def _install_missing_sdk_modules():
    """Inject stub modules for the two request types kagglesdk hasn't shipped yet."""
    hackathon_mod = "kagglesdk.competitions.types.hackathon_service"
    if not hasattr(sys.modules.get(hackathon_mod), "ExportHackathonWriteUpsCsvRequest"):
        mod = sys.modules.get(hackathon_mod) or types.ModuleType(hackathon_mod)
        mod.ExportHackathonWriteUpsCsvRequest = type("ExportHackathonWriteUpsCsvRequest", (), {})
        sys.modules[hackathon_mod] = mod
    writeups_mod = "kagglesdk.discussions.types.writeups_service"
    if writeups_mod not in sys.modules:
        mod = types.ModuleType(writeups_mod)
        mod.GetResolvedWriteUpLinksRequest = type("GetResolvedWriteUpLinksRequest", (), {})
        sys.modules[writeups_mod] = mod


_install_missing_sdk_modules()

# ---- Fixtures & helpers ----


@pytest.fixture
def api():
    a = KaggleApi()
    a.authenticate = MagicMock()
    mock_client = MagicMock()
    a.build_kaggle_client = MagicMock()
    a.build_kaggle_client.return_value.__enter__.return_value = mock_client
    a._mock_competitions = mock_client.competitions.competition_api_client
    a._mock_hackathon = mock_client.competitions.hackathon_client
    # The current SDK has no WriteUpsClient — pin one onto the mock so the
    # production lookup (`getattr(...)`) finds it.
    a._mock_writeups = MagicMock()
    mock_client.discussions.writeups_client = a._mock_writeups
    return a


def _make_overview_response(pages):
    r = MagicMock()
    r.pages = pages
    return r


def _make_page(name, content=""):
    p = MagicMock()
    p.name = name
    p.content = content
    return p


def _make_writeup(id, title, team_name="The Team", url="https://kaggle.com/w/1", competition_id=42):
    w = MagicMock()
    w.id = id
    team = MagicMock()
    team.name = team_name
    w.team = team
    write_up = MagicMock()
    write_up.title = title
    write_up.url = url
    w.write_up = write_up
    w.competition_id = competition_id
    w.template = False
    return w


def _make_writeups_response(writeups, total_count=0, next_page_token=""):
    r = MagicMock()
    r.hackathon_write_ups = writeups
    r.total_count = total_count
    r.next_page_token = next_page_token
    return r


def _make_csv_response(csv_body):
    r = MagicMock()
    r.csv = csv_body
    r.csv_content = None
    r.content = None
    return r


def _make_link(url, type_, title):
    link = MagicMock()
    link.url = url
    link.type = type_
    link.title = title
    return link


# ---- hackathons get ----


class TestHackathonsGet:
    def test_prints_overview_pages(self, api, capsys):
        pages = [_make_page("Overview", "Some content"), _make_page("Rules", "Be nice")]
        api._mock_competitions.get_hackathon_overview.return_value = _make_overview_response(pages)

        api.hackathon_get_overview_cli("titanic")
        out = capsys.readouterr().out
        assert "Overview" in out
        assert "Rules" in out
        assert "Some content" in out

    def test_csv_mode(self, api, capsys):
        api._mock_competitions.get_hackathon_overview.return_value = _make_overview_response(
            [_make_page("Overview", "x")]
        )
        api.hackathon_get_overview_cli("titanic", csv_display=True)
        out = capsys.readouterr().out
        assert "name" in out.lower() or "Overview" in out

    def test_no_pages(self, api, capsys):
        api._mock_competitions.get_hackathon_overview.return_value = _make_overview_response([])
        api.hackathon_get_overview_cli("titanic")
        assert "No hackathon overview pages found" in capsys.readouterr().out

    def test_missing_competition(self, api):
        with pytest.raises(ValueError, match="No competition specified"):
            api.hackathon_get_overview_cli(None)


# ---- hackathons writeups list ----


class TestHackathonsWriteupsList:
    def test_prints_writeups(self, api, capsys):
        writeups = [
            _make_writeup(1, "Best Solution"),
            _make_writeup(2, "Runner Up", team_name="Team B"),
        ]
        api._mock_competitions.list_hackathon_write_ups.return_value = _make_writeups_response(writeups, total_count=2)
        api.hackathon_list_writeups_cli("hackathon-2026")
        out = capsys.readouterr().out
        assert "Best Solution" in out
        assert "Runner Up" in out
        assert "Total: 2" in out

    def test_no_writeups(self, api, capsys):
        api._mock_competitions.list_hackathon_write_ups.return_value = _make_writeups_response([])
        api.hackathon_list_writeups_cli("hackathon-2026")
        assert "No hackathon write-ups found" in capsys.readouterr().out

    def test_csv_mode(self, api, capsys):
        api._mock_competitions.list_hackathon_write_ups.return_value = _make_writeups_response(
            [_make_writeup(1, "Best Solution")], total_count=1
        )
        api.hackathon_list_writeups_cli("hackathon-2026", csv_display=True)
        out = capsys.readouterr().out
        assert "Best Solution" in out
        assert "id" in out

    def test_quiet_suppresses_total(self, api, capsys):
        api._mock_competitions.list_hackathon_write_ups.return_value = _make_writeups_response(
            [_make_writeup(1, "A")], total_count=1, next_page_token="next"
        )
        api.hackathon_list_writeups_cli("c", quiet=True)
        out = capsys.readouterr().out
        assert "Total" not in out
        assert "Next page token" not in out

    def test_missing_competition(self, api):
        with pytest.raises(ValueError, match="No competition specified"):
            api.hackathon_list_writeups_cli(None)


# ---- hackathons writeups download ----


class TestHackathonsWriteupsDownload:
    def test_default_path(self, api, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        api._mock_hackathon.export_hackathon_write_ups_csv.return_value = _make_csv_response(
            "id,team,score\n1,Alpha,99\n"
        )
        api.hackathon_download_writeups_cli("hackathon-2026")
        out = capsys.readouterr().out
        expected = tmp_path / "hackathon-2026-writeups.csv"
        assert expected.exists()
        assert expected.read_text() == "id,team,score\n1,Alpha,99\n"
        assert "Downloaded" in out

    def test_explicit_path(self, api, tmp_path):
        api._mock_hackathon.export_hackathon_write_ups_csv.return_value = _make_csv_response("id\n1\n")
        outfile = tmp_path / "out.csv"
        api.hackathon_download_writeups_cli("hackathon-2026", path=str(outfile), quiet=True)
        assert outfile.exists()
        assert outfile.read_text() == "id\n1\n"

    def test_directory_path_appends_default_name(self, api, tmp_path):
        api._mock_hackathon.export_hackathon_write_ups_csv.return_value = _make_csv_response("id\n1\n")
        api.hackathon_download_writeups_cli("hackathon-2026", path=str(tmp_path), quiet=True)
        assert (tmp_path / "hackathon-2026-writeups.csv").exists()

    def test_empty_csv_raises(self, api):
        api._mock_hackathon.export_hackathon_write_ups_csv.return_value = _make_csv_response("")
        with pytest.raises(ValueError, match="Empty CSV"):
            api.hackathon_download_writeups_cli("hackathon-2026")

    def test_missing_competition(self, api):
        with pytest.raises(ValueError, match="No competition specified"):
            api.hackathon_download_writeups_cli(None)


# ---- hackathons writeups resolve-links ----


class TestHackathonsWriteupsResolveLinks:
    def test_prints_links(self, api, capsys):
        resp = MagicMock()
        resp.resolved_links = [
            _make_link("https://kaggle.com/datasets/x/y", "DATASET", "x/y"),
            _make_link("https://github.com/foo", "EXTERNAL", "foo"),
        ]
        api._mock_writeups.get_resolved_writeup_links.return_value = resp
        api.hackathon_resolve_writeup_links_cli("123")
        out = capsys.readouterr().out
        assert "https://kaggle.com/datasets/x/y" in out
        assert "https://github.com/foo" in out

    def test_no_links(self, api, capsys):
        resp = MagicMock()
        resp.resolved_links = []
        resp.links = []
        api._mock_writeups.get_resolved_writeup_links.return_value = resp
        api.hackathon_resolve_writeup_links_cli("123")
        assert "No links found" in capsys.readouterr().out

    def test_non_integer_id(self, api):
        with pytest.raises(ValueError, match="must be an integer"):
            api.hackathon_resolve_writeup_links_cli("not-a-number")

    def test_missing_id(self, api):
        with pytest.raises(ValueError, match="No writeup_id"):
            api.hackathon_resolve_writeup_links_cli(None)


# ---- argparse wiring ----


class TestCliArgParsing:
    """Ensure each `kaggle hackathons …` subcommand parses correctly and
    routes to the right API method.
    """

    def _make_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        with patch.object(kaggle_cli, "api", MagicMock()):
            kaggle_cli.parse_hackathons(subparsers)
        return parser

    def test_get_routes(self):
        parser = self._make_parser()
        args = parser.parse_args(["hackathons", "get", "titanic"])
        assert args.competition == "titanic"

    def test_writeups_list_routes(self):
        parser = self._make_parser()
        args = parser.parse_args(["hackathons", "writeups", "list", "titanic"])
        assert args.competition == "titanic"

    def test_writeups_download_routes(self):
        parser = self._make_parser()
        args = parser.parse_args(["hackathons", "writeups", "download", "titanic", "-p", "/tmp/out.csv"])
        assert args.competition == "titanic"
        assert args.path == "/tmp/out.csv"

    def test_writeups_resolve_links_routes(self):
        parser = self._make_parser()
        args = parser.parse_args(["hackathons", "writeups", "resolve-links", "42"])
        assert args.writeup_id == "42"

    def test_alias_h(self):
        parser = self._make_parser()
        args = parser.parse_args(["h", "get", "titanic"])
        assert args.competition == "titanic"
