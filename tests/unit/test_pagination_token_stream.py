# coding=utf-8
"""Tests that a pagination token never corrupts machine-readable output.

``--format json`` and ``--format csv`` are documented, machine-readable
formats, so stdout has to stay a single parseable document. The "Next Page
Token = ..." notice therefore goes to stderr for those formats, while the
human-readable table keeps it on stdout where it has always been.
"""

import csv
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi

TOKEN = "TOKEN123"
TOKEN_LINE = "Next Page Token = " + TOKEN


class _Obj:
    """A tiny stand-in for an SDK response item with real (printable) values."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _run(func, *args, **kwargs):
    """Calls func, returning its (stdout, stderr) output as a pair of strings."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        func(*args, **kwargs)
    return out.getvalue(), err.getvalue()


class TestPrintNextPageToken(unittest.TestCase):
    """Unit tests for the print_next_page_token helper itself."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def test_table_keeps_token_on_stdout(self):
        out, err = _run(self.api.print_next_page_token, TOKEN)
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_explicit_table_format_keeps_token_on_stdout(self):
        out, err = _run(self.api.print_next_page_token, TOKEN, output_format="table")
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_json_sends_token_to_stderr(self):
        out, err = _run(self.api.print_next_page_token, TOKEN, output_format="json")
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_json_projection_sends_token_to_stderr(self):
        out, err = _run(self.api.print_next_page_token, TOKEN, output_format="json(name,size)")
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_csv_format_sends_token_to_stderr(self):
        out, err = _run(self.api.print_next_page_token, TOKEN, output_format="csv")
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_legacy_csv_display_sends_token_to_stderr(self):
        out, err = _run(self.api.print_next_page_token, TOKEN, csv_display=True)
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_unrecognized_format_keeps_token_on_stdout(self):
        # _get_output_format falls back to a table for anything it cannot parse.
        out, err = _run(self.api.print_next_page_token, TOKEN, output_format="bogus")
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_non_string_token_is_printed(self):
        # "competitions list-topics" pages by number rather than by opaque token.
        out, err = _run(self.api.print_next_page_token, 2, output_format="json")
        self.assertEqual("", out)
        self.assertEqual("Next Page Token = 2", err.strip())

    def test_empty_token_prints_nothing(self):
        for token in (None, "", 0):
            out, err = _run(self.api.print_next_page_token, token, output_format="json")
            self.assertEqual("", out, "token %r" % (token,))
            self.assertEqual("", err, "token %r" % (token,))


class _CommandCase(unittest.TestCase):
    """Shared assertions covering the three output formats of a command."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.already_printed_version_warning = True

    def assert_json_is_parseable(self, invoke, expected_rows):
        out, err = invoke("json")
        self.assertNotIn("Next Page Token", out)
        self.assertEqual(TOKEN_LINE, err.strip())
        self.assertEqual(expected_rows, len(json.loads(out)))

    def assert_csv_is_parseable(self, invoke, expected_rows):
        out, err = invoke("csv")
        self.assertNotIn("Next Page Token", out)
        self.assertEqual(TOKEN_LINE, err.strip())
        rows = [row for row in csv.reader(io.StringIO(out)) if row]
        self.assertEqual(expected_rows + 1, len(rows))  # + header row

    def assert_table_keeps_token(self, invoke):
        out, err = invoke("table")
        self.assertIn(TOKEN_LINE, out)
        self.assertEqual("", err)


class TestDatasetListFilesTokenStream(_CommandCase):
    """Regression test for the originally reported command."""

    def _invoke(self, output_format):
        result = MagicMock()
        result.error_message = None
        result.next_page_token = TOKEN
        result.files = [_Obj(name="train.csv", total_bytes=10, size=10, creation_date="2024-01-01")]
        with patch.object(KaggleApi, "dataset_list_files", return_value=result):
            return _run(self.api.dataset_list_files_cli, "owner/dataset", output_format=output_format)

    def test_json(self):
        self.assert_json_is_parseable(self._invoke, 1)

    def test_csv(self):
        self.assert_csv_is_parseable(self._invoke, 1)

    def test_table(self):
        self.assert_table_keeps_token(self._invoke)


class TestDatasetListTokenStream(_CommandCase):
    def _invoke(self, output_format):
        response = MagicMock()
        response.next_page_token = TOKEN
        response.datasets = [
            _Obj(
                ref="owner/dataset",
                title="A dataset",
                total_bytes=10,
                last_updated="2024-01-01",
                download_count=1,
                vote_count=2,
                usability_rating=0.5,
            )
        ]
        with patch.object(KaggleApi, "dataset_list_with_response", return_value=response):
            return _run(self.api.dataset_list_cli, output_format=output_format)

    def test_json(self):
        self.assert_json_is_parseable(self._invoke, 1)

    def test_csv(self):
        self.assert_csv_is_parseable(self._invoke, 1)

    def test_table(self):
        self.assert_table_keeps_token(self._invoke)


class TestCompetitionListFilesTokenStream(_CommandCase):
    def _invoke(self, output_format):
        result = MagicMock()
        result.next_page_token = TOKEN
        result.files = [_Obj(name="train.csv", total_bytes=10, creation_date="2024-01-01")]
        with patch.object(KaggleApi, "competition_list_files", return_value=result):
            return _run(
                self.api.competition_list_files_cli,
                "titanic",
                quiet=True,
                output_format=output_format,
            )

    def test_json(self):
        self.assert_json_is_parseable(self._invoke, 1)

    def test_csv(self):
        self.assert_csv_is_parseable(self._invoke, 1)

    def test_table(self):
        self.assert_table_keeps_token(self._invoke)


class TestKernelsListFilesTokenStream(_CommandCase):
    def _invoke(self, output_format):
        result = MagicMock()
        result.nextPageToken = TOKEN
        result.files = [_Obj(name="script.py", size=10, creation_date="2024-01-01")]
        with patch.object(KaggleApi, "kernels_list_files", return_value=result):
            return _run(self.api.kernels_list_files_cli, "owner/kernel", output_format=output_format)

    def test_json(self):
        self.assert_json_is_parseable(self._invoke, 1)

    def test_csv(self):
        self.assert_csv_is_parseable(self._invoke, 1)

    def test_table(self):
        self.assert_table_keeps_token(self._invoke)


class TestModelListTokenStream(_CommandCase):
    """model_list prints the token itself, so the format has to reach it."""

    def _invoke(self, output_format):
        response = MagicMock()
        response.next_page_token = TOKEN
        response.models = [_Obj(id=1, ref="owner/model", title="A model", subtitle="", author="owner")]
        kaggle = MagicMock()
        kaggle.models.model_api_client.list_models.return_value = response
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=kaggle)
        client.__exit__ = MagicMock(return_value=False)
        with patch.object(KaggleApi, "build_kaggle_client", return_value=client):
            return _run(self.api.model_list_cli, output_format=output_format)

    def test_json(self):
        self.assert_json_is_parseable(self._invoke, 1)

    def test_csv(self):
        self.assert_csv_is_parseable(self._invoke, 1)

    def test_table(self):
        self.assert_table_keeps_token(self._invoke)

    def test_library_call_without_a_format_keeps_token_on_stdout(self):
        """Direct callers of the library method keep the pre-existing behavior."""
        out, err = self._invoke(None)
        self.assertIn(TOKEN_LINE, out)
        self.assertEqual("", err)


class TestModelInstancesListTokenStream(_CommandCase):
    def _invoke(self, output_format):
        response = MagicMock()
        response.next_page_token = TOKEN
        response.instances = [
            _Obj(version_number=1, version_notes="notes", creation_status="ok", total_uncompressed_bytes=10)
        ]
        with patch.object(KaggleApi, "model_instances_list", return_value=response):
            return _run(self.api.model_instances_list_cli, "owner/model", output_format=output_format)

    def test_json(self):
        self.assert_json_is_parseable(self._invoke, 1)

    def test_csv(self):
        self.assert_csv_is_parseable(self._invoke, 1)

    def test_table(self):
        self.assert_table_keeps_token(self._invoke)


class TestCompetitionLeaderboardTokenStream(_CommandCase):
    """competition_leaderboard_view prints the token, so the format has to reach it."""

    def _invoke(self, output_format):
        response = MagicMock()
        response.next_page_token = TOKEN
        response.submissions = [_Obj(team_id=1, team_name="A team", submission_date="2024-01-01", score="0.5")]
        kaggle = MagicMock()
        kaggle.competitions.competition_api_client.get_leaderboard.return_value = response
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=kaggle)
        client.__exit__ = MagicMock(return_value=False)
        with patch.object(KaggleApi, "build_kaggle_client", return_value=client):
            return _run(
                self.api.competition_leaderboard_cli,
                "titanic",
                view=True,
                quiet=True,
                output_format=output_format,
            )

    def test_json(self):
        self.assert_json_is_parseable(self._invoke, 1)

    def test_csv(self):
        self.assert_csv_is_parseable(self._invoke, 1)

    def test_table(self):
        self.assert_table_keeps_token(self._invoke)


if __name__ == "__main__":
    unittest.main()
