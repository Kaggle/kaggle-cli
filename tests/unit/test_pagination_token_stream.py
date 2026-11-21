# coding=utf-8
"""Tests that a pagination token can be kept out of machine-readable output.

``--format json`` and ``--format csv`` are documented, machine-readable
formats, but the "Next Page Token = ..." notice is printed before the results,
so stdout is not a single parseable document when more results exist. Setting
``KAGGLE_PAGE_TOKEN_STDERR`` opts in to reporting that notice on stderr for
those formats. The variable is opt-in, so with it unset every format keeps the
notice on stdout exactly as it always has, and the human-readable table keeps
it on stdout either way.
"""

import csv
import io
import json
import os
import sys
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi, PAGE_TOKEN_STDERR_ENV_VAR

TOKEN = "TOKEN123"
TOKEN_LINE = "Next Page Token = " + TOKEN


class _Obj:
    """A tiny stand-in for an SDK response item with real (printable) values."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@contextmanager
def opted_in(value="1"):
    """Runs the block with KAGGLE_PAGE_TOKEN_STDERR set."""
    with patch.dict(os.environ, {PAGE_TOKEN_STDERR_ENV_VAR: value}):
        yield


@contextmanager
def opted_out():
    """Runs the block with KAGGLE_PAGE_TOKEN_STDERR unset.

    Set explicitly rather than assumed, so that the default-behavior tests hold
    even when the developer running them has opted in for their own shell.
    """
    with patch.dict(os.environ):
        os.environ.pop(PAGE_TOKEN_STDERR_ENV_VAR, None)
        yield


def _run(func, *args, **kwargs):
    """Calls func, returning its (stdout, stderr) output as a pair of strings."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        func(*args, **kwargs)
    return out.getvalue(), err.getvalue()


class TestPrintNextPageTokenOptedIn(unittest.TestCase):
    """The helper's behavior once KAGGLE_PAGE_TOKEN_STDERR is set."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def _run(self, *args, **kwargs):
        with opted_in():
            return _run(self.api.print_next_page_token, *args, **kwargs)

    def test_table_keeps_token_on_stdout(self):
        out, err = self._run(TOKEN)
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_explicit_table_format_keeps_token_on_stdout(self):
        out, err = self._run(TOKEN, output_format="table")
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_json_sends_token_to_stderr(self):
        out, err = self._run(TOKEN, output_format="json")
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_json_projection_sends_token_to_stderr(self):
        out, err = self._run(TOKEN, output_format="json(name,size)")
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_csv_format_sends_token_to_stderr(self):
        out, err = self._run(TOKEN, output_format="csv")
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_legacy_csv_display_sends_token_to_stderr(self):
        out, err = self._run(TOKEN, csv_display=True)
        self.assertEqual("", out)
        self.assertEqual(TOKEN_LINE, err.strip())

    def test_unrecognized_format_keeps_token_on_stdout(self):
        # _get_output_format falls back to a table for anything it cannot parse.
        out, err = self._run(TOKEN, output_format="bogus")
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_non_string_token_is_printed(self):
        # "competitions list-topics" pages by number rather than by opaque token.
        out, err = self._run(2, output_format="json")
        self.assertEqual("", out)
        self.assertEqual("Next Page Token = 2", err.strip())

    def test_empty_token_prints_nothing(self):
        for token in (None, "", 0):
            out, err = self._run(token, output_format="json")
            self.assertEqual("", out, "token %r" % (token,))
            self.assertEqual("", err, "token %r" % (token,))

    def test_every_documented_truthy_value_opts_in(self):
        for value in ("1", "true", "TRUE", "True", "yes", "YES"):
            with opted_in(value):
                out, err = _run(self.api.print_next_page_token, TOKEN, output_format="json")
            self.assertEqual("", out, "value %r" % (value,))
            self.assertEqual(TOKEN_LINE, err.strip(), "value %r" % (value,))


class TestPrintNextPageTokenDefault(unittest.TestCase):
    """Without the variable the helper must not change any existing output."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def _run(self, *args, **kwargs):
        with opted_out():
            return _run(self.api.print_next_page_token, *args, **kwargs)

    def test_json_keeps_token_on_stdout(self):
        out, err = self._run(TOKEN, output_format="json")
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_json_projection_keeps_token_on_stdout(self):
        out, err = self._run(TOKEN, output_format="json(name,size)")
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_csv_keeps_token_on_stdout(self):
        out, err = self._run(TOKEN, output_format="csv")
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_legacy_csv_display_keeps_token_on_stdout(self):
        out, err = self._run(TOKEN, csv_display=True)
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_table_keeps_token_on_stdout(self):
        out, err = self._run(TOKEN)
        self.assertEqual(TOKEN_LINE, out.strip())
        self.assertEqual("", err)

    def test_falsy_values_do_not_opt_in(self):
        for value in ("", "0", "false", "no", "off", "maybe"):
            with opted_in(value):
                out, err = _run(self.api.print_next_page_token, TOKEN, output_format="json")
            self.assertEqual(TOKEN_LINE, out.strip(), "value %r" % (value,))
            self.assertEqual("", err, "value %r" % (value,))

    def test_empty_token_prints_nothing(self):
        for token in (None, "", 0):
            out, err = self._run(token, output_format="json")
            self.assertEqual("", out, "token %r" % (token,))
            self.assertEqual("", err, "token %r" % (token,))


class _CommandCase(unittest.TestCase):
    """Shared assertions covering the output formats of a paginated command."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.already_printed_version_warning = True

    def assert_json_is_parseable(self, invoke, expected_rows):
        """Opted in, stdout is a JSON document and the token is on stderr."""
        with opted_in():
            out, err = invoke("json")
        self.assertNotIn("Next Page Token", out)
        self.assertEqual(TOKEN_LINE, err.strip())
        self.assertEqual(expected_rows, len(json.loads(out)))

    def assert_csv_is_parseable(self, invoke, expected_rows):
        """Opted in, stdout is a CSV document and the token is on stderr."""
        with opted_in():
            out, err = invoke("csv")
        self.assertNotIn("Next Page Token", out)
        self.assertEqual(TOKEN_LINE, err.strip())
        rows = [row for row in csv.reader(io.StringIO(out)) if row]
        self.assertEqual(expected_rows + 1, len(rows))  # + header row

    def assert_table_keeps_token(self, invoke):
        """The table format keeps the token on stdout even when opted in."""
        with opted_in():
            out, err = invoke("table")
        self.assertIn(TOKEN_LINE, out)
        self.assertEqual("", err)

    def assert_default_keeps_token_on_stdout(self, invoke):
        """Without the variable, json and csv stdout is unchanged."""
        for output_format in ("json", "csv", "table", None):
            with opted_out():
                out, err = invoke(output_format)
            self.assertIn(TOKEN_LINE, out, "format %r" % (output_format,))
            self.assertEqual("", err, "format %r" % (output_format,))


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

    def test_default(self):
        self.assert_default_keeps_token_on_stdout(self._invoke)


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

    def test_default(self):
        self.assert_default_keeps_token_on_stdout(self._invoke)


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

    def test_default(self):
        self.assert_default_keeps_token_on_stdout(self._invoke)


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

    def test_default(self):
        self.assert_default_keeps_token_on_stdout(self._invoke)


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

    def test_default(self):
        self.assert_default_keeps_token_on_stdout(self._invoke)

    def test_library_call_without_a_format_keeps_token_on_stdout(self):
        """Direct callers of the library method keep the pre-existing behavior."""
        with opted_in():
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

    def test_default(self):
        self.assert_default_keeps_token_on_stdout(self._invoke)


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

    def test_default(self):
        self.assert_default_keeps_token_on_stdout(self._invoke)


if __name__ == "__main__":
    unittest.main()
