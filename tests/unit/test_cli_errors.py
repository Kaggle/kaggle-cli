# coding=utf-8
import argparse
import io
import sys
import unittest
from unittest.mock import patch

from requests.exceptions import HTTPError
from requests.models import Response

import kaggle.cli as cli
from kaggle.api.kaggle_api_extended import format_http_error


def _http_error(status_code):
    response = Response()
    response.status_code = status_code
    response.url = "https://www.kaggle.com/api/v1/competitions/nope"
    return HTTPError(f"{status_code} Client Error for url: {response.url}", response=response)


def _run_main(argv, func):
    """Runs cli.main() with a stubbed-out command implementation."""
    parse_args = argparse.ArgumentParser.parse_args

    def parse_args_with_stub(self, *args, **kwargs):
        namespace = parse_args(self, *args, **kwargs)
        namespace.func = func
        return namespace

    stdout, stderr = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", argv):
        with patch("kaggle.cli.api") as mock_api:
            mock_api._authenticated = True
            with patch.object(argparse.ArgumentParser, "parse_args", parse_args_with_stub):
                with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                    try:
                        cli.main()
                        exit_code = 0
                    except SystemExit as e:
                        exit_code = e.code
    return exit_code, stdout.getvalue(), stderr.getvalue()


class TestFormatHttpError(unittest.TestCase):
    def test_keeps_the_original_message(self):
        self.assertIn("404 Client Error", format_http_error(_http_error(404)))

    def test_404_suggests_checking_the_reference(self):
        self.assertIn("kaggle search", format_http_error(_http_error(404)))

    def test_403_mentions_competition_rules(self):
        self.assertIn("rules", format_http_error(_http_error(403)))

    def test_429_explains_rate_limiting(self):
        self.assertIn("rate limited", format_http_error(_http_error(429)))

    def test_5xx_is_attributed_to_kaggle(self):
        self.assertIn("Kaggle's side", format_http_error(_http_error(503)))

    def test_status_without_a_hint_is_left_alone(self):
        self.assertEqual(
            "400 Client Error for url: " + _http_error(400).response.url, format_http_error(_http_error(400))
        )

    def test_error_without_a_response_is_left_alone(self):
        self.assertEqual("boom", format_http_error(HTTPError("boom")))


class TestMainErrorHandling(unittest.TestCase):
    def test_http_error_prints_hint_and_exits_nonzero(self):
        def raise_404(**kwargs):
            raise _http_error(404)

        exit_code, _, stderr = _run_main(["kaggle", "quota"], raise_404)

        self.assertEqual(1, exit_code)
        self.assertIn("404 Client Error", stderr)
        self.assertIn("kaggle search", stderr)

    def test_401_still_prints_auth_help(self):
        def raise_401(**kwargs):
            raise _http_error(401)

        exit_code, stdout, _ = _run_main(["kaggle", "quota"], raise_401)

        self.assertEqual(1, exit_code)
        self.assertIn("Authentication required", stdout)

    def test_unexpected_error_reports_a_bug_instead_of_a_traceback(self):
        def raise_bug(**kwargs):
            raise KeyError("some_missing_field")

        exit_code, _, stderr = _run_main(["kaggle", "quota"], raise_bug)

        self.assertEqual(1, exit_code)
        self.assertIn("KeyError", stderr)
        self.assertIn("--debug", stderr)
        self.assertIn("github.com/Kaggle/kaggle-cli/issues", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_debug_flag_lets_the_traceback_through(self):
        def raise_bug(**kwargs):
            raise KeyError("some_missing_field")

        with self.assertRaises(KeyError):
            _run_main(["kaggle", "--debug", "quota"], raise_bug)

    def test_debug_flag_is_not_passed_to_the_command(self):
        received = {}

        def record(**kwargs):
            received.update(kwargs)
            return None

        _run_main(["kaggle", "--debug", "quota"], record)

        self.assertNotIn("debug", received)

    def test_value_error_is_still_reported_without_a_bug_notice(self):
        def raise_value_error(**kwargs):
            raise ValueError("bad input")

        exit_code, _, stderr = _run_main(["kaggle", "quota"], raise_value_error)

        self.assertEqual(1, exit_code)
        self.assertIn("bad input", stderr)
        self.assertNotIn("bug in the Kaggle CLI", stderr)


class TestHelpExamples(unittest.TestCase):
    def test_examples_cover_the_common_first_commands(self):
        examples = cli.Help.examples

        self.assertIn("kaggle auth login", examples)
        self.assertIn("kaggle competitions download -c titanic", examples)
        self.assertIn("kaggle competitions submit", examples)


if __name__ == "__main__":
    unittest.main()
