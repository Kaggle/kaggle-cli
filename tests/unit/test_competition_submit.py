# coding=utf-8
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch
import sys

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.models.kaggle_models_extended import ResumableUploadResult
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest,
    ApiCreateSubmissionRequest,
    ApiCreateSubmissionResponse,
    ApiCreateCodeSubmissionRequest,
    ApiCreateCodeSubmissionResponse,
)


class TestCompetitionSubmit(unittest.TestCase):
    """Tests for competition_submit and competition_submit_code."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    # competition_submit tests
    def test_submit_missing_competition_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit("file.csv", "message", None)
        self.assertIn("No competition specified", str(context.exception))

    def test_submit_missing_file_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit(None, "message", "comp-name")
        self.assertIn("No file specified", str(context.exception))

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_submit_valid_file_succeeds(self, mock_client, mock_upload_complete):
        mock_kaggle = MagicMock()

        mock_upload_response = MagicMock()
        mock_upload_response.create_url = "http://upload-url"
        mock_upload_response.token = "token-123"
        mock_kaggle.competitions.competition_api_client.start_submission_upload.return_value = mock_upload_response

        mock_submit_response = ApiCreateSubmissionResponse()
        mock_submit_response.message = "Submission successful"
        mock_kaggle.competitions.competition_api_client.create_submission.return_value = mock_submit_response

        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        dummy_file = os.path.join(self.temp_dir, "submission.csv")
        with open(dummy_file, "w") as f:
            f.write("data")

        response = self.api.competition_submit(dummy_file, "my message", "comp-name", quiet=True, sandbox=True)

        self.assertEqual(response, mock_submit_response)

        mock_kaggle.competitions.competition_api_client.start_submission_upload.assert_called_once()
        upload_request = mock_kaggle.competitions.competition_api_client.start_submission_upload.call_args[0][0]
        self.assertEqual(upload_request.competition_name, "comp-name")
        self.assertEqual(upload_request.file_name, "submission.csv")
        self.assertEqual(upload_request.content_length, 4)

        mock_upload_complete.assert_called_once_with(dummy_file, "http://upload-url", True)

        mock_kaggle.competitions.competition_api_client.create_submission.assert_called_once()
        submit_request = mock_kaggle.competitions.competition_api_client.create_submission.call_args[0][0]
        self.assertEqual(submit_request.competition_name, "comp-name")
        self.assertEqual(submit_request.blob_file_tokens, "token-123")
        self.assertEqual(submit_request.submission_description, "my message")
        self.assertTrue(submit_request.sandbox)

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.FAILED)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_submit_upload_failure_fails(self, mock_client, mock_upload_complete):
        mock_kaggle = MagicMock()
        mock_upload_response = MagicMock()
        mock_upload_response.create_url = "http://upload-url"
        mock_kaggle.competitions.competition_api_client.start_submission_upload.return_value = mock_upload_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        dummy_file = os.path.join(self.temp_dir, "submission.csv")
        open(dummy_file, "w").close()

        response = self.api.competition_submit(dummy_file, "message", "comp-name")

        self.assertEqual(response.message, "Could not submit to competition")
        mock_kaggle.competitions.competition_api_client.create_submission.assert_not_called()

    @patch.object(KaggleApi, "get_config_value")
    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_submit_fallback_config_succeeds(self, mock_client, mock_upload_complete, mock_get_config):
        mock_get_config.return_value = "config-comp"
        mock_kaggle = MagicMock()
        mock_upload_response = MagicMock()
        mock_upload_response.token = "token-config"
        mock_kaggle.competitions.competition_api_client.start_submission_upload.return_value = mock_upload_response
        mock_kaggle.competitions.competition_api_client.create_submission.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        dummy_file = os.path.join(self.temp_dir, "submission.csv")
        open(dummy_file, "w").close()

        self.api.competition_submit(dummy_file, "message", None, quiet=True)

        mock_get_config.assert_called_with(KaggleApi.CONFIG_NAME_COMPETITION)
        upload_request = mock_kaggle.competitions.competition_api_client.start_submission_upload.call_args[0][0]
        self.assertEqual(upload_request.competition_name, "config-comp")

    # competition_submit_code tests
    def test_submit_code_missing_competition_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_code("file.csv", "message", None, "owner/notebook")
        self.assertIn("No competition specified", str(context.exception))

    def test_submit_code_missing_kernel_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_code("file.csv", "message", "comp", None)
        self.assertIn("No kernel specified", str(context.exception))

    def test_submit_code_invalid_kernel_format_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_code("file.csv", "message", "comp", "notebook-only")
        self.assertIn("The kernel must be specified as <owner>/<notebook>", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_submit_code_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = ApiCreateCodeSubmissionResponse()
        mock_kaggle.competitions.competition_api_client.create_code_submission.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        response = self.api.competition_submit_code(
            "output.csv", "code message", "comp-name", "owner/notebook", kernel_version=5
        )

        self.assertEqual(response, mock_response)
        mock_kaggle.competitions.competition_api_client.create_code_submission.assert_called_once()
        request = mock_kaggle.competitions.competition_api_client.create_code_submission.call_args[0][0]
        self.assertEqual(request.file_name, "output.csv")
        self.assertEqual(request.competition_name, "comp-name")
        self.assertEqual(request._kernel_owner, "owner")
        self.assertEqual(request.kernel_slug, "notebook")
        self.assertEqual(request.kernel_version, 5)
        self.assertEqual(request.submission_description, "code message")

    @patch.dict(os.environ, {"KAGGLE_COMPETITION_SUBMISSION_MODEL_VERSION_ID": "456"})
    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_submit_admin_model_succeeds(self, mock_client, mock_upload_complete):
        mock_kaggle = MagicMock()
        mock_upload_response = MagicMock()
        mock_upload_response.create_url = "http://upload-url"
        mock_upload_response.token = "token-123"
        mock_kaggle.competitions.competition_api_client.start_submission_upload.return_value = mock_upload_response
        mock_kaggle.competitions.competition_api_client.create_submission.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        dummy_file = os.path.join(self.temp_dir, "submission.csv")
        open(dummy_file, "w").close()

        self.api.competition_submit(dummy_file, "message", "comp-name", quiet=True)

        submit_request = mock_kaggle.competitions.competition_api_client.create_submission.call_args[0][0]
        self.assertEqual(submit_request.benchmark_model_version_id, 456)

    @patch.object(KaggleApi, "get_config_value", return_value="config-comp")
    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_submit_fallback_config_verbose_succeeds(self, mock_client, mock_upload_complete, mock_get_config):
        mock_kaggle = MagicMock()
        mock_upload_response = MagicMock()
        mock_upload_response.token = "token-config"
        mock_kaggle.competitions.competition_api_client.start_submission_upload.return_value = mock_upload_response
        mock_kaggle.competitions.competition_api_client.create_submission.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        dummy_file = os.path.join(self.temp_dir, "submission.csv")
        open(dummy_file, "w").close()

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.api.competition_submit(dummy_file, "message", None, quiet=False)

        self.assertIn("Using competition: config-comp", f.getvalue())

    @patch.object(KaggleApi, "get_config_value", return_value="config-comp")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_submit_code_fallback_config_verbose_succeeds(self, mock_client, mock_get_config):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.create_code_submission.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.api.competition_submit_code("output.csv", "message", None, "owner/notebook", quiet=False)

        self.assertIn("Using competition: config-comp", f.getvalue())

    def test_competition_submit_code_requires_file(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_code(
                None,
                "message",
                "comp",
                "owner/notebook",
                1,
            )

        self.assertIn("No file specified", str(context.exception))


class TestCompetitionSubmitCli(unittest.TestCase):
    """Tests for competition_submit_cli argument validation."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True

    def test_cli_code_submission_requires_kernel_and_version_together(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_cli(
                file_name="output.csv",
                message="message",
                competition="comp-name",
                version="1",
            )
        self.assertIn(
            "Code competition submissions require both the kernel name and the version number",
            str(context.exception),
        )

        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_cli(
                file_name="output.csv",
                message="message",
                competition="comp-name",
                kernel="owner/notebook",
            )
        self.assertIn(
            "Code competition submissions require both the kernel name and the version number",
            str(context.exception),
        )

    def test_cli_code_submission_requires_file(self):
        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_cli(
                file_name=None,
                message="message",
                competition="comp-name",
                kernel="owner/notebook",
                version="1",
            )
        self.assertIn("No file specified", str(context.exception))


def _http_error(status_code, url="https://api.kaggle.com/v1/competitions.CompetitionApiService/StartSubmissionUpload"):
    """Returns the HTTPError that a real requests.Response raises for `status_code`.

    A MagicMock response would hide the bug under test: a mock is always truthy, while a
    real Response is falsy for every 4xx/5xx status.
    """
    response = requests.Response()
    response.status_code = status_code
    response.reason = "Not Found" if status_code == 404 else "Server Error"
    response.url = url
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return e
    raise AssertionError(f"{status_code} is not an error status")


class TestCompetitionSubmitCli404(unittest.TestCase):
    """A 404 from the submit calls must fail the CLI with a hint that names what may be missing.

    Regression: the check was `if e.response and ...`, which never matched a 404, and its
    branch printed to stdout and returned, which would have exited 0.
    """

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True

    @patch.object(KaggleApi, "competition_submit")
    def test_cli_404_on_file_submission_raises_competition_hint(self, mock_submit):
        error = _http_error(404)
        self.assertFalse(error.response)  # the falsy Response the old check tripped over
        mock_submit.side_effect = error

        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_cli(file_name="sub.csv", message="msg", competition="no-such-comp")

        message = str(context.exception)
        self.assertIn("Could not find competition - please verify", message)
        self.assertNotIn("notebook", message)
        self.assertIn("404 Client Error", message)  # the server's own error is kept
        self.assertIs(context.exception.__cause__, error)

    @patch.object(KaggleApi, "competition_submit_code")
    def test_cli_404_on_code_submission_names_notebook_and_version(self, mock_submit_code):
        mock_submit_code.side_effect = _http_error(
            404, url="https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission"
        )

        with self.assertRaises(ValueError) as context:
            self.api.competition_submit_cli(
                file_name="submission.csv", message="msg", competition="comp", kernel="owner/notebook", version="3"
            )

        message = str(context.exception)
        self.assertIn("Could not find competition or notebook", message)
        self.assertIn("notebook and version", message)
        self.assertIn("CreateCodeSubmission", message)

    @patch.object(KaggleApi, "competition_submit")
    def test_cli_non_404_http_error_propagates_unchanged(self, mock_submit):
        error = _http_error(500)
        mock_submit.side_effect = error

        with self.assertRaises(requests.exceptions.HTTPError) as context:
            self.api.competition_submit_cli(file_name="sub.csv", message="msg", competition="comp")

        self.assertIs(context.exception, error)

    @patch.object(KaggleApi, "competition_submit")
    def test_cli_request_error_without_response_propagates_unchanged(self, mock_submit):
        error = requests.exceptions.ConnectionError("connection refused")
        mock_submit.side_effect = error

        with self.assertRaises(requests.exceptions.ConnectionError) as context:
            self.api.competition_submit_cli(file_name="sub.csv", message="msg", competition="comp")

        self.assertIs(context.exception, error)

    def test_main_exits_non_zero_and_reports_404_on_stderr(self):
        # End to end: only the HTTP transport is faked, so the real SDK raises the 404 from
        # the JSON error body Kaggle sends, and the real main() turns it into the exit code.
        import kaggle.cli

        def send(session, request, **kwargs):
            response = requests.Response()
            response.status_code = 404
            response.headers["Content-Type"] = "application/json"
            response._content = json.dumps({"code": 404, "message": "Competition not found"}).encode()
            response.url = request.url
            response.request = request
            return response

        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        submission = os.path.join(temp_dir, "submission.csv")
        with open(submission, "w") as f:
            f.write("id,target\n1,0\n")

        # A real API object of our own, so the test does not depend on whatever an earlier
        # test may have left in kaggle.cli.api.
        cli_api = KaggleApi()
        cli_api.config_values = {"username": "testuser", "key": "testkey"}
        cli_api._authenticated = True

        argv = ["kaggle", "competitions", "submit", "no-such-comp", "-f", submission, "-m", "msg"]
        out, err = io.StringIO(), io.StringIO()
        exit_code = 0
        with (
            patch.object(sys, "argv", argv),
            patch.object(kaggle.cli, "api", cli_api),
            patch.object(requests.Session, "send", send),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            try:
                kaggle.cli.main()
            except SystemExit as e:
                exit_code = e.code

        self.assertEqual(exit_code, 1, f"stdout={out.getvalue()!r} stderr={err.getvalue()!r}")
        self.assertIn("Could not find competition - please verify", err.getvalue())
        self.assertIn("Competition not found", err.getvalue())
        self.assertNotIn("Could not find competition", out.getvalue())


class TestCompetitionSubmitCliLimits(unittest.TestCase):
    """After a successful submit, the CLI wrapper should print the remaining
    submission allowance for the day. Limits errors must not fail the submit."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True

    @patch.object(KaggleApi, "competition_get_submission_limits")
    @patch.object(KaggleApi, "competition_submit")
    def test_cli_prints_remaining_after_success(self, mock_submit, mock_limits):
        resp = ApiCreateSubmissionResponse()
        resp.message = "submitted"
        mock_submit.return_value = resp
        mock_limits.return_value = MagicMock(num_allowed_now=4)

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = self.api.competition_submit_cli(
                file_name="sub.csv", message="msg", competition="my-comp", quiet=False
            )

        self.assertEqual(result, "submitted")
        mock_limits.assert_called_once_with("my-comp")
        self.assertIn("4 submissions remaining today.", buf.getvalue())

    @patch.object(KaggleApi, "competition_get_submission_limits")
    @patch.object(KaggleApi, "competition_submit")
    def test_cli_swallows_limits_failure(self, mock_submit, mock_limits):
        resp = ApiCreateSubmissionResponse()
        resp.message = "submitted"
        mock_submit.return_value = resp
        mock_limits.side_effect = RuntimeError("limits endpoint blip")

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = self.api.competition_submit_cli(
                file_name="sub.csv", message="msg", competition="my-comp", quiet=False
            )

        self.assertEqual(result, "submitted")
        self.assertNotIn("remaining today", buf.getvalue())

    @patch.object(KaggleApi, "competition_get_submission_limits")
    @patch.object(KaggleApi, "competition_submit")
    def test_cli_skips_limits_when_quiet(self, mock_submit, mock_limits):
        resp = ApiCreateSubmissionResponse()
        resp.message = "submitted"
        mock_submit.return_value = resp

        self.api.competition_submit_cli(file_name="sub.csv", message="msg", competition="my-comp", quiet=True)

        mock_limits.assert_not_called()

    @patch.object(KaggleApi, "competition_get_submission_limits")
    @patch.object(KaggleApi, "competition_submit")
    def test_cli_raises_on_upload_failure(self, mock_submit, mock_limits):
        # competition_submit returns a synthetic response with this sentinel
        # message when the file upload fails (no exception raised). The CLI
        # must raise so `kaggle competitions submit` exits non-zero, and must
        # NOT print the limits line, which would imply the submission counted.
        resp = ApiCreateSubmissionResponse()
        resp.message = KaggleApi.COMPETITION_SUBMIT_UPLOAD_FAILED_MESSAGE
        mock_submit.return_value = resp

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(ValueError) as context:
                self.api.competition_submit_cli(file_name="sub.csv", message="msg", competition="my-comp", quiet=False)

        self.assertIn(KaggleApi.COMPETITION_SUBMIT_UPLOAD_FAILED_MESSAGE, str(context.exception))
        mock_limits.assert_not_called()
        self.assertNotIn("remaining today", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
