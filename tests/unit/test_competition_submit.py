# coding=utf-8
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, "../..")

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
    def test_cli_skips_limits_on_upload_failure(self, mock_submit, mock_limits):
        # competition_submit returns a synthetic response with this sentinel
        # message when the file upload fails (no exception raised). We must
        # NOT print the limits line in that case — it would imply the
        # submission counted when it didn't.
        resp = ApiCreateSubmissionResponse()
        resp.message = KaggleApi.COMPETITION_SUBMIT_UPLOAD_FAILED_MESSAGE
        mock_submit.return_value = resp

        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = self.api.competition_submit_cli(
                file_name="sub.csv", message="msg", competition="my-comp", quiet=False
            )

        self.assertEqual(result, KaggleApi.COMPETITION_SUBMIT_UPLOAD_FAILED_MESSAGE)
        mock_limits.assert_not_called()
        self.assertNotIn("remaining today", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
