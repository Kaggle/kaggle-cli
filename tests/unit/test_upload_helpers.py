# coding=utf-8
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from typing import Any
import sys
import requests
from urllib3.util import Retry
import json
import time

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.models.kaggle_models_extended import ResumableUploadResult
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadResponse
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetVersionRequestBody, ApiDatasetColumn
from kaggle.models.upload_file import UploadFile


class TestUploadHelpers(unittest.TestCase):
    """Tests for upload helpers: _resume_upload, _get_bytes_already_uploaded, upload_complete."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_dummy_file(self, size):
        path = os.path.join(self.temp_dir, "dummy.bin")
        with open(path, "wb") as f:
            f.write(b"\0" * size)
        return path

    def _mock_blob_client(self, mock_client):
        """Wires a build_kaggle_client mock whose start_blob_upload returns a fresh
        url/token per call, so tests can count how often a session was initiated."""
        mock_kaggle = MagicMock()
        started = []

        def start_blob_upload(request):
            started.append(request)
            response = ApiStartBlobUploadResponse()
            response.create_url = "http://upload-url/%d" % len(started)
            response.token = "token-%d" % len(started)
            return response

        mock_kaggle.blobs.blob_api_client.start_blob_upload.side_effect = start_blob_upload
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle.blobs.blob_api_client.start_blob_upload

    # _get_bytes_already_uploaded tests
    def test_get_bytes_no_header_returns_minus_one(self):
        response = MagicMock()
        response.headers = {}
        bytes_uploaded = self.api._get_bytes_already_uploaded(response, quiet=True)
        self.assertEqual(bytes_uploaded, -1)

    def test_get_bytes_valid_header_returns_bytes(self):
        response = MagicMock()
        response.headers = {"Range": "bytes=0-1000"}
        bytes_uploaded = self.api._get_bytes_already_uploaded(response, quiet=True)
        self.assertEqual(bytes_uploaded, 1000)

    def test_get_bytes_valid_header_no_bytes_prefix_returns_bytes(self):
        response = MagicMock()
        response.headers = {"Range": "0-2000"}
        bytes_uploaded = self.api._get_bytes_already_uploaded(response, quiet=True)
        self.assertEqual(bytes_uploaded, 2000)

    def test_get_bytes_invalid_header_format_returns_zero(self):
        response = MagicMock()
        response.headers = {"Range": "invalid"}
        bytes_uploaded = self.api._get_bytes_already_uploaded(response, quiet=True)
        self.assertIsNone(bytes_uploaded)

    def test_get_bytes_invalid_int_returns_zero(self):
        response = MagicMock()
        response.headers = {"Range": "bytes=0-abc"}
        bytes_uploaded = self.api._get_bytes_already_uploaded(response, quiet=True)
        self.assertIsNone(bytes_uploaded)

    # _resume_upload tests
    @patch("requests.Session")
    def test_resume_upload_success_200_returns_complete(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.COMPLETE)
        mock_session.headers.update.assert_called_once_with({"Content-Length": "0", "Content-Range": "bytes */1000"})

    @patch("requests.Session")
    def test_resume_upload_success_201_returns_complete(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.COMPLETE)

    @patch("requests.Session")
    def test_resume_upload_expired_404_returns_expired(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        # An expired session cannot be resumed but the file was never rejected, so the
        # upload has to be restarted from scratch rather than abandoned.
        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.EXPIRED)

    @patch.object(KaggleApi, "_get_bytes_already_uploaded", return_value=500)
    @patch("requests.Session")
    def test_resume_upload_incomplete_308_returns_incomplete(self, mock_session_cls, mock_get_bytes):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 308
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.INCOMPLETE)
        self.assertEqual(res.bytes_uploaded, 500)
        self.assertEqual(res.start_at, 501)
        mock_get_bytes.assert_called_once_with(mock_response, True)

    @patch.object(KaggleApi, "_get_bytes_already_uploaded", return_value=None)
    @patch("requests.Session")
    def test_resume_upload_incomplete_308_error_returns_expired(self, mock_session_cls, mock_get_bytes):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 308
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        # A broken Range header means the offset is unknown, so start from scratch.
        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.EXPIRED)

    @patch("requests.Session")
    def test_resume_upload_other_error_returns_failed(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.FAILED)

    # upload_complete tests
    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_success_returns_complete(self, mock_tqdm, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        path = self._create_dummy_file(100)
        res = self.api.upload_complete(path, "http://url", quiet=True)
        self.assertEqual(res, ResumableUploadResult.COMPLETE)
        mock_session.put.assert_called_once()

    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_503_returns_incomplete(self, mock_tqdm, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        path = self._create_dummy_file(100)
        res = self.api.upload_complete(path, "http://url", quiet=True)
        self.assertEqual(res, ResumableUploadResult.INCOMPLETE)

    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_500_returns_failed(self, mock_tqdm, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        path = self._create_dummy_file(100)
        res = self.api.upload_complete(path, "http://url", quiet=True)
        self.assertEqual(res, ResumableUploadResult.FAILED)

    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_exception_returns_failed(self, mock_tqdm, mock_session_cls):
        mock_session = MagicMock()
        mock_session.put.side_effect = Exception("network error")
        mock_session_cls.return_value = mock_session

        path = self._create_dummy_file(100)
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            res = self.api.upload_complete(path, "http://url", quiet=True)
        self.assertEqual(res, ResumableUploadResult.INCOMPLETE)
        self.assertIn("network error", f.getvalue())

    @patch.object(KaggleApi, "_resume_upload")
    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_resume_complete_returns_complete(self, mock_tqdm, mock_session_cls, mock_resume):
        mock_resume.return_value = ResumableUploadResult.Complete()

        path = self._create_dummy_file(100)
        res = self.api.upload_complete(path, "http://url", quiet=True, resume=True)
        self.assertEqual(res, ResumableUploadResult.COMPLETE)
        mock_resume.assert_called_once_with(path, "http://url", 100, True)
        mock_session_cls.assert_not_called()

    @patch.object(KaggleApi, "_resume_upload")
    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_resume_failed_returns_failed(self, mock_tqdm, mock_session_cls, mock_resume):
        mock_resume.return_value = ResumableUploadResult.Failed()

        path = self._create_dummy_file(100)
        res = self.api.upload_complete(path, "http://url", quiet=True, resume=True)
        self.assertEqual(res, ResumableUploadResult.FAILED)
        mock_resume.assert_called_once()
        mock_session_cls.assert_not_called()

    @patch.object(KaggleApi, "_resume_upload")
    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_resume_expired_returns_expired(self, mock_tqdm, mock_session_cls, mock_resume):
        mock_resume.return_value = ResumableUploadResult.Expired()

        path = self._create_dummy_file(100)
        res = self.api.upload_complete(path, "http://url", quiet=True, resume=True)
        self.assertEqual(res, ResumableUploadResult.EXPIRED)
        mock_resume.assert_called_once()
        mock_session_cls.assert_not_called()

    @patch.object(KaggleApi, "_resume_upload")
    @patch("requests.Session")
    @patch("kaggle.api.kaggle_api_extended.tqdm")
    def test_upload_complete_resume_incomplete_resumes_and_succeeds(self, mock_tqdm, mock_session_cls, mock_resume):
        mock_resume.return_value = ResumableUploadResult.Incomplete(40)

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        path = self._create_dummy_file(100)
        res = self.api.upload_complete(path, "http://url", quiet=True, resume=True)
        self.assertEqual(res, ResumableUploadResult.COMPLETE)
        mock_resume.assert_called_once()

        mock_session.headers.update.assert_any_call({"Content-Length": "59", "Content-Range": "bytes 41-99/100"})

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_valid_file_succeeds(self, mock_client, mock_upload_complete):
        mock_kaggle = MagicMock()
        mock_response = ApiStartBlobUploadResponse()
        mock_response.create_url = "http://upload-url"
        mock_response.token = "token-123"
        mock_kaggle.blobs.blob_api_client.start_blob_upload.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        path = self._create_dummy_file(100)

        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        with patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir", return_value=self.temp_dir):
            context = ResumableUploadContext(no_resume=False)
            with context:
                token = self.api._upload_blob(path, quiet=True, blob_type=ApiBlobType.INBOX, upload_context=context)

        self.assertEqual(token, "token-123")
        mock_kaggle.blobs.blob_api_client.start_blob_upload.assert_called_once()
        mock_upload_complete.assert_called_once_with(path, "http://upload-url", True, resume=False)

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_no_resume_raises_attribute_error(self, mock_client, mock_upload_complete):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.create_url = "http://upload-url"
        mock_response.token = "token-123"
        mock_kaggle.blobs.blob_api_client.start_blob_upload.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        path = self._create_dummy_file(100)

        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        context = ResumableUploadContext(no_resume=True)

        with self.assertRaises(AttributeError) as cm:
            self.api._upload_blob(path, quiet=True, blob_type=ApiBlobType.INBOX, upload_context=context)

        self.assertIn("'NoneType' object has no attribute", str(cm.exception))

    # _upload_file tests
    @patch.object(KaggleApi, "_upload_blob", return_value="token-123")
    def test_upload_file_no_resources_succeeds(self, mock_upload_blob):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        context = ResumableUploadContext(no_resume=True)
        path = self._create_dummy_file(100)

        result = self.api._upload_file("dummy.bin", path, ApiBlobType.INBOX, context, quiet=True, resources=None)

        assert result is not None
        self.assertEqual(result.token, "token-123")
        self.assertIsNone(result.description)
        mock_upload_blob.assert_called_once_with(path, True, ApiBlobType.INBOX, context, None)

    @patch.object(KaggleApi, "_upload_blob", return_value=None)
    def test_upload_file_fails(self, mock_upload_blob):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        context = ResumableUploadContext(no_resume=True)
        path = self._create_dummy_file(100)

        # A failed upload must abort rather than be reported back as a skipped file,
        # which would leave it out of the dataset/model version.
        with self.assertRaises(ValueError) as cm:
            self.api._upload_file("dummy.bin", path, ApiBlobType.INBOX, context, quiet=True, resources=None)

        self.assertIn("Upload unsuccessful: dummy.bin", str(cm.exception))

    @patch.object(KaggleApi, "_upload_blob", return_value="token-123")
    def test_upload_file_with_resources_succeeds(self, mock_upload_blob):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        context = ResumableUploadContext(no_resume=True)
        path = self._create_dummy_file(100)

        resources: list[dict[str, Any]] = [
            {
                "path": "dummy.bin",
                "description": "my description",
                "schema": {
                    "fields": [
                        {"name": "col1", "type": "string", "description": "col1 desc"},
                        {"name": "col2", "type": "numeric"},
                        {"name": "col3", "type": "boolean"},
                        {"name": "col4", "type": "datetime"},
                        {"name": "col5", "type": "unknown-type"},
                    ]
                },
            }
        ]

        result = self.api._upload_file("dummy.bin", path, ApiBlobType.INBOX, context, quiet=True, resources=resources)

        assert result is not None
        self.assertEqual(result.token, "token-123")
        self.assertEqual(result.description, "my description")
        self.assertEqual(len(result.columns), 5)
        self.assertEqual(result.columns[0].name, "col1")
        self.assertEqual(result.columns[0].type, "string")
        self.assertEqual(result.columns[0].description, "col1 desc")
        self.assertEqual(result.columns[0].order, 0)
        self.assertEqual(result.columns[1].name, "col2")
        self.assertEqual(result.columns[1].type, "numeric")
        self.assertEqual(result.columns[1].order, 1)
        self.assertEqual(result.columns[2].name, "col3")
        self.assertEqual(result.columns[2].type, "boolean")
        self.assertEqual(result.columns[2].order, 2)
        self.assertEqual(result.columns[3].name, "col4")
        self.assertEqual(result.columns[3].type, "datetime")
        self.assertEqual(result.columns[3].order, 3)
        self.assertEqual(result.columns[4].name, "col5")
        self.assertEqual(result.columns[4].type, "unknown-type")
        self.assertEqual(result.columns[4].order, 4)

    @patch.object(KaggleApi, "_upload_blob", return_value="token-123")
    def test_upload_file_verbose_succeeds(self, mock_upload_blob):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        context = ResumableUploadContext(no_resume=True)
        path = self._create_dummy_file(100)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.api._upload_file("dummy.bin", path, ApiBlobType.INBOX, context, quiet=False, resources=None)

        self.assertIsNotNone(result)
        self.assertIn("Starting upload for file dummy.bin", f.getvalue())
        self.assertIn("Upload successful: dummy.bin", f.getvalue())

    @patch.object(KaggleApi, "_upload_blob", return_value=None)
    def test_upload_file_failure_verbose_fails(self, mock_upload_blob):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        context = ResumableUploadContext(no_resume=True)
        path = self._create_dummy_file(100)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f), self.assertRaises(ValueError) as cm:
            self.api._upload_file("dummy.bin", path, ApiBlobType.INBOX, context, quiet=False, resources=None)

        self.assertIn("Upload unsuccessful: dummy.bin", str(cm.exception))
        self.assertIn("Starting upload for file dummy.bin", f.getvalue())

    # _upload_blob additional tests
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_already_complete_returns_token(self, mock_client):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        context = MagicMock(spec=ResumableUploadContext)
        mock_file_upload = MagicMock()
        mock_file_upload.upload_complete = True
        mock_file_upload.get_token.return_value = "token-already-done"
        context.new_resumable_file_upload.return_value = mock_file_upload

        path = self._create_dummy_file(100)
        result = self.api._upload_blob(path, quiet=True, blob_type=ApiBlobType.INBOX, upload_context=context)

        self.assertEqual(result, "token-already-done")
        mock_client.assert_not_called()

    @patch.object(KaggleApi, "upload_complete")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_retry_loop_calls_start_upload_once_succeeds(self, mock_client, mock_upload_complete):
        mock_kaggle = MagicMock()
        mock_response = ApiStartBlobUploadResponse()
        mock_response.create_url = "http://upload-url"
        mock_response.token = "token-123"
        mock_kaggle.blobs.blob_api_client.start_blob_upload.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        mock_upload_complete.side_effect = [ResumableUploadResult.INCOMPLETE, ResumableUploadResult.COMPLETE]

        path = self._create_dummy_file(100)

        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        with patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir", return_value=self.temp_dir):
            context = ResumableUploadContext(no_resume=False)
            with context:
                token = self.api._upload_blob(path, quiet=True, blob_type=ApiBlobType.INBOX, upload_context=context)

        self.assertEqual(token, "token-123")
        # Bug fixed: start_blob_upload should only be called once
        self.assertEqual(mock_kaggle.blobs.blob_api_client.start_blob_upload.call_count, 1)

    def _upload_blob_in_context(self, path, quiet=True):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        with patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir", return_value=self.temp_dir):
            context = ResumableUploadContext(no_resume=False)
            with context:
                return self.api._upload_blob(path, quiet, ApiBlobType.INBOX, context)

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.FAILED)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_failed_does_not_restart_returns_none(self, mock_client, mock_upload_complete):
        start_blob_upload = self._mock_blob_client(mock_client)
        path = self._create_dummy_file(100)

        token = self._upload_blob_in_context(path)

        # FAILED means the server rejected the file, so retrying would only hit the
        # same error: give up after a single attempt and report no token.
        self.assertIsNone(token)
        self.assertEqual(mock_upload_complete.call_count, 1)
        self.assertEqual(start_blob_upload.call_count, 1)

    @patch.object(KaggleApi, "upload_complete")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_expired_restarts_upload_succeeds(self, mock_client, mock_upload_complete):
        start_blob_upload = self._mock_blob_client(mock_client)
        mock_upload_complete.side_effect = [ResumableUploadResult.EXPIRED, ResumableUploadResult.COMPLETE]
        path = self._create_dummy_file(100)

        token = self._upload_blob_in_context(path)

        # The expired session is dropped and a brand new one is initiated, so the
        # token must come from the second start_blob_upload call.
        self.assertEqual(token, "token-2")
        self.assertEqual(start_blob_upload.call_count, 2)
        # The restarted attempt uploads from scratch instead of resuming.
        self.assertEqual(mock_upload_complete.call_args_list[1][1]["resume"], False)
        self.assertEqual(mock_upload_complete.call_args_list[1][0][1], "http://upload-url/2")

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.EXPIRED)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_expired_restart_respects_max_attempts_returns_none(self, mock_client, mock_upload_complete):
        start_blob_upload = self._mock_blob_client(mock_client)
        path = self._create_dummy_file(100)

        token = self._upload_blob_in_context(path)

        # Restarting is bounded by MAX_UPLOAD_RESUME_ATTEMPTS, i.e., it cannot loop forever.
        self.assertIsNone(token)
        self.assertEqual(mock_upload_complete.call_count, KaggleApi.MAX_UPLOAD_RESUME_ATTEMPTS)
        self.assertEqual(start_blob_upload.call_count, KaggleApi.MAX_UPLOAD_RESUME_ATTEMPTS)

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.INCOMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_blob_incomplete_exhausts_attempts_returns_none(self, mock_client, mock_upload_complete):
        start_blob_upload = self._mock_blob_client(mock_client)
        path = self._create_dummy_file(100)

        token = self._upload_blob_in_context(path)

        # Resuming keeps the same session, and running out of attempts yields no token.
        self.assertIsNone(token)
        self.assertEqual(mock_upload_complete.call_count, KaggleApi.MAX_UPLOAD_RESUME_ATTEMPTS)
        self.assertEqual(start_blob_upload.call_count, 1)

    # upload_files tests
    def _create_upload_folder(self, *file_names):
        folder = os.path.join(self.temp_dir, "folder")
        os.makedirs(folder, exist_ok=True)
        for file_name in file_names:
            with open(os.path.join(folder, file_name), "w") as f:
                f.write("data")
        return folder

    @patch.object(KaggleApi, "upload_complete")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_files_failure_aborts_and_keeps_no_partial_files(self, mock_client, mock_upload_complete):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        self._mock_blob_client(mock_client)

        def upload_complete(path, url, quiet, resume=False):
            if os.path.basename(path) == "two.csv":
                return ResumableUploadResult.FAILED
            return ResumableUploadResult.COMPLETE

        mock_upload_complete.side_effect = upload_complete

        folder = self._create_upload_folder("one.csv", "two.csv", "three.csv")
        request = ApiCreateDatasetVersionRequestBody()
        request.files = []

        with patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir", return_value=self.temp_dir):
            context = ResumableUploadContext(no_resume=False)
            with self.assertRaises(ValueError) as cm:
                with context:
                    self.api.upload_files(request, None, folder, ApiBlobType.DATASET, context, quiet=True)

        # The failed file must not be silently skipped: the whole operation aborts so
        # that no version is created without it.
        self.assertIn("Upload unsuccessful: two.csv", str(cm.exception))
        self.assertLess(len(request.files), 3)

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_files_all_succeed_registers_every_file(self, mock_client, mock_upload_complete):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        self._mock_blob_client(mock_client)

        folder = self._create_upload_folder("one.csv", "two.csv", "three.csv")
        request = ApiCreateDatasetVersionRequestBody()
        request.files = []

        with patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir", return_value=self.temp_dir):
            context = ResumableUploadContext(no_resume=False)
            with context:
                self.api.upload_files(request, None, folder, ApiBlobType.DATASET, context, quiet=True)

        self.assertEqual(len(request.files), 3)
        self.assertEqual(sorted(f.token for f in request.files), ["token-1", "token-2", "token-3"])

    @patch.object(KaggleApi, "upload_complete", return_value=ResumableUploadResult.COMPLETE)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_files_skips_folder_without_dir_mode_succeeds(self, mock_client, mock_upload_complete):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext

        self._mock_blob_client(mock_client)

        folder = self._create_upload_folder("one.csv")
        os.makedirs(os.path.join(folder, "subfolder"), exist_ok=True)
        request = ApiCreateDatasetVersionRequestBody()
        request.files = []

        with patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir", return_value=self.temp_dir):
            context = ResumableUploadContext(no_resume=False)
            with context:
                self.api.upload_files(request, None, folder, ApiBlobType.DATASET, context, quiet=True)

        # An intentionally skipped folder is not an upload failure and must not abort.
        self.assertEqual(len(request.files), 1)
        self.assertEqual(request.files[0].token, "token-1")

    # Verbose/print tests
    @patch("requests.Session")
    def test_resume_upload_expired_verbose_returns_expired(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.api._resume_upload("path/to/file", "url", 1000, quiet=False)
        self.assertIn("Upload of path/to/file expired", f.getvalue())

    @patch.object(KaggleApi, "_get_bytes_already_uploaded", return_value=500)
    @patch("requests.Session")
    def test_resume_upload_incomplete_verbose_returns_incomplete(self, mock_session_cls, mock_get_bytes):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 308
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.api._resume_upload("path", "url", 1000, quiet=False)
        self.assertIn("Already uploaded 500 bytes. Will resume upload at 501", f.getvalue())

    @patch("requests.Session")
    def test_resume_upload_other_error_verbose_returns_failed(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.api._resume_upload("path", "url", 1000, quiet=False)
        self.assertIn("Server returned 500", f.getvalue())

    def test_get_bytes_invalid_header_format_verbose_returns_zero(self):
        response = MagicMock()
        response.headers = {"Range": "invalid"}
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.api._get_bytes_already_uploaded(response, quiet=False)
        self.assertIn("Invalid Range header format", f.getvalue())

    def test_get_bytes_invalid_int_verbose_returns_zero(self):
        response = MagicMock()
        response.headers = {"Range": "bytes=0-abc"}
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.api._get_bytes_already_uploaded(response, quiet=False)
        self.assertIn("Invalid Range header format", f.getvalue())

    def test_resumable_upload_result_no_range_offset(self):
        result = ResumableUploadResult.Incomplete(-1)
        self.assertEqual(result.bytes_uploaded, 0)
        self.assertEqual(result.start_at, 0)

    def test_resumable_upload_result_range_zero_offset(self):
        result = ResumableUploadResult.Incomplete(0)
        self.assertEqual(result.bytes_uploaded, 0)
        self.assertEqual(result.start_at, 1)

    def test_resumable_upload_result_range_mid_offset(self):
        result = ResumableUploadResult.Incomplete(499)
        self.assertEqual(result.bytes_uploaded, 499)
        self.assertEqual(result.start_at, 500)

    @patch("requests.Session")
    def test_resume_upload_no_range_header(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 308
        mock_response.headers = {}
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.INCOMPLETE)
        self.assertEqual(res.bytes_uploaded, 0)
        self.assertEqual(res.start_at, 0)

    @patch("requests.Session")
    def test_resume_upload_range_zero(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 308
        mock_response.headers = {"Range": "bytes=0-0"}
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.INCOMPLETE)
        self.assertEqual(res.bytes_uploaded, 0)
        self.assertEqual(res.start_at, 1)

    @patch("requests.Session")
    def test_resume_upload_range_mid(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 308
        mock_response.headers = {"Range": "bytes=0-499"}
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        res = self.api._resume_upload("path", "url", 1000, quiet=True)
        self.assertEqual(res.result, ResumableUploadResult.INCOMPLETE)
        self.assertEqual(res.bytes_uploaded, 499)
        self.assertEqual(res.start_at, 500)


class TestResumableFileUpload(unittest.TestCase):
    """Tests for ResumableFileUpload class details."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest

        self.req = ApiStartBlobUploadRequest()
        self.req.name = "test.bin"
        self.req.type = ApiBlobType.INBOX

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_token_not_complete_returns_none(self):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext, ResumableFileUpload

        context = ResumableUploadContext(no_resume=True)
        file_upload = ResumableFileUpload("path", self.req, context)
        self.assertIsNone(file_upload.get_token())

    @patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir")
    def test_load_previous_invalid_json_returns_false(self, mock_tempdir):
        mock_tempdir.return_value = self.temp_dir
        from kaggle.api.kaggle_api_extended import ResumableUploadContext, ResumableFileUpload

        context = ResumableUploadContext(no_resume=False)
        with context:
            file_upload = ResumableFileUpload("path/to/file.bin", self.req, context)

            info_path = context.get_upload_info_file_path("path/to/file.bin")
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, "w") as f:
                f.write("{invalid json")

            import io
            from contextlib import redirect_stdout

            f_err = io.StringIO()
            with redirect_stdout(f_err):
                file_upload.load()

            self.assertFalse(file_upload.can_resume)
            self.assertIn("Error while trying to load upload info", f_err.getvalue())

    def test_upload_completed_no_resume_returns_early(self):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext, ResumableFileUpload

        context = ResumableUploadContext(no_resume=True)
        file_upload = ResumableFileUpload("path", self.req, context)
        file_upload.upload_completed()
        self.assertFalse(file_upload.upload_complete)

    def test_cleanup_no_resume_returns_early(self):
        from kaggle.api.kaggle_api_extended import ResumableUploadContext, ResumableFileUpload

        context = ResumableUploadContext(no_resume=True)
        file_upload = ResumableFileUpload("path", self.req, context)
        file_upload.cleanup()

    @patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir")
    def test_cleanup_file_not_found_ignores_error(self, mock_tempdir):
        mock_tempdir.return_value = self.temp_dir
        from kaggle.api.kaggle_api_extended import ResumableUploadContext, ResumableFileUpload

        context = ResumableUploadContext(no_resume=False)
        with context:
            file_upload = ResumableFileUpload("path", self.req, context)
            file_upload.cleanup()

    @patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir")
    def test_load_previous_expired_returns_false(self, mock_tempdir):
        mock_tempdir.return_value = self.temp_dir
        from kaggle.api.kaggle_api_extended import ResumableUploadContext, ResumableFileUpload

        context = ResumableUploadContext(no_resume=False)
        with context:
            file_upload = ResumableFileUpload("path.bin", self.req, context)
            expired_time = int(time.time() - (8 * 24 * 3600))

            previous = ResumableFileUpload("path.bin", self.req, context)
            previous.timestamp = expired_time
            previous.start_blob_upload_response = ApiStartBlobUploadResponse()
            previous.start_blob_upload_response.token = "old-token"
            previous.start_blob_upload_response.create_url = "http://old-url"

            info_path = context.get_upload_info_file_path("path.bin")
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, "w") as f:
                json.dump(previous.to_dict(), f)

            file_upload.load()
            self.assertFalse(file_upload.can_resume)

    @patch("kaggle.api.kaggle_api_extended.tempfile.gettempdir")
    def test_load_previous_valid_succeeds(self, mock_tempdir):
        mock_tempdir.return_value = self.temp_dir
        from kaggle.api.kaggle_api_extended import ResumableUploadContext, ResumableFileUpload

        context = ResumableUploadContext(no_resume=False)
        with context:
            file_upload = ResumableFileUpload("path.bin", self.req, context)
            fresh_time = int(time.time() - 100)

            previous = ResumableFileUpload("path.bin", self.req, context)
            previous.timestamp = fresh_time
            previous.start_blob_upload_response = ApiStartBlobUploadResponse()
            previous.start_blob_upload_response.token = "valid-token"
            previous.start_blob_upload_response.create_url = "http://valid-url"

            info_path = context.get_upload_info_file_path("path.bin")
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, "w") as f:
                json.dump(previous.to_dict(), f)

            file_upload.load()

            self.assertTrue(file_upload.can_resume)


if __name__ == "__main__":
    unittest.main()
