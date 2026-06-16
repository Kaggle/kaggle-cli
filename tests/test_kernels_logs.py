# coding=utf-8
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "..")

from kaggle.api.kaggle_api_extended import KaggleApi


def _sse_lines(events):
    """Encode a list of event payloads as SSE `data:` lines."""
    out = []
    for evt in events:
        if isinstance(evt, str):
            out.append(f"data: {evt}")
        else:
            out.append(f"data: {json.dumps(evt)}")
        out.append("")  # SSE event terminator
    return out


class TestKernelsLogs(unittest.TestCase):
    """Tests for the kernels_logs / kernels_logs_stream / kernels_logs_cli methods."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    @patch("kaggle.api.kaggle_api_extended.requests.get")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_output_file_pattern_searches_all_pages(self, mock_client, mock_get):
        """Test output download applies file_pattern across all paged results."""
        first_response = MagicMock()
        first_response.files = [MagicMock(file_name="first.txt", url="https://example.com/first.txt")]
        first_response.next_page_token = "page-2"
        first_response.log = None

        second_response = MagicMock()
        second_response.files = [MagicMock(file_name="result.png", url="https://example.com/result.png")]
        second_response.next_page_token = ""
        second_response.log = None

        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.side_effect = [
            first_response,
            second_response,
        ]
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = MagicMock(content=b"png")
        self.api.download_needed = MagicMock(return_value=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            outfiles, token = self.api.kernels_output(
                "owner/kernel-slug", temp_dir, file_pattern=r".*\.png$", quiet=True, page_size=1
            )

        self.assertEqual(token, "")
        self.assertEqual(len(outfiles), 1)
        self.assertTrue(outfiles[0].endswith("result.png"))
        mock_get.assert_called_once_with("https://example.com/result.png", stream=True)
        self.assertEqual(mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.call_count, 2)
        second_request = mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.call_args_list[1][0][0]
        self.assertEqual(second_request.page_token, "page-2")
        self.assertEqual(second_request.page_size, 1)

    @patch("kaggle.api.kaggle_api_extended.requests.get")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_output_page_token_downloads_specific_page(self, mock_client, mock_get):
        """Test output download uses a supplied page token for one page only."""
        response = MagicMock()
        response.files = [MagicMock(file_name="page-file.csv", url="https://example.com/page-file.csv")]
        response.next_page_token = "page-3"
        response.log = None

        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = MagicMock(content=b"csv")
        self.api.download_needed = MagicMock(return_value=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            outfiles, token = self.api.kernels_output(
                "owner/kernel-slug", temp_dir, quiet=True, page_token="page-2", page_size=50
            )

        self.assertEqual(token, "page-3")
        self.assertEqual(len(outfiles), 1)
        self.assertTrue(outfiles[0].endswith("page-file.csv"))
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.assert_called_once()
        request = mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.call_args[0][0]
        self.assertEqual(request.page_token, "page-2")
        self.assertEqual(request.page_size, 50)

    @patch.object(KaggleApi, "kernels_output")
    def test_kernels_output_cli_passes_page_size(self, mock_output):
        """Test CLI wrapper passes page_size to kernels_output."""
        mock_output.return_value = ([], "")

        self.api.kernels_output_cli("owner/kernel-slug", page_size=75)

        mock_output.assert_called_once_with(
            "owner/kernel-slug", None, None, False, False, page_token=None, page_size=75
        )

    # ------------------------------------------------------------------
    # kernels_logs (one-shot, persisted blob)
    # ------------------------------------------------------------------

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_returns_log_string(self, mock_validate, mock_client):
        mock_response = MagicMock()
        mock_response.log = "Line 1\nLine 2\nLine 3"
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.kernels_logs("owner/kernel-slug")
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_returns_empty_string_when_no_log(self, mock_validate, mock_client):
        mock_response = MagicMock()
        mock_response.log = None
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.kernels_logs("owner/kernel-slug")
        self.assertEqual(result, "")

    def test_kernels_logs_raises_when_kernel_none(self):
        with self.assertRaises(ValueError):
            self.api.kernels_logs(None)

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "get_config_value", return_value="defaultuser")
    def test_kernels_logs_uses_default_user_for_bare_slug(self, mock_config, mock_client):
        mock_response = MagicMock()
        mock_response.log = "some log"
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.kernels_logs("my-kernel")
        self.assertEqual(result, "some log")

        call_args = mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "defaultuser")
        self.assertEqual(request.kernel_slug, "my-kernel")

    # ------------------------------------------------------------------
    # kernels_logs_stream (live SSE)
    # ------------------------------------------------------------------

    def _make_streaming_kaggle_client(self, response_mock):
        http_session = MagicMock()
        http_session.headers = {"User-Agent": "test", "Content-Type": "application/json"}
        http_session.auth = None
        http_session.get.return_value = response_mock

        http_client = MagicMock()
        http_client._session = http_session
        http_client._endpoint = "http://localhost"
        # Force the non-PROD code path so we exercise the `/api` prefix.
        http_client._env = MagicMock(name="LOCAL")

        kaggle = MagicMock()
        kaggle._http_client = http_client

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=kaggle)
        cm.__exit__ = MagicMock(return_value=False)
        return cm, http_session

    @staticmethod
    def _sse_response(events):
        response = MagicMock()
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = iter(_sse_lines(events))
        response.raise_for_status = MagicMock()
        return response

    @staticmethod
    def _blob_response(lines, content_type="text/plain"):
        response = MagicMock()
        response.headers = {"Content-Type": content_type}
        response.iter_lines.return_value = iter(lines)
        response.raise_for_status = MagicMock()
        return response

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_stream_yields_events_and_stops_on_sentinel(self, _validate, mock_client):
        events = [
            {"stream_name": "stdout", "time": "t1", "data": "hello"},
            {"stream_name": "stderr", "time": "t2", "data": "warn"},
            "END_OF_LOG",
            # Anything after the sentinel must be ignored.
            {"stream_name": "stdout", "time": "t3", "data": "ignored"},
        ]
        response = self._sse_response(events)

        cm, http_session = self._make_streaming_kaggle_client(response)
        # Force PROD path off
        with patch("kaggle.api.kaggle_api_extended.KaggleEnv") as mock_env:
            mock_env.PROD = "PROD"
            cm.__enter__.return_value._http_client._env = "LOCAL"
            mock_client.return_value = cm

            result = list(self.api.kernels_logs_stream("owner/kernel-slug"))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["data"], "hello")
        self.assertEqual(result[1]["data"], "warn")

        # Verify URL and Accept header (advertise SSE but accept anything for blob fallback).
        call_args = http_session.get.call_args
        url = call_args[0][0]
        self.assertEqual(url, "http://localhost/api/v1/kernels/logs/stream/owner/kernel-slug")
        self.assertEqual(call_args.kwargs["headers"]["Accept"], "text/event-stream, */*")
        self.assertNotIn("Content-Type", call_args.kwargs["headers"])
        self.assertTrue(call_args.kwargs["stream"])
        response.close.assert_called_once()

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_stream_handles_non_json_payload(self, _validate, mock_client):
        response = MagicMock()
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = iter(["data: not-json", "", "data: END_OF_LOG", ""])
        response.raise_for_status = MagicMock()

        cm, _ = self._make_streaming_kaggle_client(response)
        with patch("kaggle.api.kaggle_api_extended.KaggleEnv") as mock_env:
            mock_env.PROD = "PROD"
            cm.__enter__.return_value._http_client._env = "LOCAL"
            mock_client.return_value = cm
            result = list(self.api.kernels_logs_stream("owner/kernel-slug"))

        self.assertEqual(result, [{"data": "not-json"}])

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_stream_falls_back_to_blob_for_completed_session(self, _validate, mock_client):
        # When the session is done the midtier returns the persisted GCS blob
        # with a non-SSE content type. We should yield one event per line.
        response = self._blob_response(["line one", "line two", "line three"])

        cm, _ = self._make_streaming_kaggle_client(response)
        with patch("kaggle.api.kaggle_api_extended.KaggleEnv") as mock_env:
            mock_env.PROD = "PROD"
            cm.__enter__.return_value._http_client._env = "LOCAL"
            mock_client.return_value = cm
            result = list(self.api.kernels_logs_stream("owner/kernel-slug"))

        self.assertEqual(
            result,
            [{"data": "line one"}, {"data": "line two"}, {"data": "line three"}],
        )
        response.close.assert_called_once()

    @patch.object(KaggleApi, "build_kaggle_client")
    @patch.object(KaggleApi, "validate_kernel_string")
    def test_kernels_logs_stream_blob_fallback_with_octet_stream(self, _validate, mock_client):
        # GCS blobs may come back as application/octet-stream; same handling.
        response = self._blob_response(["only-line"], content_type="application/octet-stream")

        cm, _ = self._make_streaming_kaggle_client(response)
        with patch("kaggle.api.kaggle_api_extended.KaggleEnv") as mock_env:
            mock_env.PROD = "PROD"
            cm.__enter__.return_value._http_client._env = "LOCAL"
            mock_client.return_value = cm
            result = list(self.api.kernels_logs_stream("owner/kernel-slug"))

        self.assertEqual(result, [{"data": "only-line"}])

    def test_kernels_logs_stream_raises_when_kernel_none(self):
        with self.assertRaises(ValueError):
            list(self.api.kernels_logs_stream(None))

    # ------------------------------------------------------------------
    # kernels_logs_cli
    # ------------------------------------------------------------------

    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_oneshot(self, mock_logs):
        mock_logs.return_value = "Line 1\nLine 2\nDone"
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug")
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue(), "Line 1\nLine 2\nDone\n")
        mock_logs.assert_called_once_with("owner/kernel-slug")

    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_uses_kernel_opt(self, mock_logs):
        mock_logs.return_value = "log output"
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli(None, kernel_opt="owner/kernel-slug")
        finally:
            sys.stdout = sys.__stdout__
        mock_logs.assert_called_once_with("owner/kernel-slug")

    @patch.object(KaggleApi, "kernels_logs_stream")
    def test_kernels_logs_cli_follow_streams_events(self, mock_stream):
        mock_stream.return_value = iter(
            [
                {"stream_name": "stdout", "time": "t1", "data": "hello"},
                {"stream_name": "stderr", "time": "t2", "data": "warn\n"},
                {"stream_name": "stdout", "time": "t3", "data": "bye"},
            ]
        )
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug", follow=True)
        finally:
            sys.stdout = sys.__stdout__

        # Each event's data is printed; lines without trailing newline get one.
        self.assertEqual(captured.getvalue(), "hello\nwarn\nbye\n")
        mock_stream.assert_called_once_with("owner/kernel-slug")

    @patch.object(KaggleApi, "kernels_logs_stream")
    def test_kernels_logs_cli_follow_skips_events_without_data(self, mock_stream):
        mock_stream.return_value = iter(
            [
                {"stream_name": "stdout", "time": "t1"},
                {"stream_name": "stdout", "time": "t2", "data": "only-line"},
            ]
        )
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug", follow=True)
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue(), "only-line\n")

    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_empty_log(self, mock_logs):
        mock_logs.return_value = ""
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.kernels_logs_cli("owner/kernel-slug")
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue(), "\n")

    @patch.object(KaggleApi, "kernels_logs_stream")
    @patch.object(KaggleApi, "kernels_logs")
    def test_kernels_logs_cli_interval_is_ignored(self, mock_logs, mock_stream):
        # `interval` is retained for backwards compatibility but no longer used.
        mock_stream.return_value = iter([])
        self.api.kernels_logs_cli("owner/kernel-slug", follow=True, interval=42)
        mock_logs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
