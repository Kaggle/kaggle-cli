# coding=utf-8
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../../src")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestKernelVersionLabel(unittest.TestCase):
    """Tests for _kernel_version_label, which formats a version the way the API expects."""

    def test_number_is_prefixed(self):
        self.assertEqual(KaggleApi._kernel_version_label("2"), "v2")

    def test_multi_digit_number_is_prefixed(self):
        self.assertEqual(KaggleApi._kernel_version_label("12"), "v12")

    def test_value_already_prefixed_is_unchanged(self):
        self.assertEqual(KaggleApi._kernel_version_label("v2"), "v2")

    def test_value_that_is_not_a_number_is_unchanged(self):
        self.assertEqual(KaggleApi._kernel_version_label("1.2"), "1.2")


class _KernelRequestTestCase(unittest.TestCase):
    """Shared setup for the commands that accept <owner>/<kernel>/<version>."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    @staticmethod
    def _mock_client(mock_client):
        mock_kaggle = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @staticmethod
    def _sent(api_method):
        return api_method.call_args[0][0]


class TestVersionIsSentAsVersionLabel(_KernelRequestTestCase):

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_files_sends_the_version(self, mock_client):
        kaggle = self._mock_client(mock_client)

        self.api.kernels_list_files("owner/my-slug/2")

        request = self._sent(kaggle.kernels.kernels_api_client.list_kernel_files)
        self.assertEqual(request.kernel_slug, "my-slug")
        self.assertEqual(request.version_label, "v2")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_status_sends_the_version(self, mock_client):
        kaggle = self._mock_client(mock_client)

        self.api.kernels_status("owner/my-slug/2")

        request = self._sent(kaggle.kernels.kernels_api_client.get_kernel_session_status)
        self.assertEqual(request.kernel_slug, "my-slug")
        self.assertEqual(request.version_label, "v2")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_logs_sends_the_version(self, mock_client):
        kaggle = self._mock_client(mock_client)
        kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = MagicMock(log="log of v2")

        log = self.api.kernels_logs("owner/my-slug/2")

        request = self._sent(kaggle.kernels.kernels_api_client.list_kernel_session_output)
        self.assertEqual(request.kernel_slug, "my-slug")
        self.assertEqual(request.version_label, "v2")
        self.assertEqual(log, "log of v2")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_output_sends_the_version_on_every_page(self, mock_client):
        kaggle = self._mock_client(mock_client)
        list_output = kaggle.kernels.kernels_api_client.list_kernel_session_output
        list_output.side_effect = [
            MagicMock(files=[], log=None, next_page_token="page-2"),
            MagicMock(files=[], log=None, next_page_token=""),
        ]
        target = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target, True)

        self.api.kernels_output("owner/my-slug/2", target)

        self.assertEqual(list_output.call_count, 2)
        for call in list_output.call_args_list:
            request = call[0][0]
            self.assertEqual(request.kernel_slug, "my-slug")
            self.assertEqual(request.version_label, "v2")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_an_already_prefixed_version_is_not_prefixed_twice(self, mock_client):
        kaggle = self._mock_client(mock_client)

        self.api.kernels_status("owner/my-slug/v2")

        self.assertEqual(self._sent(kaggle.kernels.kernels_api_client.get_kernel_session_status).version_label, "v2")


class TestNoVersionIsUnchanged(_KernelRequestTestCase):
    """Without a version the requests must stay exactly as they were, so the latest version is used."""

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_files_without_a_version(self, mock_client):
        kaggle = self._mock_client(mock_client)

        self.api.kernels_list_files("owner/my-slug")

        self.assertEqual(self._sent(kaggle.kernels.kernels_api_client.list_kernel_files).version_label, "")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_status_without_a_version(self, mock_client):
        kaggle = self._mock_client(mock_client)

        self.api.kernels_status("owner/my-slug")

        self.assertEqual(self._sent(kaggle.kernels.kernels_api_client.get_kernel_session_status).version_label, "")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_logs_without_a_version(self, mock_client):
        kaggle = self._mock_client(mock_client)
        kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = MagicMock(log="")

        self.api.kernels_logs("owner/my-slug")

        request = self._sent(kaggle.kernels.kernels_api_client.list_kernel_session_output)
        self.assertEqual(request.version_label, "")


class TestFollowSendsTheVersionAsAQueryParameter(unittest.TestCase):
    """`kernels logs --follow` streams from a raw endpoint rather than a generated request."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    def _stream(self, kernel):
        response = MagicMock()
        response.headers = {"Content-Type": "application/json"}
        response.text = "[]"
        session = MagicMock()
        session.headers = {"User-Agent": "test", "Content-Type": "application/json"}
        session.auth = None
        session.get.return_value = response
        http_client = MagicMock(_session=session, _endpoint="http://localhost", _env="LOCAL")
        kaggle = MagicMock(_http_client=http_client)

        with patch.object(KaggleApi, "build_kaggle_client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=kaggle)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            list(self.api.kernels_logs_stream(kernel))
        return session.get.call_args

    def test_version_goes_in_the_query_not_the_path(self):
        call = self._stream("owner/my-slug/2")

        self.assertEqual(call[0][0], "http://localhost/api/v1/kernels/logs/stream/owner/my-slug")
        self.assertEqual(call.kwargs["params"], {"versionLabel": "v2"})

    def test_no_version_adds_no_query(self):
        call = self._stream("owner/my-slug")

        self.assertEqual(call[0][0], "http://localhost/api/v1/kernels/logs/stream/owner/my-slug")
        self.assertIsNone(call.kwargs["params"])


if __name__ == "__main__":
    unittest.main()
