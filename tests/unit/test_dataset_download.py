# coding=utf-8
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../../src")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestDatasetDownload(unittest.TestCase):

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch("zipfile.ZipFile")
    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_cached_unzip_succeeds(
        self, mock_client, mock_download_file, mock_download_needed, mock_zipfile, mock_remove, mock_exists
    ):
        """Case 1: When dataset is cached (download_needed is False) and unzip=True,
        it should still extract the existing ZIP and remove it, without downloading again.
        """
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        path = os.path.normpath("/tmp/dummy")

        # Call method with unzip=True, force=False
        self.api.dataset_download_files("owner/my-dataset", path=path, unzip=True, force=False)

        # Verify download_file was NOT called
        mock_download_file.assert_not_called()

        # Verify zipfile.ZipFile was called to extract the file
        expected_outfile = os.path.join(path, "my-dataset.zip")
        mock_zipfile.assert_called_once()
        called_args = mock_zipfile.call_args[0][0]
        self.assertEqual(os.path.normpath(called_args), expected_outfile)

        # Verify extractall and remove were called
        mock_zipfile().__enter__().extractall.assert_called_once_with(path)
        mock_remove.assert_called_once()
        self.assertEqual(os.path.normpath(mock_remove.call_args[0][0]), expected_outfile)

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch("zipfile.ZipFile")
    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_fresh_unzip_succeeds(
        self, mock_client, mock_download_file, mock_download_needed, mock_zipfile, mock_remove, mock_exists
    ):
        """Case 2: When dataset is not cached (download_needed is True) and unzip=True,
        it should download the ZIP, extract it, and remove it.
        """
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        path = os.path.normpath("/tmp/dummy")

        # Call method with unzip=True, force=False
        self.api.dataset_download_files("owner/my-dataset", path=path, unzip=True, force=False)

        # Verify download_file was called
        mock_download_file.assert_called_once()

        # Verify zipfile.ZipFile was called to extract the file
        expected_outfile = os.path.join(path, "my-dataset.zip")
        mock_zipfile.assert_called_once()
        called_args = mock_zipfile.call_args[0][0]
        self.assertEqual(os.path.normpath(called_args), expected_outfile)

        # Verify extractall and remove were called
        mock_zipfile().__enter__().extractall.assert_called_once_with(path)
        mock_remove.assert_called_once()
        self.assertEqual(os.path.normpath(mock_remove.call_args[0][0]), expected_outfile)

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_succeeds(self, mock_client, mock_download_file, mock_download_needed):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = "http://example.com/download/testuser/test-dataset/file1.csv?token=123"
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_download_file("owner/my-dataset", "file1.csv", path="/tmp/download")

        self.assertTrue(result)
        mock_kaggle.datasets.dataset_api_client.download_dataset.assert_called_once()
        request = mock_kaggle.datasets.dataset_api_client.download_dataset.call_args[0][0]
        self.assertEqual(request.owner_slug, "owner")
        self.assertEqual(request.dataset_slug, "my-dataset")
        self.assertEqual(request.file_name, "file1.csv")
        self.assertEqual(request.dataset_version_number, 0)

        mock_download_needed.assert_called_once()
        mock_download_file.assert_called_once()
        expected_outfile = os.path.join("/tmp/download", "file1.csv")
        self.assertEqual(mock_download_file.call_args[0][1], expected_outfile)

    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_not_needed_skips_download(
        self, mock_client, mock_download_file, mock_download_needed
    ):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = "http://example.com/download/testuser/test-dataset/file1.csv?token=123"
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_download_file("owner/my-dataset", "file1.csv", path="/tmp/download")

        self.assertFalse(result)
        mock_download_needed.assert_called_once()
        mock_download_file.assert_not_called()

    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_force_downloads(self, mock_client, mock_download_file, mock_download_needed):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = "http://example.com/download/testuser/test-dataset/file1.csv?token=123"
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_download_file("owner/my-dataset", "file1.csv", path="/tmp/download", force=True)

        self.assertTrue(result)
        mock_download_needed.assert_not_called()
        mock_download_file.assert_called_once()

    @patch.object(KaggleApi, "get_default_download_dir", return_value="/tmp/default")
    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_default_path_succeeds(
        self, mock_client, mock_download_file, mock_download_needed, mock_get_default_dir
    ):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = "http://example.com/download/testuser/test-dataset/file1.csv?token=123"
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_download_file("owner/my-dataset", "file1.csv")

        mock_get_default_dir.assert_called_once_with("datasets", "owner", "my-dataset")
        expected_outfile = os.path.join("/tmp/default", "file1.csv")
        self.assertEqual(mock_download_file.call_args[0][1], expected_outfile)

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_no_owner_defaults_owner_succeeds(
        self, mock_client, mock_download_file, mock_download_needed
    ):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = "http://example.com/download/testuser/test-dataset/file1.csv?token=123"
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_download_file("my-dataset", "file1.csv", path="/tmp/download")

        request = mock_kaggle.datasets.dataset_api_client.download_dataset.call_args[0][0]
        self.assertEqual(request.owner_slug, "testuser")
        self.assertEqual(request.dataset_slug, "my-dataset")

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_with_version_succeeds(self, mock_client, mock_download_file, mock_download_needed):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = "http://example.com/download/testuser/test-dataset/file1.csv?token=123"
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_download_file("owner/my-dataset/3", "file1.csv", path="/tmp/download")

        request = mock_kaggle.datasets.dataset_api_client.download_dataset.call_args[0][0]
        self.assertEqual(request.owner_slug, "owner")
        self.assertEqual(request.dataset_slug, "my-dataset")
        self.assertEqual(request.dataset_version_number, 3)

    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_only_a_pinned_version_requires_an_exact_local_match(self, mock_client, mock_download_needed):
        mock_kaggle = MagicMock()
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value.request.url = (
            "http://example.com/download/file1.csv?token=123"
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        for dataset, exact in [("owner/my-dataset/1", True), ("owner/my-dataset", False)]:
            for download in (
                lambda: self.api.dataset_download_files(dataset, path="/tmp/download"),
                lambda: self.api.dataset_download_file(dataset, "file1.csv", path="/tmp/download"),
            ):
                mock_download_needed.reset_mock()
                download()
                self.assertEqual(mock_download_needed.call_args.kwargs["exact"], exact)


class TestDownloadNeeded(unittest.TestCase):
    """Every dataset version is saved under the same name, so a pinned request must not
    treat a newer local copy (possibly another version) as already downloaded."""

    LAST_MODIFIED = "Mon, 16 Nov 2026 10:00:00 GMT"

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        fd, self.outfile = tempfile.mkstemp()
        os.write(fd, b"a,b\n999,999\n")
        os.close(fd)
        self.addCleanup(os.remove, self.outfile)

    def _response(self, size, last_modified=LAST_MODIFIED):
        response = MagicMock()
        response.headers = {"Last-Modified": last_modified, "Content-Length": str(size)}
        return response

    def _set_local_date(self, last_modified):
        # The same conversion download_file uses when it stamps a finished download.
        remote_date = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")
        stamp = time.mktime(remote_date.timetuple())
        os.utime(self.outfile, times=(stamp, stamp))

    def test_newer_local_copy_is_kept_for_an_unpinned_request(self):
        self._set_local_date("Tue, 17 Nov 2026 10:00:00 GMT")
        self.assertFalse(self.api.download_needed(self._response(size=8), self.outfile))

    def test_newer_local_copy_is_replaced_for_a_pinned_older_version(self):
        self._set_local_date("Tue, 17 Nov 2026 10:00:00 GMT")
        self.assertTrue(self.api.download_needed(self._response(size=8), self.outfile, exact=True))

    def test_same_size_but_different_date_is_replaced_for_a_pinned_version(self):
        self._set_local_date("Tue, 17 Nov 2026 10:00:00 GMT")
        size = os.path.getsize(self.outfile)
        self.assertTrue(self.api.download_needed(self._response(size=size), self.outfile, exact=True))

    def test_exact_copy_of_a_pinned_version_is_not_downloaded_again(self):
        self._set_local_date(self.LAST_MODIFIED)
        size = os.path.getsize(self.outfile)
        self.assertFalse(self.api.download_needed(self._response(size=size), self.outfile, exact=True))


if __name__ == "__main__":
    unittest.main()
