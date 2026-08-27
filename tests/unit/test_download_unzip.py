# coding=utf-8
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../../src")

from kaggle.api.kaggle_api_extended import (
    KaggleApi,
    _extract_and_remove_zip,
    _is_auto_compressed,
)


def _signed_url(base_name):
    """Builds a URL shaped like the signed storage URLs the API redirects downloads to."""
    return f"https://storage.googleapis.com/kagglesdsdata/datasets/1/2/{base_name}?GoogleAccessId=x&Signature=y"


def _write_zip(path, members):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


class TestIsAutoCompressed(unittest.TestCase):
    """Tests for _is_auto_compressed, which recognizes the server side zip wrapper."""

    def test_wrapper_around_requested_file(self):
        self.assertTrue(_is_auto_compressed("out/creditcard.csv.zip", "creditcard.csv"))

    def test_wrapper_around_file_requested_by_nested_path(self):
        self.assertTrue(_is_auto_compressed("out/a/b/Food_Costs.csv.zip", "a/b/Food_Costs.csv"))

    def test_wrapper_around_file_requested_with_backslashes(self):
        self.assertTrue(_is_auto_compressed("out/a/b/Food_Costs.csv.zip", "a\\b\\Food_Costs.csv"))

    def test_file_that_is_itself_a_zip_is_not_a_wrapper(self):
        # The download is not renamed, so a genuine .zip in the dataset must be left alone.
        self.assertFalse(_is_auto_compressed("out/archive.zip", "archive.zip"))

    def test_uncompressed_response_is_not_a_wrapper(self):
        self.assertFalse(_is_auto_compressed("out/train.csv", "train.csv"))

    def test_unrelated_name_is_not_a_wrapper(self):
        self.assertFalse(_is_auto_compressed("out/other.csv.zip", "train.csv"))

    def test_empty_file_name_is_not_a_wrapper(self):
        self.assertFalse(_is_auto_compressed("out/train.csv.zip", ""))


class TestExtractAndRemoveZip(unittest.TestCase):
    """Tests for _extract_and_remove_zip, shared by every download that honors --unzip."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.destination = os.path.join(self.root, "dest")
        os.makedirs(self.destination)

    def test_extracts_members_and_removes_the_archive(self):
        archive = os.path.join(self.destination, "bundle.zip")
        _write_zip(archive, {"train.csv": b"a,b\n"})

        _extract_and_remove_zip(archive, self.destination)

        self.assertEqual(sorted(os.listdir(self.destination)), ["train.csv"])
        with open(os.path.join(self.destination, "train.csv"), "rb") as extracted:
            self.assertEqual(extracted.read(), b"a,b\n")

    def test_preserves_directories_inside_the_archive(self):
        archive = os.path.join(self.destination, "bundle.zip")
        _write_zip(archive, {"train/labels.csv": b"x"})

        _extract_and_remove_zip(archive, self.destination)

        self.assertTrue(os.path.isfile(os.path.join(self.destination, "train", "labels.csv")))

    def test_members_cannot_escape_the_destination(self):
        archive = os.path.join(self.destination, "bundle.zip")
        _write_zip(archive, {"../escaped.txt": b"x", "a/../../also.txt": b"y"})

        _extract_and_remove_zip(archive, self.destination)

        self.assertEqual(sorted(os.listdir(self.root)), ["dest"])

    def test_corrupted_archive_raises_value_error(self):
        archive = os.path.join(self.destination, "bundle.zip")
        with open(archive, "wb") as broken:
            broken.write(b"not a zip")

        with self.assertRaises(ValueError) as context:
            _extract_and_remove_zip(archive, self.destination)

        self.assertIn("github.com/Kaggle/kaggle-cli/issues", str(context.exception))
        self.assertTrue(os.path.exists(archive))

    def test_missing_archive_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            _extract_and_remove_zip(os.path.join(self.destination, "absent.zip"), self.destination)


class _DownloadTestCase(unittest.TestCase):
    """Shared setup for the download methods that honor --unzip."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.destination = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.destination, True)

    @staticmethod
    def _mock_client(mock_client, base_name):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = _signed_url(base_name)
        mock_response.url = _signed_url(base_name)
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_kaggle.competitions.competition_api_client.download_data_file.return_value = mock_response
        mock_kaggle.competitions.competition_api_client.download_data_files.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @staticmethod
    def _writes_zip(members):
        """A download_file side effect that puts a real archive where the caller expects it."""

        def write(*args, **kwargs):
            _write_zip(args[1], members)

        return write

    @staticmethod
    def _writes_plain_file(data=b"payload"):
        def write(*args, **kwargs):
            outfile = args[1]
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, "wb") as handle:
                handle.write(data)

        return write

    def _paths(self, *parts):
        return os.path.join(self.destination, *parts)


class TestDatasetDownloadFileUnzip(_DownloadTestCase):

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_unzip_extracts_the_wrapper_and_removes_it(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "creditcard.csv.zip")
        mock_download_file.side_effect = self._writes_zip({"creditcard.csv": b"a,b\n1,2\n"})

        self.api.dataset_download_file("owner/ds", "creditcard.csv", path=self.destination, unzip=True)

        self.assertTrue(os.path.isfile(self._paths("creditcard.csv")))
        self.assertFalse(os.path.exists(self._paths("creditcard.csv.zip")))

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_without_unzip_the_wrapper_is_kept(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "creditcard.csv.zip")
        mock_download_file.side_effect = self._writes_zip({"creditcard.csv": b"a,b\n1,2\n"})

        self.api.dataset_download_file("owner/ds", "creditcard.csv", path=self.destination)

        self.assertTrue(os.path.isfile(self._paths("creditcard.csv.zip")))
        self.assertFalse(os.path.exists(self._paths("creditcard.csv")))

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_a_file_that_is_itself_a_zip_is_not_extracted(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "archive.zip")
        mock_download_file.side_effect = self._writes_zip({"inner.csv": b"x"})

        self.api.dataset_download_file("owner/ds", "archive.zip", path=self.destination, unzip=True)

        self.assertTrue(os.path.isfile(self._paths("archive.zip")))
        self.assertFalse(os.path.exists(self._paths("inner.csv")))

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_uncompressed_response_with_unzip_is_left_alone(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "train.csv")
        mock_download_file.side_effect = self._writes_plain_file(b"a,b\n")

        self.api.dataset_download_file("owner/ds", "train.csv", path=self.destination, unzip=True)

        self.assertEqual(sorted(os.listdir(self.destination)), ["train.csv"])

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_nested_request_extracts_beside_the_requested_path(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "Food_Costs.csv.zip")
        mock_download_file.side_effect = self._writes_zip({"Food_Costs.csv": b"x"})

        self.api.dataset_download_file(
            "owner/ds", "WICAgencies2014ytd/Food_Costs.csv", path=self.destination, unzip=True
        )

        self.assertTrue(os.path.isfile(self._paths("WICAgencies2014ytd", "Food_Costs.csv")))
        self.assertFalse(os.path.exists(self._paths("WICAgencies2014ytd", "Food_Costs.csv.zip")))

    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_already_downloaded_archive_is_still_extracted(self, mock_client, mock_download_file, mock_needed):
        # Matches the cached behavior of a full dataset download, see #1086.
        self._mock_client(mock_client, "creditcard.csv.zip")
        _write_zip(self._paths("creditcard.csv.zip"), {"creditcard.csv": b"cached"})

        self.api.dataset_download_file("owner/ds", "creditcard.csv", path=self.destination, unzip=True)

        mock_download_file.assert_not_called()
        self.assertTrue(os.path.isfile(self._paths("creditcard.csv")))
        self.assertFalse(os.path.exists(self._paths("creditcard.csv.zip")))


class TestDatasetDownloadCliForwarding(unittest.TestCase):

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    @patch.object(KaggleApi, "dataset_download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_cli_forwards_unzip_to_the_single_file_download(self, mock_client, mock_download_file):
        mock_kaggle = MagicMock()
        mock_kaggle.datasets.dataset_api_client.get_dataset_metadata.return_value.error_message = None
        mock_kaggle.datasets.dataset_api_client.get_dataset_metadata.return_value.info.licenses = []
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_download_cli("owner/ds", file_name="train.csv", path="/tmp/x", unzip=True)

        self.assertTrue(mock_download_file.call_args.kwargs["unzip"])


class TestCompetitionDownloadUnzip(_DownloadTestCase):

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_unzip_extracts_the_bundle_and_removes_it(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "archive.zip")
        mock_download_file.side_effect = self._writes_zip({"train.csv": b"a\n", "test/x.csv": b"b\n"})

        self.api.competition_download_files("my-competition", path=self.destination, unzip=True)

        self.assertTrue(os.path.isfile(self._paths("train.csv")))
        self.assertTrue(os.path.isfile(self._paths("test", "x.csv")))
        self.assertFalse(os.path.exists(self._paths("my-competition.zip")))

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_without_unzip_the_bundle_is_kept(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "archive.zip")
        mock_download_file.side_effect = self._writes_zip({"train.csv": b"a\n"})

        self.api.competition_download_files("my-competition", path=self.destination)

        self.assertEqual(sorted(os.listdir(self.destination)), ["my-competition.zip"])

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_bundle_that_is_not_a_zip_is_left_alone(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "archive.tar")
        mock_download_file.side_effect = self._writes_plain_file(b"not a zip")

        self.api.competition_download_files("my-competition", path=self.destination, unzip=True)

        self.assertEqual(sorted(os.listdir(self.destination)), ["my-competition.tar"])

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_single_file_wrapper_is_extracted(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "big.csv.zip")
        mock_download_file.side_effect = self._writes_zip({"big.csv": b"x"})

        self.api.competition_download_file("my-competition", "big.csv", path=self.destination, unzip=True)

        self.assertTrue(os.path.isfile(self._paths("big.csv")))
        self.assertFalse(os.path.exists(self._paths("big.csv.zip")))

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_single_uncompressed_file_with_unzip_is_left_alone(self, mock_client, mock_download_file, mock_needed):
        self._mock_client(mock_client, "train.csv")
        mock_download_file.side_effect = self._writes_plain_file(b"a,b\n")

        self.api.competition_download_file("my-competition", "train.csv", path=self.destination, unzip=True)

        self.assertEqual(sorted(os.listdir(self.destination)), ["train.csv"])


class TestCompetitionDownloadCliForwarding(unittest.TestCase):

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    @patch.object(KaggleApi, "competition_download_files")
    def test_cli_forwards_unzip_to_the_bundle_download(self, mock_download_files):
        self.api.competition_download_cli("my-competition", path="/tmp/x", unzip=True)
        self.assertIs(mock_download_files.call_args[0][4], True)

    @patch.object(KaggleApi, "competition_download_file")
    def test_cli_forwards_unzip_to_the_single_file_download(self, mock_download_file):
        self.api.competition_download_cli("my-competition", file_name="train.csv", path="/tmp/x", unzip=True)
        self.assertIs(mock_download_file.call_args[0][5], True)


if __name__ == "__main__":
    unittest.main()
