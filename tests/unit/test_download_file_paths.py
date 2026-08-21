# coding=utf-8
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../../src")

from kaggle.api.kaggle_api_extended import KaggleApi, _resolve_download_path


def _signed_url(base_name):
    """Builds a URL shaped like the signed storage URLs the API redirects downloads to."""
    return f"https://storage.googleapis.com/kagglesdsdata/datasets/1/2/{base_name}?GoogleAccessId=x&Signature=y"


class TestResolveDownloadPath(unittest.TestCase):
    """Tests for _resolve_download_path, which decides where a single file download lands."""

    def _resolve(self, file_name, base_name=None):
        url = _signed_url(base_name if base_name is not None else file_name.split("/")[-1])
        return os.path.normpath(_resolve_download_path("dest", file_name, url))

    def test_flat_file_lands_directly_in_destination(self):
        self.assertEqual(self._resolve("train.csv"), os.path.normpath("dest/train.csv"))

    def test_nested_file_keeps_its_directory(self):
        self.assertEqual(
            self._resolve("WICAgencies2014ytd/Food_Costs.csv"),
            os.path.normpath("dest/WICAgencies2014ytd/Food_Costs.csv"),
        )

    def test_deeply_nested_file_keeps_every_level(self):
        self.assertEqual(
            self._resolve("kaggle_evaluation/core/generated/__init__.py"),
            os.path.normpath("dest/kaggle_evaluation/core/generated/__init__.py"),
        )

    def test_same_base_name_in_different_directories_resolves_to_different_paths(self):
        first = self._resolve("WICAgencies2013ytd/Food_Costs.csv")
        second = self._resolve("WICAgencies2014ytd/Food_Costs.csv")
        self.assertNotEqual(first, second)

    def test_flat_compressed_response_keeps_the_zip_suffix(self):
        self.assertEqual(
            self._resolve("creditcard.csv", base_name="creditcard.csv.zip"),
            os.path.normpath("dest/creditcard.csv.zip"),
        )

    def test_nested_compressed_response_keeps_directory_and_zip_suffix(self):
        self.assertEqual(
            self._resolve("cord_19_embeddings/embeddings.csv", base_name="embeddings.csv.zip"),
            os.path.normpath("dest/cord_19_embeddings/embeddings.csv.zip"),
        )

    def test_backslash_separators_are_normalized(self):
        self.assertEqual(
            self._resolve("train\\labels\\y.csv", base_name="y.csv"),
            os.path.normpath("dest/train/labels/y.csv"),
        )

    def test_redundant_separators_and_dot_segments_are_dropped(self):
        self.assertEqual(self._resolve("./train//y.csv", base_name="y.csv"), os.path.normpath("dest/train/y.csv"))

    def test_leading_separator_stays_inside_the_destination(self):
        self.assertEqual(self._resolve("/train/y.csv", base_name="y.csv"), os.path.normpath("dest/train/y.csv"))

    def test_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            _resolve_download_path("dest", "../evil.csv", _signed_url("evil.csv"))

    def test_rejects_traversal_hidden_mid_path(self):
        with self.assertRaises(ValueError):
            _resolve_download_path("dest", "train/../../../evil.csv", _signed_url("evil.csv"))

    def test_rejects_backslash_traversal(self):
        with self.assertRaises(ValueError):
            _resolve_download_path("dest", "..\\..\\evil.csv", _signed_url("evil.csv"))


class TestDownloadFileDestinations(unittest.TestCase):
    """Tests that both single file download methods write to the requested path."""

    DESTINATION = os.path.normpath("/tmp/download")

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    @staticmethod
    def _mock_client(mock_client, base_name):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.request.url = _signed_url(base_name)
        mock_kaggle.datasets.dataset_api_client.download_dataset.return_value = mock_response
        mock_kaggle.competitions.competition_api_client.download_data_file.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_writes_nested_path(self, mock_client, mock_download_file, mock_download_needed):
        self._mock_client(mock_client, "Food_Costs.csv")

        self.api.dataset_download_file("owner/my-dataset", "WICAgencies2014ytd/Food_Costs.csv", path=self.DESTINATION)

        expected = os.path.join(self.DESTINATION, "WICAgencies2014ytd", "Food_Costs.csv")
        self.assertEqual(mock_download_file.call_args[0][1], expected)

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_download_file_flat_path_is_unchanged(self, mock_client, mock_download_file, mock_download_needed):
        self._mock_client(mock_client, "train.csv")

        self.api.dataset_download_file("owner/my-dataset", "train.csv", path=self.DESTINATION)

        self.assertEqual(mock_download_file.call_args[0][1], os.path.join(self.DESTINATION, "train.csv"))

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competition_download_file_writes_nested_path(self, mock_client, mock_download_file, mock_download_needed):
        self._mock_client(mock_client, "rsna_gateway.py")

        self.api.competition_download_file(
            "some-competition", "kaggle_evaluation/rsna_gateway.py", path=self.DESTINATION
        )

        expected = os.path.join(self.DESTINATION, "kaggle_evaluation", "rsna_gateway.py")
        self.assertEqual(mock_download_file.call_args[0][1], expected)

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competition_download_file_flat_path_is_unchanged(
        self, mock_client, mock_download_file, mock_download_needed
    ):
        self._mock_client(mock_client, "train.csv")

        self.api.competition_download_file("some-competition", "train.csv", path=self.DESTINATION)

        self.assertEqual(mock_download_file.call_args[0][1], os.path.join(self.DESTINATION, "train.csv"))

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_competition_download_file_rejects_traversal(self, mock_client, mock_download_file, mock_download_needed):
        self._mock_client(mock_client, "evil.csv")

        with self.assertRaises(ValueError):
            self.api.competition_download_file("some-competition", "../evil.csv", path=self.DESTINATION)

        mock_download_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
