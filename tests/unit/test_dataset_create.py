# coding=utf-8
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys
from requests.exceptions import HTTPError

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestDatasetCreate(unittest.TestCase):
    """Tests for dataset_create_new and dataset_create_version."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True

    def _get_valid_metadata(self):
        return {
            "id": "testuser/test-dataset",
            "title": "Test Dataset Title",
            "licenses": [{"name": "CC0-1.0"}],
        }

    def _write_metadata(self, folder, metadata):
        meta_file_path = os.path.join(folder, "dataset-metadata.json")
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def test_dataset_create_new_invalid_folder_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_create_new("/non/existent/folder")
        self.assertIn("Invalid folder", str(context.exception))

    def test_dataset_create_new_default_slug_fails(self):
        metadata = self._get_valid_metadata()
        metadata["id"] = "testuser/INSERT_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("Default slug detected", str(context.exception))

    def test_dataset_create_new_default_title_fails(self):
        metadata = self._get_valid_metadata()
        metadata["title"] = "INSERT_TITLE_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("Default title detected", str(context.exception))

    def test_dataset_create_new_multiple_licenses_fails(self):
        metadata = self._get_valid_metadata()
        metadata["licenses"] = [{"name": "CC0-1.0"}, {"name": "CC-BY-4.0"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("Please specify exactly one license", str(context.exception))

    def test_dataset_create_new_no_licenses_fails(self):
        metadata = self._get_valid_metadata()
        metadata["licenses"] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("Please specify exactly one license", str(context.exception))

    def test_dataset_create_new_short_slug_fails(self):
        metadata = self._get_valid_metadata()
        metadata["id"] = "testuser/abc"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("The dataset slug must be between 6 and 50 characters", str(context.exception))

    def test_dataset_create_new_long_slug_fails(self):
        metadata = self._get_valid_metadata()
        metadata["id"] = "testuser/" + ("a" * 51)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("The dataset slug must be between 6 and 50 characters", str(context.exception))

    def test_dataset_create_new_short_title_fails(self):
        metadata = self._get_valid_metadata()
        metadata["title"] = "abc"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("The dataset title must be between 6 and 50 characters", str(context.exception))

    def test_dataset_create_new_long_title_fails(self):
        metadata = self._get_valid_metadata()
        metadata["title"] = "a" * 51
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("The dataset title must be between 6 and 50 characters", str(context.exception))

    def test_dataset_create_new_short_subtitle_fails(self):
        metadata = self._get_valid_metadata()
        metadata["subtitle"] = "too short"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("Subtitle length must be between 20 and 80 characters", str(context.exception))

    def test_dataset_create_new_long_subtitle_fails(self):
        metadata = self._get_valid_metadata()
        metadata["subtitle"] = "a" * 81
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("Subtitle length must be between 20 and 80 characters", str(context.exception))

    @patch.object(KaggleApi, "dataset_status")
    def test_dataset_create_new_duplicate_title_fails(self, mock_status):
        mock_status.return_value = MagicMock()
        metadata = self._get_valid_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            response = self.api.dataset_create_new(tmpdir)
            self.assertEqual(response.status, "error")
            self.assertIn("already in use by a dataset", response.error)
            mock_status.assert_called_once_with("testuser/test-dataset")

    @patch.object(KaggleApi, "dataset_status")
    @patch.object(KaggleApi, "upload_files")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_create_new_valid_metadata_succeeds(self, mock_client, mock_upload, mock_status):
        mock_status.side_effect = HTTPError()

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.error = ""
        mock_kaggle.datasets.dataset_api_client.create_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_metadata()
        metadata["subtitle"] = "This is a valid subtitle with enough length"
        metadata["description"] = "Some description"
        metadata["keywords"] = ["key1", "key2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            response = self.api.dataset_create_new(tmpdir, public=True)

            mock_status.assert_called_once_with("testuser/test-dataset")
            mock_upload.assert_called_once()
            mock_kaggle.datasets.dataset_api_client.create_dataset.assert_called_once()
            request = mock_kaggle.datasets.dataset_api_client.create_dataset.call_args[0][0]
            self.assertEqual(request.title, "Test Dataset Title")
            self.assertEqual(request.slug, "test-dataset")
            self.assertEqual(request.owner_slug, "testuser")
            self.assertEqual(request.license_name, "CC0-1.0")
            self.assertEqual(request.subtitle, "This is a valid subtitle with enough length")
            self.assertEqual(request.description, "Some description")
            self.assertEqual(request.category_ids, ["key1", "key2"])
            self.assertFalse(request.is_private)
            self.assertEqual(response, mock_response)

    @patch.object(KaggleApi, "dataset_status")
    @patch.object(KaggleApi, "upload_files")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_create_new_with_valid_resources_succeeds(self, mock_client, mock_upload, mock_status):
        mock_status.side_effect = HTTPError()
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.error = ""
        mock_kaggle.datasets.dataset_api_client.create_dataset.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_metadata()
        metadata["resources"] = [{"path": "file1.csv"}, {"path": "file2.csv"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            open(os.path.join(tmpdir, "file1.csv"), "w").close()
            open(os.path.join(tmpdir, "file2.csv"), "w").close()

            response = self.api.dataset_create_new(tmpdir)

            self.assertEqual(response, mock_response)
            mock_upload.assert_called_once()

    def test_dataset_create_new_with_missing_resources_fails(self):
        metadata = self._get_valid_metadata()
        metadata["resources"] = [{"path": "file1.csv"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("file1.csv does not exist", str(context.exception))

    def test_dataset_create_new_with_duplicate_resources_fails(self):
        metadata = self._get_valid_metadata()
        metadata["resources"] = [{"path": "file1.csv"}, {"path": "file1.csv"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            open(os.path.join(tmpdir, "file1.csv"), "w").close()

            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_new(tmpdir)
            self.assertIn("path was specified more than once", str(context.exception))

    def test_dataset_create_version_invalid_folder_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_create_version("/non/existent/folder", "notes")
        self.assertIn("Invalid folder", str(context.exception))

    def test_dataset_create_version_missing_id_fails(self):
        metadata = self._get_valid_metadata()
        del metadata["id"]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_version(tmpdir, "notes")
            self.assertIn("ID or slug must be specified", str(context.exception))

    def test_dataset_create_version_default_slug_fails(self):
        metadata = self._get_valid_metadata()
        metadata["id"] = "testuser/INSERT_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_version(tmpdir, "notes")
            self.assertIn("Default slug detected", str(context.exception))

    def test_dataset_create_version_invalid_subtitle_fails(self):
        metadata = self._get_valid_metadata()
        metadata["subtitle"] = "too short"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_version(tmpdir, "notes")
            self.assertIn("Subtitle length must be between 20 and 80 characters", str(context.exception))

    def test_dataset_create_version_invalid_dataset_slug_fails(self):
        metadata = self._get_valid_metadata()
        metadata["id"] = "invalid_slug_no_slash"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.dataset_create_version(tmpdir, "notes")
            self.assertIn("Dataset must be specified in the form", str(context.exception))

    @patch.object(KaggleApi, "upload_files")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_create_version_with_slug_succeeds(self, mock_client, mock_upload):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.datasets.dataset_api_client.create_dataset_version.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_metadata()
        metadata["subtitle"] = "This is a valid subtitle with enough length"
        metadata["description"] = "Some description"
        metadata["keywords"] = ["key1", "key2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            response = self.api.dataset_create_version(tmpdir, "version notes here", delete_old_versions=True)

            mock_upload.assert_called_once()
            mock_kaggle.datasets.dataset_api_client.create_dataset_version.assert_called_once()
            request = mock_kaggle.datasets.dataset_api_client.create_dataset_version.call_args[0][0]
            self.assertEqual(request.owner_slug, "testuser")
            self.assertEqual(request.dataset_slug, "test-dataset")

            body = request.body
            self.assertEqual(body.version_notes, "version notes here")
            self.assertEqual(body.subtitle, "This is a valid subtitle with enough length")
            self.assertEqual(body.description, "Some description")
            self.assertEqual(body.category_ids, ["key1", "key2"])
            self.assertTrue(body.delete_old_versions)

            self.assertEqual(response, mock_response)

    @patch.object(KaggleApi, "upload_files")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_create_version_with_id_succeeds(self, mock_client, mock_upload):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.datasets.dataset_api_client.create_dataset_version_by_id.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = {
            "id_no": "12345",
            "subtitle": "This is a valid subtitle with enough length",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            response = self.api.dataset_create_version(tmpdir, "version notes here")

            mock_upload.assert_called_once()
            mock_kaggle.datasets.dataset_api_client.create_dataset_version_by_id.assert_called_once()
            request = mock_kaggle.datasets.dataset_api_client.create_dataset_version_by_id.call_args[0][0]
            self.assertEqual(request.id, 12345)

            body = request.body
            self.assertEqual(body.version_notes, "version notes here")
            self.assertEqual(body.subtitle, "This is a valid subtitle with enough length")

            self.assertEqual(response, mock_response)


if __name__ == "__main__":
    unittest.main()
