import unittest
from unittest.mock import MagicMock, patch
import os
import json
import tempfile
import shutil

# Ensure parent directory is in path for imports
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_types import DatasetSettings, DatasetSettingsFile, DatasetSettingsFileColumn
from kagglesdk.datasets.types.dataset_api_service import ApiUpdateDatasetMetadataRequest


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    api.config_values = {"username": "owner"}
    return api


class TestDatasetMetadataUpdate(unittest.TestCase):
    def setUp(self):
        self.api = _make_api()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_data_succeeds(self, mock_build):
        # Prepare metadata file with 'data'
        metadata = {
            "title": "New Title",
            "data": [
                {
                    "name": "file.csv",
                    "description": "file desc",
                    "columns": [{"name": "col1", "description": "col1 desc", "type": "string"}],
                }
            ],
        }
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        # Assertions
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.assert_called_once()
        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        self.assertIsInstance(call_args, ApiUpdateDatasetMetadataRequest)
        self.assertEqual(call_args.settings.title, "New Title")
        self.assertEqual(len(call_args.settings.data), 1)
        self.assertEqual(call_args.settings.data[0].name, "file.csv")
        self.assertEqual(call_args.settings.data[0].description, "file desc")
        self.assertEqual(len(call_args.settings.data[0].columns), 1)
        self.assertEqual(call_args.settings.data[0].columns[0].name, "col1")
        self.assertEqual(call_args.settings.data[0].columns[0].description, "col1 desc")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_resources_succeeds(self, mock_build):
        # Prepare metadata file with 'resources'
        metadata = {
            "title": "New Title",
            "resources": [
                {
                    "path": "file.csv",
                    "description": "file desc",
                    "schema": {
                        "fields": [
                            {"name": "col1", "description": "col1 desc", "type": "string"},
                            {"name": "col2", "title": "col2 desc", "type": "integer"},  # test title fallback
                        ]
                    },
                }
            ],
        }
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        # Assertions
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.assert_called_once()
        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        self.assertIsInstance(call_args, ApiUpdateDatasetMetadataRequest)
        self.assertEqual(call_args.settings.title, "New Title")
        self.assertEqual(len(call_args.settings.data), 1)
        self.assertEqual(call_args.settings.data[0].name, "file.csv")
        self.assertEqual(call_args.settings.data[0].description, "file desc")
        self.assertEqual(len(call_args.settings.data[0].columns), 2)
        self.assertEqual(call_args.settings.data[0].columns[0].name, "col1")
        self.assertEqual(call_args.settings.data[0].columns[0].description, "col1 desc")
        self.assertEqual(call_args.settings.data[0].columns[1].name, "col2")
        self.assertEqual(call_args.settings.data[0].columns[1].description, "col2 desc")  # title mapped to description

    def test_process_column_has_description_uses_description(self):
        col_dict = {"name": "col", "description": "desc", "type": "string"}
        processed = self.api.process_column(col_dict)
        self.assertEqual(processed.description, "desc")

    def test_process_column_has_title_only_falls_back_to_title(self):
        col_dict = {"name": "col", "title": "desc", "type": "string"}
        processed = self.api.process_column(col_dict)
        self.assertEqual(processed.description, "desc")

    def test_process_column_has_both_prefers_description(self):
        col_dict = {"name": "col", "description": "desc", "title": "ignored", "type": "string"}
        processed = self.api.process_column(col_dict)
        self.assertEqual(processed.description, "desc")

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_image_succeeds(self, mock_build, mock_upload_file):
        metadata = {"title": "New Title", "image": "cover.png"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        image_file = os.path.join(self.temp_dir, "cover.png")
        open(image_file, "w").close()

        mock_upload_file_result = MagicMock()
        mock_upload_file_result.token = "image-token-123"
        mock_upload_file.return_value = mock_upload_file_result

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        mock_upload_file.assert_called_once()
        self.assertEqual(mock_upload_file.call_args[0][0], "cover.png")
        self.assertEqual(mock_upload_file.call_args[0][1], image_file)

        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.assert_called_once()
        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        self.assertIsInstance(call_args, ApiUpdateDatasetMetadataRequest)
        self.assertEqual(call_args.settings.title, "New Title")
        self.assertIsNotNone(call_args.settings.image)
        self.assertEqual(call_args.settings.image.token, "image-token-123")
        self.assertEqual(len(call_args.settings.image.crop_rectangles), 2)
        self.assertEqual(call_args.settings.image.crop_rectangles[0].title, "cover image")
        self.assertEqual(call_args.settings.image.crop_rectangles[1].title, "thumbnail")

    def test_metadata_update_with_image_file_not_found_fails(self):
        metadata = {"image": "non-existent.png"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        with self.assertRaises(ValueError) as context:
            self.api.dataset_metadata_update("owner/dataset", self.temp_dir)
        self.assertIn("Image file was not found", str(context.exception))

    def test_metadata_update_with_image_invalid_extension_fails(self):
        metadata = {"image": "cover.txt"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        image_file = os.path.join(self.temp_dir, "cover.txt")
        open(image_file, "w").close()

        with self.assertRaises(ValueError) as context:
            self.api.dataset_metadata_update("owner/dataset", self.temp_dir)
        self.assertIn("Image file requires an extension of", str(context.exception))

    @patch.object(KaggleApi, "_upload_file", return_value=None)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_image_upload_failure_fails(self, mock_build, mock_upload_file):
        metadata = {"image": "cover.png"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        image_file = os.path.join(self.temp_dir, "cover.png")
        open(image_file, "w").close()

        with self.assertRaises(ValueError) as context:
            self.api.dataset_metadata_update("owner/dataset", self.temp_dir)
        self.assertIn("Error uploading image file", str(context.exception))


if __name__ == "__main__":
    unittest.main()
