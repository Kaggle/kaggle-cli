# coding=utf-8
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_enums import ModelFramework, ModelInstanceType
from kagglesdk.models.types.model_api_service import (
    ApiCreateModelResponse,
    ApiCreateModelRequest,
    ApiUpdateModelRequest,
)
from kagglesdk.blobs.types.blob_api_service import ApiBlobType


class TestModelCreate(unittest.TestCase):
    """Tests for model_instance_create and model_instance_update."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True

    def _get_valid_instance_metadata(self):
        return {
            "ownerSlug": "testuser",
            "modelSlug": "test-model",
            "instanceSlug": "test-instance",
            "framework": "keras",
            "licenseName": "Apache 2.0",
        }

    def _write_metadata(self, folder, metadata, filename="model-instance-metadata.json"):
        meta_file_path = os.path.join(folder, filename)
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def test_model_instance_create_invalid_folder_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_create("/non/existent/folder")
        self.assertIn("Invalid folder", str(context.exception))

    def test_model_instance_create_default_owner_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["ownerSlug"] = "INSERT_OWNER_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_create(tmpdir)
            self.assertIn("Default ownerSlug detected", str(context.exception))

    def test_model_instance_create_default_model_slug_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["modelSlug"] = "INSERT_EXISTING_MODEL_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_create(tmpdir)
            self.assertIn("Default modelSlug detected", str(context.exception))

    def test_model_instance_create_default_instance_slug_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["instanceSlug"] = "INSERT_INSTANCE_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_create(tmpdir)
            self.assertIn("Default instanceSlug detected", str(context.exception))

    def test_model_instance_create_default_framework_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["framework"] = "INSERT_FRAMEWORK_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_create(tmpdir)
            self.assertIn("Default framework detected", str(context.exception))

    def test_model_instance_create_missing_license_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["licenseName"] = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_create(tmpdir)
            self.assertIn("Please specify a license", str(context.exception))

    def test_model_instance_create_invalid_fine_tunable_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["fineTunable"] = "not-a-bool"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_create(tmpdir)
            self.assertIn("modelInstance.fineTunable must be a boolean", str(context.exception))

    def test_model_instance_create_invalid_training_data_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["trainingData"] = "not-a-list"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_create(tmpdir)
            self.assertIn("modelInstance.trainingData must be a list", str(context.exception))

    @patch.object(KaggleApi, "upload_files")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_instance_create_valid_metadata_succeeds(self, mock_client, mock_upload):
        mock_kaggle = MagicMock()
        mock_response = ApiCreateModelResponse()
        mock_kaggle.models.model_api_client.create_model_instance.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_instance_metadata()
        metadata["overview"] = "My model overview"
        metadata["usage"] = "My model usage"
        metadata["fineTunable"] = True
        metadata["trainingData"] = ["dataset1", "dataset2"]
        metadata["modelInstanceType"] = "kaggleVariant"
        metadata["baseModelInstance"] = "base-instance"
        metadata["externalBaseModelUrl"] = "http://example.com"

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            response = self.api.model_instance_create(tmpdir, quiet=True, dir_mode="zip")

            self.assertEqual(response, mock_response)
            mock_upload.assert_called_once()
            call_args = mock_upload.call_args[0]
            self.assertEqual(call_args[2], tmpdir)
            self.assertEqual(call_args[3], ApiBlobType.MODEL)
            self.assertTrue(call_args[5])
            self.assertEqual(call_args[6], "zip")

            mock_kaggle.models.model_api_client.create_model_instance.assert_called_once()
            request = mock_kaggle.models.model_api_client.create_model_instance.call_args[0][0]
            self.assertEqual(request.owner_slug, "testuser")
            self.assertEqual(request.model_slug, "test-model")

            body = request.body
            self.assertEqual(body.framework, ModelFramework.MODEL_FRAMEWORK_KERAS)
            self.assertEqual(body.instance_slug, "test-instance")
            self.assertEqual(body.overview, "My model overview")
            self.assertEqual(body.usage, "My model usage")
            self.assertEqual(body.license_name, "Apache 2.0")
            self.assertTrue(body.fine_tunable)
            self.assertEqual(body.fine_tunable, True)
            self.assertEqual(body.training_data, ["dataset1", "dataset2"])
            self.assertEqual(body.model_instance_type, ModelInstanceType.MODEL_INSTANCE_TYPE_KAGGLE_VARIANT)
            self.assertEqual(body.base_model_instance, "base-instance")
            self.assertEqual(body.external_base_model_url, "http://example.com")

    def test_model_instance_update_invalid_folder_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_update("/non/existent/folder")
        self.assertIn("Invalid folder", str(context.exception))

    def test_model_instance_update_default_owner_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["ownerSlug"] = "INSERT_OWNER_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_update(tmpdir)
            self.assertIn("Default ownerSlug detected", str(context.exception))

    def test_model_instance_update_default_model_slug_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["modelSlug"] = "INSERT_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_update(tmpdir)
            self.assertIn("Default model slug detected", str(context.exception))

    def test_model_instance_update_default_instance_slug_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["instanceSlug"] = "INSERT_INSTANCE_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_update(tmpdir)
            self.assertIn("Default instance slug detected", str(context.exception))

    def test_model_instance_update_default_framework_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["framework"] = "INSERT_FRAMEWORK_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_update(tmpdir)
            self.assertIn("Default framework detected", str(context.exception))

    def test_model_instance_update_invalid_fine_tunable_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["fineTunable"] = "not-a-bool"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_update(tmpdir)
            self.assertIn("modelInstance.fineTunable must be a boolean", str(context.exception))

    def test_model_instance_update_invalid_training_data_fails(self):
        metadata = self._get_valid_instance_metadata()
        metadata["trainingData"] = "not-a-list"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.model_instance_update(tmpdir)
            self.assertIn("modelInstance.trainingData must be a list", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_instance_update_all_fields_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.models.model_api_client.update_model_instance.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_instance_metadata()
        metadata["overview"] = "Updated overview"
        metadata["usage"] = "Updated usage"
        metadata["licenseName"] = "MIT"
        metadata["fineTunable"] = True
        metadata["trainingData"] = ["dataset3"]
        metadata["modelInstanceType"] = "kaggleVariant"
        metadata["baseModelInstance"] = "new-base"
        metadata["externalBaseModelUrl"] = "http://new-example.com"

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            response = self.api.model_instance_update(tmpdir)

            self.assertEqual(response, mock_response)
            mock_kaggle.models.model_api_client.update_model_instance.assert_called_once()
            request = mock_kaggle.models.model_api_client.update_model_instance.call_args[0][0]

            self.assertEqual(request.owner_slug, "testuser")
            self.assertEqual(request.model_slug, "test-model")
            self.assertEqual(request.framework, ModelFramework.MODEL_FRAMEWORK_KERAS)
            self.assertEqual(request.instance_slug, "test-instance")
            self.assertEqual(request.overview, "Updated overview")
            self.assertEqual(request.usage, "Updated usage")
            self.assertEqual(request.license_name, "MIT")
            self.assertEqual(request.fine_tunable, True)
            self.assertEqual(request.training_data, ["dataset3"])
            self.assertEqual(request.model_instance_type, ModelInstanceType.MODEL_INSTANCE_TYPE_KAGGLE_VARIANT)
            self.assertEqual(request.base_model_instance, "new-base")
            self.assertEqual(request.external_base_model_url, "http://new-example.com")

            self.assertIsNotNone(request.update_mask)
            self.assertEqual(
                list(request.update_mask.paths),
                [
                    "overview",
                    "usage",
                    "license_name",
                    "fine_tunable",
                    "training_data",
                    "model_instance_type",
                    "base_model_instance",
                    "external_base_model_url",
                ],
            )

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_instance_update_partial_fields_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.update_model_instance.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_instance_metadata()
        del metadata["licenseName"]
        metadata["overview"] = "Updated overview"
        metadata["usage"] = "Updated usage"

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata)
            self.api.model_instance_update(tmpdir)

            request = mock_kaggle.models.model_api_client.update_model_instance.call_args[0][0]

            self.assertEqual(request.overview, "Updated overview")
            self.assertEqual(request.usage, "Updated usage")
            self.assertFalse(request.fine_tunable)
            self.assertEqual(request.license_name, "Apache 2.0")

            self.assertIsNotNone(request.update_mask)
            self.assertEqual(list(request.update_mask.paths), ["overview", "usage"])

    def _get_valid_model_metadata(self):
        return {
            "ownerSlug": "testuser",
            "slug": "test-model",
            "title": "Test Model Title",
            "isPrivate": True,
            "description": "Test model description",
        }

    def test_model_create_new_invalid_folder_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_create_new("/non/existent/folder")
        self.assertIn("Invalid folder", str(context.exception))

    def test_model_create_new_default_owner_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["ownerSlug"] = "INSERT_OWNER_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_create_new(tmpdir)
            self.assertIn("Default ownerSlug detected", str(context.exception))

    def test_model_create_new_default_title_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["title"] = "INSERT_TITLE_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_create_new(tmpdir)
            self.assertIn("Default title detected", str(context.exception))

    def test_model_create_new_default_slug_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["slug"] = "INSERT_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_create_new(tmpdir)
            self.assertIn("Default slug detected", str(context.exception))

    def test_model_create_new_invalid_is_private_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["isPrivate"] = "not-a-bool"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_create_new(tmpdir)
            self.assertIn("model.isPrivate must be a boolean", str(context.exception))

    def test_model_create_new_invalid_publish_time_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["publishTime"] = "invalid-date"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_create_new(tmpdir)
            self.assertIn("does not match format", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_create_new_valid_metadata_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = ApiCreateModelResponse()
        mock_kaggle.models.model_api_client.create_model.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_model_metadata()
        metadata["subtitle"] = "Valid subtitle"
        metadata["provenanceSources"] = "source1"
        # publishTime is omitted due to Bug, see Task 5.3

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            response = self.api.model_create_new(tmpdir)

            self.assertEqual(response, mock_response)
            mock_kaggle.models.model_api_client.create_model.assert_called_once()
            request = mock_kaggle.models.model_api_client.create_model.call_args[0][0]
            self.assertEqual(request.owner_slug, "testuser")
            self.assertEqual(request.slug, "test-model")
            self.assertEqual(request.title, "Test Model Title")
            self.assertEqual(request.subtitle, "Valid subtitle")
            self.assertTrue(request.is_private)
            self.assertEqual(request.description, "Test model description")
            self.assertIsNone(request.publish_time)
            self.assertEqual(request.provenance_sources, "source1")

    def test_model_update_invalid_folder_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_update("/non/existent/folder")
        self.assertIn("Invalid folder", str(context.exception))

    def test_model_update_default_owner_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["ownerSlug"] = "INSERT_OWNER_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_update(tmpdir)
            self.assertIn("Default ownerSlug detected", str(context.exception))

    def test_model_update_default_slug_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["slug"] = "INSERT_SLUG_HERE"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_update(tmpdir)
            self.assertIn("Default slug detected", str(context.exception))

    def test_model_update_invalid_is_private_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["isPrivate"] = "not-a-bool"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_update(tmpdir)
            self.assertIn("model.isPrivate must be a boolean", str(context.exception))

    def test_model_update_invalid_publish_time_fails(self):
        metadata = self._get_valid_model_metadata()
        metadata["publishTime"] = "invalid-date"
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            with self.assertRaises(ValueError) as context:
                self.api.model_update(tmpdir)
            self.assertIn("does not match format", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_update_all_fields_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.models.model_api_client.update_model.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = self._get_valid_model_metadata()
        metadata["title"] = "Updated Model Title"
        metadata["subtitle"] = "Updated subtitle"
        metadata["isPrivate"] = False
        metadata["description"] = "Updated description"
        # publishTime and provenanceSources are omitted to avoid bugs, see Tasks 5.3, 5.4

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            response = self.api.model_update(tmpdir)

            self.assertEqual(response, mock_response)
            mock_kaggle.models.model_api_client.update_model.assert_called_once()
            request = mock_kaggle.models.model_api_client.update_model.call_args[0][0]

            self.assertEqual(request.owner_slug, "testuser")
            self.assertEqual(request.model_slug, "test-model")
            self.assertEqual(request.title, "Updated Model Title")
            self.assertEqual(request.subtitle, "Updated subtitle")
            self.assertFalse(request.is_private)
            self.assertEqual(request.description, "Updated description")
            self.assertIsNone(request.publish_time)
            self.assertEqual(request.provenance_sources, "")

            self.assertIsNotNone(request.update_mask)
            self.assertEqual(
                list(request.update_mask.paths),
                ["title", "subtitle", "is_private", "description"],
            )

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_update_partial_fields_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.update_model.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata = {
            "ownerSlug": "testuser",
            "slug": "test-model",
            "title": "Updated Title",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata, "model-metadata.json")
            self.api.model_update(tmpdir)

            request = mock_kaggle.models.model_api_client.update_model.call_args[0][0]

            self.assertEqual(request.title, "Updated Title")
            self.assertEqual(request.subtitle, "")
            self.assertTrue(request.is_private)
            self.assertEqual(request.description, "")

            self.assertIsNotNone(request.update_mask)
            self.assertEqual(list(request.update_mask.paths), ["title"])


if __name__ == "__main__":
    unittest.main()
