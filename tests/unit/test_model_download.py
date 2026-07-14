# coding=utf-8
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import tarfile
import io
import sys

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_enums import ModelFramework
from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest


class TestModelDownload(unittest.TestCase):
    """Tests for model_instance_version_download."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_download_missing_version_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_version_download(None)
        self.assertIn("A model_instance_version must be specified", str(context.exception))

    def test_download_invalid_format_fails(self):
        # Too few slashes
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_version_download("owner/model/keras/instance")
        self.assertIn("Model instance version must be specified in the form", str(context.exception))

    def test_download_empty_parts_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_version_download("owner/model/keras//1")
        self.assertIn("Invalid model instance version specification", str(context.exception))

    def test_download_non_int_version_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_version_download("owner/model/keras/instance/abc")
        self.assertIn("version-number must be an integer", str(context.exception))

    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_download_cached_skips_download(self, mock_client, mock_download_needed):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.models.model_api_client.download_model_instance_version.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        version_str = "owner/model/keras/instance/2"
        outfile = self.api.model_instance_version_download(version_str, path=self.temp_dir)

        expected_outfile = os.path.join(self.temp_dir, "model.tar.gz")
        self.assertEqual(outfile, expected_outfile)

        mock_kaggle.models.model_api_client.download_model_instance_version.assert_called_once()
        request = mock_kaggle.models.model_api_client.download_model_instance_version.call_args[0][0]
        self.assertEqual(request.owner_slug, "owner")
        self.assertEqual(request.model_slug, "model")
        self.assertEqual(request.framework, ModelFramework.MODEL_FRAMEWORK_KERAS)
        self.assertEqual(request.instance_slug, "instance")
        self.assertEqual(request.version_number, 2)

        mock_download_needed.assert_called_once_with(mock_response, expected_outfile, True)

    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_download_needed_downloads(self, mock_client, mock_download_needed, mock_download_file):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.models.model_api_client.download_model_instance_version.return_value = mock_response
        mock_kaggle.http_client.return_value = "mock-http-client"
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        version_str = "owner/model/keras/instance/2"
        outfile = self.api.model_instance_version_download(version_str, path=self.temp_dir, quiet=False)

        expected_outfile = os.path.join(self.temp_dir, "model.tar.gz")
        self.assertEqual(outfile, expected_outfile)

        mock_download_needed.assert_called_once_with(mock_response, expected_outfile, False)
        mock_download_file.assert_called_once_with(mock_response, expected_outfile, "mock-http-client", False, True)

    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_download_force_downloads(self, mock_client, mock_download_needed, mock_download_file):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_kaggle.models.model_api_client.download_model_instance_version.return_value = mock_response
        mock_kaggle.http_client.return_value = "mock-http-client"
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        version_str = "owner/model/keras/instance/2"
        outfile = self.api.model_instance_version_download(version_str, path=self.temp_dir, force=True)

        expected_outfile = os.path.join(self.temp_dir, "model.tar.gz")
        self.assertEqual(outfile, expected_outfile)

        mock_download_needed.assert_not_called()
        mock_download_file.assert_called_once_with(mock_response, expected_outfile, "mock-http-client", True, False)

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_download_untar_succeeds(self, mock_client, mock_download_needed):
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            file_data = b"dummy content"
            tarinfo = tarfile.TarInfo(name="dummy.txt")
            tarinfo.size = len(file_data)
            tar.addfile(tarinfo, io.BytesIO(file_data))
        tar_bytes = tar_buffer.getvalue()

        def side_effect_download(response, outfile, http_client, quiet, show_progress):
            with open(outfile, "wb") as f:
                f.write(tar_bytes)

        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.download_model_instance_version.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        version_str = "owner/model/keras/instance/2"

        with patch.object(KaggleApi, "download_file", side_effect=side_effect_download):
            outfile = self.api.model_instance_version_download(version_str, path=self.temp_dir, untar=True)

        self.assertFalse(os.path.exists(outfile))
        extracted_file = os.path.join(self.temp_dir, "dummy.txt")
        self.assertTrue(os.path.exists(extracted_file))
        with open(extracted_file, "rb") as f:
            self.assertEqual(f.read(), b"dummy content")

    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_download_untar_failure_fails(self, mock_client, mock_download_needed, mock_download_file):
        def side_effect_download(response, outfile, http_client, quiet, show_progress):
            open(outfile, "w").close()

        mock_download_file.side_effect = side_effect_download

        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.download_model_instance_version.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        version_str = "owner/model/keras/instance/2"

        with self.assertRaises(ValueError) as context:
            self.api.model_instance_version_download(version_str, path=self.temp_dir, untar=True)
        self.assertIn("Error extracting the tar.gz file", str(context.exception))

        expected_outfile = os.path.join(self.temp_dir, "model.tar.gz")
        self.assertTrue(os.path.exists(expected_outfile))

    @patch.object(KaggleApi, "get_default_download_dir")
    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_download_default_path_succeeds(self, mock_client, mock_download_needed, mock_get_dir):
        mock_get_dir.return_value = self.temp_dir
        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.download_model_instance_version.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        version_str = "owner/model/keras/instance/2"
        outfile = self.api.model_instance_version_download(version_str, path=None)

        expected_outfile = os.path.join(self.temp_dir, "model.tar.gz")
        self.assertEqual(outfile, expected_outfile)
        mock_get_dir.assert_called_once_with("models", "owner", "model", "keras", "instance", "2")


if __name__ == "__main__":
    unittest.main()
