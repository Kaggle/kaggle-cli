# coding=utf-8
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, "../../src")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestKernelsPull(unittest.TestCase):

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_without_version(self, mock_client, mock_open_file, mock_isfile, mock_exists):
        # Setup mocks
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_blob = MagicMock()
        mock_blob.language = "python"
        mock_blob.kernel_type = "script"
        mock_blob.slug = "my-slug"
        mock_blob.source = "print('hello')"
        mock_response.blob = mock_blob

        mock_kaggle.kernels.kernels_api_client.get_kernel.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        # Call method
        self.api.kernels_pull("owner/my-slug", path="/tmp/dummy")

        # Verify request
        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug")

        # Verify file write
        mock_open_file.assert_called_once_with("/tmp/dummy/my-slug.py", "w", encoding="utf-8")
        mock_open_file().write.assert_called_once_with("print('hello')")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_with_version(self, mock_client, mock_open_file, mock_isfile, mock_exists):
        # Setup mocks
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_blob = MagicMock()
        mock_blob.language = "python"
        mock_blob.kernel_type = "script"
        mock_blob.slug = "my-slug"
        mock_blob.source = "print('hello')"
        mock_response.blob = mock_blob

        mock_kaggle.kernels.kernels_api_client.get_kernel.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        # Call method with version
        self.api.kernels_pull("owner/my-slug/3", path="/tmp/dummy")

        # Verify request has version in slug
        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug/3")

        # Verify file write (path should still use clean slug)
        mock_open_file.assert_called_once_with("/tmp/dummy/my-slug.py", "w", encoding="utf-8")
        mock_open_file().write.assert_called_once_with("print('hello')")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_with_metadata(self, mock_client, mock_open_file, mock_isfile, mock_exists):
        # Setup mocks
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_blob = MagicMock()
        mock_blob.language = "python"
        mock_blob.kernel_type = "script"
        mock_blob.slug = "my-slug"
        mock_blob.source = "print('hello')"
        mock_response.blob = mock_blob

        mock_metadata = MagicMock()
        mock_metadata.ref = "owner/my-slug"
        mock_metadata.id = 123
        mock_metadata.title = "My Title"
        mock_metadata.language = "python"
        mock_metadata.kernel_type = "script"
        mock_metadata.is_private = True
        mock_metadata.enable_gpu = False
        mock_metadata.enable_tpu = False
        mock_metadata.enable_internet = True
        mock_metadata.category_ids = []
        mock_metadata.dataset_data_sources = []
        mock_metadata.kernel_data_sources = []
        mock_metadata.competition_data_sources = []
        mock_metadata.model_data_sources = []
        mock_metadata.docker_image = "some-image"
        mock_metadata.machine_shape = "NvidiaTeslaT4"
        mock_response.metadata = mock_metadata

        mock_kaggle.kernels.kernels_api_client.get_kernel.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        # Call method with metadata=True
        self.api.kernels_pull("owner/my-slug/3", path="/tmp/dummy", metadata=True)

        # Verify request has version in slug
        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug/3")

        # Verify open was called twice (one for script, one for metadata)
        self.assertEqual(mock_open_file.call_count, 2)

        # We can check the calls
        # First call: script
        mock_open_file.assert_any_call("/tmp/dummy/my-slug.py", "w", encoding="utf-8")
        # Second call: metadata
        mock_open_file.assert_any_call("/tmp/dummy/kernel-metadata.json", "w")


if __name__ == "__main__":
    unittest.main()
