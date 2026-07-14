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
        self.api.kernels_pull("owner/my-slug", path=os.path.join("/tmp", "dummy"))

        # Verify request
        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug")

        # Verify file write
        expected_path = os.path.join("/tmp", "dummy", "my-slug.py")
        mock_open_file.assert_called_once_with(expected_path, "w", encoding="utf-8")
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
        self.api.kernels_pull("owner/my-slug/3", path=os.path.join("/tmp", "dummy"))

        # Verify request has version in slug
        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug/3")

        # Verify file write (path should still use clean slug)
        expected_path = os.path.join("/tmp", "dummy", "my-slug.py")
        mock_open_file.assert_called_once_with(expected_path, "w", encoding="utf-8")
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
        self.api.kernels_pull("owner/my-slug/3", path=os.path.join("/tmp", "dummy"), metadata=True)

        # Verify request has version in slug
        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug/3")

        # Verify open was called twice (one for script, one for metadata)
        self.assertEqual(mock_open_file.call_count, 2)

        # We can check the calls
        # First call: script
        expected_script_path = os.path.join("/tmp", "dummy", "my-slug.py")
        mock_open_file.assert_any_call(expected_script_path, "w", encoding="utf-8")
        # Second call: metadata
        expected_metadata_path = os.path.join("/tmp", "dummy", "kernel-metadata.json")
        mock_open_file.assert_any_call(expected_metadata_path, "w")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_fallback_scenarios(self, mock_client, mock_open_file, mock_isfile, mock_exists):
        scenarios = [
            ("unsupported-lang", "script"),
            ("", "script"),
            ("python", ""),
            ("python", "something-new"),
        ]
        for lang, kt in scenarios:
            # Setup mocks
            mock_kaggle = MagicMock()
            mock_response = MagicMock()
            mock_blob = MagicMock()
            mock_blob.language = lang
            mock_blob.kernel_type = kt
            mock_blob.slug = "my-slug"
            mock_blob.source = "print('hello')"
            mock_response.blob = mock_blob

            mock_kaggle.kernels.kernels_api_client.get_kernel.return_value = mock_response
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)

            mock_open_file.reset_mock()
            # Call method
            self.api.kernels_pull("owner/my-slug", path=os.path.join("/tmp", "dummy"))

            # Verify file write uses script.py
            expected_path = os.path.join("/tmp", "dummy", "script.py")
            mock_open_file.assert_called_once_with(expected_path, "w", encoding="utf-8")
            mock_open_file().write.assert_called_once_with("print('hello')")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_extensions(self, mock_client, mock_open_file, mock_isfile, mock_exists):
        scenarios = [
            ("python", "script", ".py"),
            ("r", "script", ".R"),
            ("rmarkdown", "script", ".Rmd"),
            ("sqlite", "script", ".sql"),
            ("julia", "script", ".jl"),
            ("python", "notebook", ".ipynb"),
            ("r", "notebook", ".irnb"),
            ("julia", "notebook", ".ijlnb"),
        ]
        for lang, kt, expected_ext in scenarios:
            # Setup mocks
            mock_kaggle = MagicMock()
            mock_response = MagicMock()
            mock_blob = MagicMock()
            mock_blob.language = lang
            mock_blob.kernel_type = kt
            mock_blob.slug = "my-slug"
            mock_blob.source = "some-source"
            mock_response.blob = mock_blob

            mock_kaggle.kernels.kernels_api_client.get_kernel.return_value = mock_response
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)

            mock_open_file.reset_mock()
            # Call method
            self.api.kernels_pull("owner/my-slug", path=os.path.join("/tmp", "dummy"))

            # Verify file write uses expected extension
            expected_path = os.path.join("/tmp", "dummy", f"my-slug{expected_ext}")
            mock_open_file.assert_called_once_with(expected_path, "w", encoding="utf-8")
            mock_open_file().write.assert_called_once_with("some-source")

    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("builtins.open")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_no_kernel_metadata_exists(self, mock_client, mock_open_file, mock_isfile, mock_exists):
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

        metadata_path = os.path.normpath("/tmp/dummy/kernel-metadata.json")
        effective_path = os.path.normpath("/tmp/dummy")

        def exists_side_effect(p):
            p_norm = os.path.normpath(p)
            return p_norm == metadata_path or p_norm == effective_path

        mock_exists.side_effect = exists_side_effect
        mock_isfile.return_value = False

        metadata_json = '{"id": "owner/my-slug", "code_file": "script.py"}'
        m = mock_open(read_data=metadata_json)
        mock_open_file.side_effect = m

        self.api.kernels_pull(None, path="/tmp/dummy")

        mock_open_file.assert_any_call(metadata_path)

        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug")

        expected_path = os.path.normpath("/tmp/dummy/script.py")
        mock_open_file.assert_any_call(expected_path, "w", encoding="utf-8")

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_kernels_pull_no_kernel_metadata_placeholder(self, mock_open_file, mock_exists):
        mock_exists.return_value = True
        metadata_json = '{"id": "owner/INSERT_KERNEL_SLUG_HERE", "code_file": "script.py"}'
        mock_open_file.side_effect = mock_open(read_data=metadata_json)

        with self.assertRaises(ValueError) as context:
            self.api.kernels_pull(None, path="/tmp/dummy")
        self.assertIn("A kernel must be specified", str(context.exception))

    @patch("os.getcwd", return_value="/tmp/cwd")
    @patch("os.path.exists")
    @patch("os.path.isfile")
    @patch("builtins.open")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_no_kernel_no_path_metadata_exists(
        self, mock_client, mock_open_file, mock_isfile, mock_exists, mock_getcwd
    ):
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

        metadata_path = os.path.normpath("/tmp/cwd/kernel-metadata.json")
        effective_path = os.path.normpath("/tmp/cwd")

        def exists_side_effect(p):
            p_norm = os.path.normpath(p)
            return p_norm == metadata_path or p_norm == effective_path

        mock_exists.side_effect = exists_side_effect
        mock_isfile.return_value = False

        metadata_json = '{"id": "owner/my-slug", "code_file": "script.py"}'
        mock_open_file.side_effect = mock_open(read_data=metadata_json)

        self.api.kernels_pull(None, None)

        mock_open_file.assert_any_call(metadata_path)

        call_args = mock_kaggle.kernels.kernels_api_client.get_kernel.call_args
        request = call_args[0][0]
        self.assertEqual(request.user_name, "owner")
        self.assertEqual(request.kernel_slug, "my-slug")

        expected_path = os.path.normpath("/tmp/cwd/script.py")
        mock_open_file.assert_any_call(expected_path, "w", encoding="utf-8")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(KaggleApi, "get_default_download_dir", return_value="/tmp/default_dir")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_pull_no_path(self, mock_client, mock_get_dir, mock_open_file, mock_isfile, mock_exists):
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

        self.api.kernels_pull("owner/my-slug", None)

        mock_get_dir.assert_called_once_with("kernels", "owner", "my-slug")

        expected_path = os.path.normpath("/tmp/default_dir/my-slug.py")
        mock_open_file.assert_called_once_with(expected_path, "w", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
