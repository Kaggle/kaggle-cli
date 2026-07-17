# coding=utf-8
import json
import os
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from kaggle.api.kaggle_api_extended import KaggleApi


class TestKernelsPush(unittest.TestCase):

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.valid_push_language_types = ["python", "r", "julia", "rmarkdown"]
        self.api.valid_push_kernel_types = ["script", "notebook"]
        self.api.valid_push_pinning_types = ["original", "latest"]
        self.api.KERNEL_METADATA_FILE = "kernel-metadata.json"

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_push_utf8_behavior(self, mock_client):
        # Setup mock client response
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.invalidTags = []
        mock_response.invalidDatasetSources = []
        mock_response.invalidCompetitionSources = []
        mock_response.invalidKernelSources = []
        mock_response.versionNumber = 1
        mock_response.url = "https://www.kaggle.com/testuser/test-kernel"
        mock_kaggle.kernels.kernels_api_client.save_kernel.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        # Content with non-ASCII characters to trigger UnicodeDecodeError if opened as ANSI
        unicode_content = (
            'print("\U0001f44b")\n'
            'print("R\u00e9sum\u00e9")\n'
            'print("\u3053\u3093\u306b\u3061\u306f")\n'
            'print("\u0928\u092e\u0938\u094d\u0924\u0947")\n'
        )

        metadata_dict = {
            "id": "testuser/test-kernel",
            "title": "Test Kernel Title",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }

        # Create actual temporary directory and files to test real filesystem behavior
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file_path = os.path.join(tmpdir, "kernel-metadata.json")
            code_file_path = os.path.join(tmpdir, "script.py")

            # Write UTF-8 metadata (which can also contain unicode characters in title)
            metadata_dict["title"] = "Test R\u00e9sum\u00e9 Title \U0001f44b"
            with open(meta_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f)

            # Write UTF-8 script code
            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(unicode_content)

            # Run kernels_push on the temporary directory, forcing preferred encoding to be cp1252
            # to verify that explicit utf-8 encoding is used regardless of the system locale.
            try:
                with patch("locale.getpreferredencoding", return_value="cp1252"):
                    self.api.kernels_push(tmpdir)
            except Exception as e:
                self.fail(f"kernels_push raised an unexpected exception: {e}")

            # Verify that save_kernel was called with the correctly read request body
            mock_kaggle.kernels.kernels_api_client.save_kernel.assert_called_once()
            call_args = mock_kaggle.kernels.kernels_api_client.save_kernel.call_args
            request = call_args[0][0]

            self.assertEqual(request.new_title, "Test R\u00e9sum\u00e9 Title \U0001f44b")
            self.assertEqual(request.text, unicode_content)

    def _get_valid_metadata(self):
        return {
            "id": "testuser/test-kernel",
            "title": "Test Kernel Title",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
        }

    def _write_metadata(self, folder, metadata):
        meta_file_path = os.path.join(folder, self.api.KERNEL_METADATA_FILE)
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def test_kernels_push_invalid_folder(self):
        with self.assertRaises(ValueError) as context:
            self.api.kernels_push("/non/existent/folder")
        self.assertIn("Invalid folder", str(context.exception))

    def test_kernels_push_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("Metadata file not found", str(context.exception))

    def test_kernels_push_title_too_short(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            metadata["title"] = "Tiny"
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("Title must be at least five characters", str(context.exception))

    def test_kernels_push_missing_code_file_in_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            metadata["code_file"] = ""
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("A source file must be specified", str(context.exception))

    def test_kernels_push_code_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            # script.py is specified but we don't create it
            self._write_metadata(tmpdir, metadata)
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("Source file not found", str(context.exception))

    def test_kernels_push_missing_id_and_id_no(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            del metadata["id"]
            if "id_no" in metadata:
                del metadata["id_no"]
            self._write_metadata(tmpdir, metadata)
            # Create the code file so we get past that check
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("ID or slug must be specified", str(context.exception))

    def test_kernels_push_slug_contains_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            metadata["id"] = "testuser/test-kernel/3"
            self._write_metadata(tmpdir, metadata)
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("cannot contain a version", str(context.exception))

    def test_kernels_push_invalid_language(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            metadata["language"] = "invalid-lang"
            self._write_metadata(tmpdir, metadata)
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("A valid language must be specified", str(context.exception))

    def test_kernels_push_invalid_kernel_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            metadata["kernel_type"] = "invalid-type"
            self._write_metadata(tmpdir, metadata)
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("A valid kernel type must be specified", str(context.exception))

    def test_kernels_push_invalid_docker_pinning_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = self._get_valid_metadata()
            metadata["docker_image_pinning_type"] = "invalid-pinning"
            self._write_metadata(tmpdir, metadata)
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("docker_image_pinning_type must be", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_push_notebook(self, mock_client):
        # Setup mock client response
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.invalidTags = []
        mock_response.invalidDatasetSources = []
        mock_response.invalidCompetitionSources = []
        mock_response.invalidKernelSources = []
        mock_response.versionNumber = 1
        mock_response.url = "https://www.kaggle.com/testuser/test-kernel"
        mock_kaggle.kernels.kernels_api_client.save_kernel.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello')\n", "print('world')"],
                    "outputs": [{"output_type": "stream", "text": "hello\nworld"}],
                },
                {"cell_type": "markdown", "source": ["# Heading"]},
            ]
        }

        metadata_dict = self._get_valid_metadata()
        metadata_dict["code_file"] = "notebook.ipynb"
        metadata_dict["kernel_type"] = "notebook"

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file_path = os.path.join(tmpdir, "kernel-metadata.json")
            code_file_path = os.path.join(tmpdir, "notebook.ipynb")

            with open(meta_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f)

            with open(code_file_path, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f)

            self.api.kernels_push(tmpdir)

            # Verify save_kernel call
            mock_kaggle.kernels.kernels_api_client.save_kernel.assert_called_once()
            request = mock_kaggle.kernels.kernels_api_client.save_kernel.call_args[0][0]

            # Verify pushed text is cleaned JSON
            pushed_json = json.loads(request.text)

            # Verify outputs are cleared for code cell
            self.assertEqual(pushed_json["cells"][0]["outputs"], [])
            # Verify source is joined for code cell
            self.assertEqual(pushed_json["cells"][0]["source"], "print('hello')\nprint('world')")
            # Markdown cell source should also be joined if it was a list (it was ["# Heading"] -> "# Heading")
            # Wait, the code only joins if "source" is in cell and isinstance(source, list).
            # Yes, it should join for markdown too if it is a list.
            self.assertEqual(pushed_json["cells"][1]["source"], "# Heading")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_push_notebook_rmarkdown(self, mock_client):
        # Setup mock client
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.invalidTags = []
        mock_response.invalidDatasetSources = []
        mock_response.invalidCompetitionSources = []
        mock_response.invalidKernelSources = []
        mock_response.versionNumber = 1
        mock_response.url = "https://www.kaggle.com/testuser/test-kernel"
        mock_kaggle.kernels.kernels_api_client.save_kernel.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        notebook_content: dict[str, list[Any]] = {"cells": []}
        metadata_dict = self._get_valid_metadata()
        metadata_dict["code_file"] = "notebook.ipynb"
        metadata_dict["kernel_type"] = "notebook"
        metadata_dict["language"] = "rmarkdown"

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata_dict)
            with open(os.path.join(tmpdir, "notebook.ipynb"), "w") as f:
                json.dump(notebook_content, f)

            self.api.kernels_push(tmpdir)

            # Verify save_kernel call has language='r' (converted from rmarkdown for notebooks)
            mock_kaggle.kernels.kernels_api_client.save_kernel.assert_called_once()
            request = mock_kaggle.kernels.kernels_api_client.save_kernel.call_args[0][0]
            self.assertEqual(request.language, "r")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_push_with_sources(self, mock_client):
        # Setup mock client
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.invalidTags = []
        mock_response.invalidDatasetSources = []
        mock_response.invalidCompetitionSources = []
        mock_response.invalidKernelSources = []
        mock_response.versionNumber = 1
        mock_response.url = "https://www.kaggle.com/testuser/test-kernel"
        mock_kaggle.kernels.kernels_api_client.save_kernel.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        metadata_dict = self._get_valid_metadata()
        metadata_dict["dataset_sources"] = ["owner/dataset-slug", "owner2/dataset-slug2/3"]
        metadata_dict["kernel_sources"] = ["owner/kernel-slug", "owner2/kernel-slug2/1"]
        metadata_dict["model_sources"] = ["owner/model/framework/instance/1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata_dict)
            open(os.path.join(tmpdir, "script.py"), "w").close()

            self.api.kernels_push(tmpdir)

            mock_kaggle.kernels.kernels_api_client.save_kernel.assert_called_once()
            request = mock_kaggle.kernels.kernels_api_client.save_kernel.call_args[0][0]
            self.assertEqual(request.dataset_data_sources, ["owner/dataset-slug", "owner2/dataset-slug2/3"])
            self.assertEqual(request.kernel_data_sources, ["owner/kernel-slug", "owner2/kernel-slug2/1"])
            self.assertEqual(request.model_data_sources, ["owner/model/framework/instance/1"])

    def test_kernels_push_invalid_dataset_source(self):
        metadata_dict = self._get_valid_metadata()
        metadata_dict["dataset_sources"] = ["invalid_source_no_slash"]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata_dict)
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("Dataset must be specified in the form", str(context.exception))

    def test_kernels_push_invalid_kernel_source(self):
        metadata_dict = self._get_valid_metadata()
        metadata_dict["kernel_sources"] = ["invalid_source_no_slash"]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata_dict)
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("Kernel must be specified in the form", str(context.exception))

    def test_kernels_push_invalid_model_source(self):
        metadata_dict = self._get_valid_metadata()
        metadata_dict["model_sources"] = ["invalid/model/source/too/few/slashes"]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_metadata(tmpdir, metadata_dict)
            open(os.path.join(tmpdir, "script.py"), "w").close()
            with self.assertRaises(ValueError) as context:
                self.api.kernels_push(tmpdir)
            self.assertIn("Model instance version must be specified in the form", str(context.exception))


if __name__ == "__main__":
    unittest.main()
