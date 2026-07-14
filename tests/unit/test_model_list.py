# coding=utf-8
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi, FileList
from kagglesdk.models.types.model_enums import ModelFramework, ListModelsOrderBy
from kagglesdk.models.types.model_api_service import (
    ApiListModelsRequest,
    ApiListModelInstanceVersionFilesRequest,
    ApiListModelInstanceVersionFilesResponse,
)


class TestModelList(unittest.TestCase):
    """Tests for model_list, model_instance_files, and model_instance_version_files."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True

    # model_list tests
    def test_model_list_invalid_sort_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_list(sort_by="invalid-sort")
        self.assertIn("Invalid sort by specified", str(context.exception))

    def test_model_list_invalid_page_size_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_list(page_size=0)
        self.assertIn("Page size must be >= 1", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_list_defaults_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_model = MagicMock()
        mock_response.models = [mock_model]
        mock_response.next_page_token = "next-token-123"
        mock_kaggle.models.model_api_client.list_models.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.api.model_list(
                sort_by="voteCount", search="query", owner="owner", page_size=10, page_token="token-123"
            )

        self.assertEqual(result, [mock_model])
        self.assertIn("Next Page Token = next-token-123", f.getvalue())

        mock_kaggle.models.model_api_client.list_models.assert_called_once()
        request = mock_kaggle.models.model_api_client.list_models.call_args[0][0]
        self.assertEqual(request.sort_by, ListModelsOrderBy.LIST_MODELS_ORDER_BY_VOTE_COUNT)
        self.assertEqual(request.search, "query")
        self.assertEqual(request.owner, "owner")
        self.assertEqual(request.page_size, 10)
        self.assertEqual(request.page_token, "token-123")

    # model_instance_files tests
    def test_model_instance_files_missing_instance_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_files(None)
        self.assertIn("A model_instance must be specified", str(context.exception))

    def test_model_instance_files_invalid_format_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_files("owner/model/keras")
        self.assertIn("Model instance must be specified in the form of", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_instance_files_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = MagicMock(spec=ApiListModelInstanceVersionFilesResponse)
        mock_response.next_page_token = "next-file-token"

        mock_file = MagicMock()
        mock_file.name = "model.h5"
        mock_file.size = 1024
        mock_response.files = [mock_file]

        mock_kaggle.models.model_api_client.list_model_instance_version_files.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.api.model_instance_files("owner/model/keras/instance", page_token="token-456", page_size=50)

        self.assertIsInstance(result, FileList)
        self.assertIn("Next Page Token = next-file-token", f.getvalue())

        mock_kaggle.models.model_api_client.list_model_instance_version_files.assert_called_once()
        request = mock_kaggle.models.model_api_client.list_model_instance_version_files.call_args[0][0]
        self.assertEqual(request.owner_slug, "owner")
        self.assertEqual(request.model_slug, "model")
        self.assertEqual(request.framework, ModelFramework.MODEL_FRAMEWORK_KERAS)
        self.assertEqual(request.instance_slug, "instance")
        self.assertEqual(request.page_size, 50)
        self.assertEqual(request.page_token, "token-456")

    # model_instance_version_files tests
    def test_model_instance_version_files_missing_version_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_version_files(None)
        self.assertIn("A model_instance_version must be specified", str(context.exception))

    def test_model_instance_version_files_invalid_format_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.model_instance_version_files("owner/model/keras/instance")
        self.assertIn("Model instance version must be specified in the form of", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_instance_version_files_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = ApiListModelInstanceVersionFilesResponse()
        mock_response.next_page_token = "next-ver-file-token"
        mock_kaggle.models.model_api_client.list_model_instance_version_files.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.api.model_instance_version_files(
                "owner/model/keras/instance/3", page_token="token-789", page_size=5
            )

        self.assertEqual(result, mock_response)
        self.assertIn("Next Page Token = next-ver-file-token", f.getvalue())

        mock_kaggle.models.model_api_client.list_model_instance_version_files.assert_called_once()
        request = mock_kaggle.models.model_api_client.list_model_instance_version_files.call_args[0][0]
        self.assertEqual(request.owner_slug, "owner")
        self.assertEqual(request.model_slug, "model")
        self.assertEqual(request.framework, ModelFramework.MODEL_FRAMEWORK_KERAS)
        self.assertEqual(request.instance_slug, "instance")
        self.assertEqual(request.version_number, 3)
        self.assertEqual(request.page_size, 5)
        self.assertEqual(request.page_token, "token-789")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_instance_files_empty_returns_empty_list(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.list_model_instance_version_files.return_value = None
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.api.model_instance_files("owner/model/keras/instance")

        self.assertIsInstance(result, FileList)
        self.assertIn("No files found", f.getvalue())

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_model_instance_version_files_empty_returns_none(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.list_model_instance_version_files.return_value = None
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = self.api.model_instance_version_files("owner/model/keras/instance/3")

        self.assertIsNone(result)
        self.assertIn("No files found", f.getvalue())


if __name__ == "__main__":
    unittest.main()
