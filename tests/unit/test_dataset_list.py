# coding=utf-8
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_enums import (
    DatasetSelectionGroup,
    DatasetSortBy,
    DatasetFileTypeGroup,
    DatasetLicenseGroup,
)
from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetsResponse


class TestDatasetList(unittest.TestCase):
    """Tests for dataset_list and dataset_list_with_response."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.already_printed_version_warning = True

    def test_dataset_list_with_response_invalid_sort_by_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(sort_by="invalid")
        self.assertIn("Invalid sort by specified", str(context.exception))

    def test_dataset_list_with_response_deprecated_size_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(size="small")
        self.assertIn("The --size parameter has been deprecated", str(context.exception))

    def test_dataset_list_with_response_invalid_file_type_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(file_type="invalid")
        self.assertIn("Invalid file type specified", str(context.exception))

    def test_dataset_list_with_response_invalid_license_name_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(license_name="invalid")
        self.assertIn("Invalid license specified", str(context.exception))

    def test_dataset_list_with_response_invalid_page_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(page=-1)
        self.assertIn("Page number must be >= 1", str(context.exception))

    def test_dataset_list_with_response_invalid_size_range_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(max_size="100", min_size="200")
        self.assertIn("Max Size must be max_size >= min_size", str(context.exception))

    def test_dataset_list_with_response_invalid_max_size_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(max_size="0")
        self.assertIn("Max Size must be > 0", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(max_size="-10")
        self.assertIn("Max Size must be > 0", str(context.exception))

    def test_dataset_list_with_response_invalid_min_size_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(min_size="-5")
        self.assertIn("Min Size must be >= 0", str(context.exception))

    def test_dataset_list_with_response_mine_and_user_conflict_fails(self):
        with self.assertRaises(ValueError) as context:
            self.api.dataset_list_with_response(mine=True, user="someuser")
        self.assertIn("Cannot specify both mine and a user", str(context.exception))

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_list_with_response_defaults_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = ApiListDatasetsResponse()
        mock_kaggle.datasets.dataset_api_client.list_datasets.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        response = self.api.dataset_list_with_response()

        mock_kaggle.datasets.dataset_api_client.list_datasets.assert_called_once()
        request = mock_kaggle.datasets.dataset_api_client.list_datasets.call_args[0][0]

        self.assertEqual(request.group, DatasetSelectionGroup.DATASET_SELECTION_GROUP_PUBLIC)
        self.assertEqual(request.sort_by, DatasetSortBy.DATASET_SORT_BY_HOTTEST)
        self.assertEqual(request.file_type, DatasetFileTypeGroup.DATASET_FILE_TYPE_GROUP_ALL)
        self.assertEqual(request.license, DatasetLicenseGroup.DATASET_LICENSE_GROUP_ALL)
        self.assertEqual(request.tag_ids, "")
        self.assertEqual(request.search, "")
        self.assertEqual(request.user, "")
        self.assertEqual(request.page, 1)
        self.assertEqual(request.page_size, 20)
        # Proto3 defaults to 0 for unset int fields when accessed in Python
        self.assertEqual(request.max_size, 0)
        self.assertEqual(request.min_size, 0)
        self.assertEqual(response, mock_response)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_list_with_response_custom_filters_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_response = ApiListDatasetsResponse()
        mock_kaggle.datasets.dataset_api_client.list_datasets.return_value = mock_response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        response = self.api.dataset_list_with_response(
            sort_by="votes",
            file_type="csv",
            license_name="cc",
            tag_ids="tag1,tag2",
            search="my search",
            page=2,
            max_size="1000",
            min_size="10",
        )

        mock_kaggle.datasets.dataset_api_client.list_datasets.assert_called_once()
        request = mock_kaggle.datasets.dataset_api_client.list_datasets.call_args[0][0]

        self.assertEqual(request.group, DatasetSelectionGroup.DATASET_SELECTION_GROUP_PUBLIC)
        self.assertEqual(request.sort_by, DatasetSortBy.DATASET_SORT_BY_VOTES)
        self.assertEqual(request.file_type, DatasetFileTypeGroup.DATASET_FILE_TYPE_GROUP_CSV)
        self.assertEqual(request.license, DatasetLicenseGroup.DATASET_LICENSE_GROUP_CC)
        self.assertEqual(request.tag_ids, "tag1,tag2")
        self.assertEqual(request.search, "my search")
        self.assertEqual(request.user, "")
        self.assertEqual(request.page, 2)
        self.assertEqual(request.page_size, 20)
        self.assertEqual(request.max_size, 1000)
        self.assertEqual(request.min_size, 10)
        self.assertEqual(response, mock_response)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_list_with_response_mine_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.datasets.dataset_api_client.list_datasets.return_value = ApiListDatasetsResponse()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_list_with_response(mine=True)

        request = mock_kaggle.datasets.dataset_api_client.list_datasets.call_args[0][0]
        self.assertEqual(request.group, DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_list_with_response_user_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.datasets.dataset_api_client.list_datasets.return_value = ApiListDatasetsResponse()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_list_with_response(user="someuser")

        request = mock_kaggle.datasets.dataset_api_client.list_datasets.call_args[0][0]
        self.assertEqual(request.group, DatasetSelectionGroup.DATASET_SELECTION_GROUP_USER)
        self.assertEqual(request.user, "someuser")

    @patch.object(KaggleApi, "dataset_list_with_response")
    def test_dataset_list_wrapper_succeeds(self, mock_list_with_response):
        mock_response = MagicMock()
        mock_datasets = [MagicMock(), MagicMock()]
        mock_response.datasets = mock_datasets
        mock_list_with_response.return_value = mock_response

        result = self.api.dataset_list(sort_by="votes", search="test")

        mock_list_with_response.assert_called_once_with(
            sort_by="votes",
            size=None,
            file_type=None,
            license_name=None,
            tag_ids=None,
            search="test",
            user=None,
            mine=False,
            page=1,
            max_size=None,
            min_size=None,
        )
        self.assertEqual(result, mock_datasets)

    @patch.object(KaggleApi, "dataset_list_with_response")
    def test_dataset_list_wrapper_empty_response_returns_none(self, mock_list_with_response):
        mock_list_with_response.return_value = None
        result = self.api.dataset_list()
        self.assertIsNone(result)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_list_with_response_page_token_succeeds(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.datasets.dataset_api_client.list_datasets.return_value = ApiListDatasetsResponse()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_list_with_response(page_token="token123")

        request = mock_kaggle.datasets.dataset_api_client.list_datasets.call_args[0][0]
        self.assertEqual(request.page_token, "token123")


if __name__ == "__main__":
    unittest.main()
