# coding=utf-8
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDatasetFile, ApiListDatasetFilesResponse


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    return api


def _mock_kaggle_client(next_page_token):
    mock_kaggle = MagicMock()
    mock_response = ApiListDatasetFilesResponse()
    mock_response.next_page_token = next_page_token
    dataset_file = ApiDatasetFile()
    dataset_file.name = "train.csv"
    dataset_file.total_bytes = 1024
    mock_response.files.extend([dataset_file])
    mock_kaggle.datasets.dataset_api_client.list_dataset_files.return_value = mock_response
    return mock_kaggle


class TestDatasetListFilesCli(unittest.TestCase):
    """Tests for dataset_list_files_cli(), including JSON/CSV/table output."""

    def setUp(self):
        self.api = _make_api()

    def _run(self, **kwargs):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.api.dataset_list_files_cli("owner/dataset-name", **kwargs)
        return stdout.getvalue(), stderr.getvalue()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_json_output_with_next_page_token_is_valid_json(self, mock_client):
        """stdout must stay parseable when the response is paginated.

        The page token used to be printed before the JSON array, which made
        `datasets files --format json` unusable for callers that parse stdout.
        """
        mock_client.return_value.__enter__ = MagicMock(return_value=_mock_kaggle_client("next-token"))
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        stdout, stderr = self._run(output_format="json")

        self.assertEqual(json.loads(stdout), [{"name": "train.csv", "size": 1024, "creationDate": None}])
        self.assertIn("Next Page Token = next-token", stderr)
        self.assertNotIn("Next Page Token", stdout)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_json_projection_with_next_page_token_is_valid_json(self, mock_client):
        mock_client.return_value.__enter__ = MagicMock(return_value=_mock_kaggle_client("next-token"))
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        stdout, stderr = self._run(output_format="json(name)")

        self.assertEqual(json.loads(stdout), [{"name": "train.csv"}])
        self.assertIn("Next Page Token = next-token", stderr)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_table_output_keeps_next_page_token_on_stdout(self, mock_client):
        """Non-JSON output keeps the historic behaviour: token on stdout."""
        mock_client.return_value.__enter__ = MagicMock(return_value=_mock_kaggle_client("next-token"))
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        stdout, stderr = self._run()

        self.assertIn("Next Page Token = next-token", stdout)
        self.assertEqual(stderr, "")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_csv_output_keeps_next_page_token_on_stdout(self, mock_client):
        mock_client.return_value.__enter__ = MagicMock(return_value=_mock_kaggle_client("next-token"))
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        stdout, stderr = self._run(csv_display=True)

        self.assertIn("Next Page Token = next-token", stdout)
        self.assertEqual(stderr, "")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_no_next_page_token_prints_nothing_extra(self, mock_client):
        mock_client.return_value.__enter__ = MagicMock(return_value=_mock_kaggle_client(""))
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        stdout, stderr = self._run(output_format="json")

        self.assertEqual(json.loads(stdout), [{"name": "train.csv", "size": 1024, "creationDate": None}])
        self.assertNotIn("Next Page Token", stderr)


if __name__ == "__main__":
    unittest.main()
