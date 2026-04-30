# coding=utf-8
import json
import unittest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, "..")

from kaggle.api.kaggle_api_extended import KaggleApi, _parse_format


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    api.config_values = {"username": "owner"}
    return api


def _mock_kaggle_client(status_name, current_version_number):
    mock_kaggle = MagicMock()
    mock_status_response = MagicMock()
    mock_status_response.status.name = status_name
    mock_kaggle.datasets.dataset_api_client.get_dataset_status.return_value = mock_status_response

    mock_dataset_response = MagicMock()
    mock_dataset_response.current_version_number = current_version_number
    mock_kaggle.datasets.dataset_api_client.get_dataset.return_value = mock_dataset_response
    return mock_kaggle


class TestDatasetStatus(unittest.TestCase):
    """Tests for dataset_status() and dataset_status_cli() including --format support."""

    def setUp(self):
        self.api = _make_api()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_returns_status_string(self, mock_build):
        """Library function dataset_status() must continue returning just the
        status string for backward compatibility with notebooks/scripts that
        import the kaggle package."""
        mock_kaggle = _mock_kaggle_client("READY", 3)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status("owner/dataset-name")

        self.assertEqual(result, "ready")
        # Library helper must not call get_dataset; it should only fetch status.
        mock_kaggle.datasets.dataset_api_client.get_dataset.assert_not_called()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_default_format_is_text_status_only(self, mock_build):
        """Default CLI behavior is the historic text output containing only
        the status (no version), for back-compat."""
        mock_kaggle = _mock_kaggle_client("READY", 5)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name")

        self.assertEqual(result, "ready")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_uses_dataset_opt(self, mock_build):
        mock_kaggle = _mock_kaggle_client("READY", 3)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli(None, dataset_opt="owner/dataset-name")

        self.assertEqual(result, "ready")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_format_json(self, mock_build):
        mock_kaggle = _mock_kaggle_client("READY", 3)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name", format="json")

        self.assertEqual(json.loads(result), {"status": "ready", "current_version_number": 3})

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_format_json_field_selection(self, mock_build):
        mock_kaggle = _mock_kaggle_client("PENDING", 7)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name", format="json(current_version_number)")

        self.assertEqual(json.loads(result), {"current_version_number": 7})
        # Field selection should skip the unrelated API call.
        mock_kaggle.datasets.dataset_api_client.get_dataset_status.assert_not_called()
        mock_kaggle.datasets.dataset_api_client.get_dataset.assert_called_once()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_format_json_status_only(self, mock_build):
        mock_kaggle = _mock_kaggle_client("READY", 9)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name", format="json(status)")

        self.assertEqual(json.loads(result), {"status": "ready"})
        mock_kaggle.datasets.dataset_api_client.get_dataset.assert_not_called()
        mock_kaggle.datasets.dataset_api_client.get_dataset_status.assert_called_once()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_format_json_multi_field_selection(self, mock_build):
        mock_kaggle = _mock_kaggle_client("READY", 2)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name", format="json(status, current_version_number)")

        self.assertEqual(json.loads(result), {"status": "ready", "current_version_number": 2})

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_unknown_format_raises(self, mock_build):
        with self.assertRaises(ValueError):
            self.api.dataset_status_cli("owner/dataset-name", format="yaml")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_dataset_status_cli_unknown_field_raises(self, mock_build):
        mock_kaggle = _mock_kaggle_client("READY", 3)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        with self.assertRaises(ValueError):
            self.api.dataset_status_cli("owner/dataset-name", format="json(bogus)")

    def test_dataset_status_raises_on_none(self):
        with self.assertRaises(ValueError):
            self.api.dataset_status(None)


class TestParseFormat(unittest.TestCase):
    def test_parse_format_plain(self):
        self.assertEqual(_parse_format("json"), ("json", []))

    def test_parse_format_single_field(self):
        self.assertEqual(
            _parse_format("json(current_version_number)"),
            ("json", ["current_version_number"]),
        )

    def test_parse_format_multiple_fields_with_whitespace(self):
        self.assertEqual(
            _parse_format("json( status , current_version_number )"),
            ("json", ["status", "current_version_number"]),
        )

    def test_parse_format_malformed_raises(self):
        with self.assertRaises(ValueError):
            _parse_format("json(status")


if __name__ == "__main__":
    unittest.main()
