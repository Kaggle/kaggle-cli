import unittest
from unittest.mock import MagicMock, patch


class TestDatasetStatus(unittest.TestCase):
    def setUp(self):
        self.api = self._create_api()

    def _create_api(self):
        with patch("kaggle.api.kaggle_api_extended.KaggleApi._read_config_file"):
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi.__new__(KaggleApi)
            api.already_printed_version_warning = True
            api.config_values = {}
            return api

    def _mock_kaggle_client(self, status_name, current_version_number):
        mock_kaggle = MagicMock()

        # Mock get_dataset_status response
        mock_status_response = MagicMock()
        mock_status_response.status.name = status_name
        mock_kaggle.datasets.dataset_api_client.get_dataset_status.return_value = mock_status_response

        # Mock get_dataset response
        mock_dataset_response = MagicMock()
        mock_dataset_response.current_version_number = current_version_number
        mock_kaggle.datasets.dataset_api_client.get_dataset.return_value = mock_dataset_response

        return mock_kaggle

    @patch("kaggle.api.kaggle_api_extended.KaggleApi.build_kaggle_client")
    def test_dataset_status_returns_status_and_version(self, mock_build):
        mock_kaggle = self._mock_kaggle_client("READY", 3)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        status, version = self.api.dataset_status("owner/dataset-name")

        self.assertEqual(status, "ready")
        self.assertEqual(version, 3)

    @patch("kaggle.api.kaggle_api_extended.KaggleApi.build_kaggle_client")
    def test_dataset_status_cli_formats_with_version(self, mock_build):
        mock_kaggle = self._mock_kaggle_client("READY", 5)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name")

        self.assertEqual(result, "ready (version 5)")

    @patch("kaggle.api.kaggle_api_extended.KaggleApi.build_kaggle_client")
    def test_dataset_status_cli_pending(self, mock_build):
        mock_kaggle = self._mock_kaggle_client("PENDING", 2)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name")

        self.assertEqual(result, "pending (version 2)")

    @patch("kaggle.api.kaggle_api_extended.KaggleApi.build_kaggle_client")
    def test_dataset_status_cli_version_none(self, mock_build):
        mock_kaggle = self._mock_kaggle_client("READY", None)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name")

        self.assertEqual(result, "ready")

    @patch("kaggle.api.kaggle_api_extended.KaggleApi.build_kaggle_client")
    def test_dataset_status_cli_version_1(self, mock_build):
        mock_kaggle = self._mock_kaggle_client("READY", 1)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli("owner/dataset-name")

        self.assertEqual(result, "ready (version 1)")

    @patch("kaggle.api.kaggle_api_extended.KaggleApi.build_kaggle_client")
    def test_dataset_status_cli_uses_dataset_opt(self, mock_build):
        mock_kaggle = self._mock_kaggle_client("READY", 3)
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.dataset_status_cli(None, dataset_opt="owner/dataset-name")

        self.assertEqual(result, "ready (version 3)")

    def test_dataset_status_raises_on_none(self):
        with self.assertRaises(ValueError):
            self.api.dataset_status(None)


if __name__ == "__main__":
    unittest.main()
