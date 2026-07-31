# coding=utf-8
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestCompetitionDownloadSubmission(unittest.TestCase):
    """Tests for competition_download_submission and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}

    def _patch_client(self, mock_client, url="https://storage.example.com/prod/submission.csv?sig=abc"):
        mock_kaggle = MagicMock()
        response = MagicMock()
        response.url = url
        mock_kaggle.competitions.competition_api_client.download_submission.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle, response

    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_builds_request_with_int_submission_id(self, mock_client, _mock_needed, _mock_download):
        mock_kaggle, _ = self._patch_client(mock_client)

        self.api.competition_download_submission("12345", path="/tmp/out")

        request = mock_kaggle.competitions.competition_api_client.download_submission.call_args[0][0]
        self.assertEqual(request.submission_id, 12345)
        self.assertIsInstance(request.submission_id, int)

    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_derives_filename_from_redirect_url(self, mock_client, _mock_needed, mock_download):
        self._patch_client(mock_client, url="https://storage.example.com/x/my-sub.csv?token=z")

        self.api.competition_download_submission(1, path="/tmp/out")

        outfile = mock_download.call_args[0][1]
        self.assertEqual(outfile, os.path.join("/tmp/out", "my-sub.csv"))

    @patch.object(KaggleApi, "get_default_download_dir", return_value="/default/dir")
    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_defaults_path_to_download_dir(self, mock_client, _mock_needed, mock_download, mock_dir):
        self._patch_client(mock_client)

        self.api.competition_download_submission(1)

        mock_dir.assert_called_once_with("competitions", "submissions")
        outfile = mock_download.call_args[0][1]
        self.assertEqual(os.path.dirname(outfile), "/default/dir")

    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_skips_download_when_not_needed(self, mock_client, _mock_needed, mock_download):
        self._patch_client(mock_client)

        self.api.competition_download_submission(1, path="/tmp/out")

        mock_download.assert_not_called()

    @patch.object(KaggleApi, "download_file")
    @patch.object(KaggleApi, "download_needed", return_value=False)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_force_downloads_even_when_not_needed(self, mock_client, mock_needed, mock_download):
        self._patch_client(mock_client)

        self.api.competition_download_submission(1, path="/tmp/out", force=True)

        # force short-circuits the up-to-date check entirely
        mock_needed.assert_not_called()
        mock_download.assert_called_once()
        # resume flag (5th positional) is the negation of force
        self.assertFalse(mock_download.call_args[0][4])

    # --- CLI wrapper --------------------------------------------------------

    @patch.object(KaggleApi, "competition_download_submission")
    def test_cli_positional_id(self, mock_download):
        self.api.competition_download_submission_cli(submission_id=42)
        mock_download.assert_called_once_with(42, None, False, False)

    @patch.object(KaggleApi, "competition_download_submission")
    def test_cli_dash_i_option(self, mock_download):
        self.api.competition_download_submission_cli(submission_id_opt=42)
        mock_download.assert_called_once_with(42, None, False, False)

    @patch.object(KaggleApi, "competition_download_submission")
    def test_cli_passes_path_and_force(self, mock_download):
        self.api.competition_download_submission_cli(submission_id=42, path="/tmp/out", force=True, quiet=True)
        mock_download.assert_called_once_with(42, "/tmp/out", True, True)

    def test_cli_missing_id_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_download_submission_cli()
        self.assertIn("submission id must be specified", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
