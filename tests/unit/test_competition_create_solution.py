# coding=utf-8
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiBlobType


def _mock_upload_file(token):
    uf = MagicMock()
    uf.token = token
    return uf


class TestCompetitionCreateSolution(unittest.TestCase):
    """Tests for competition_create_solution and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {}
        self.tmp = tempfile.mkdtemp()
        self.solution_path = os.path.join(self.tmp, "solution.csv")
        with open(self.solution_path, "w") as f:
            f.write("id,target\n1,0\n2,1\n")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_client(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.create_competition_solution.return_value = None
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_uploads_file_via_competition_solution_bucket_and_sends_token(self, mock_client, mock_upload):
        mock_upload.return_value = _mock_upload_file("blob-tok-1")
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_create_solution(competition_name="my-comp", path=self.solution_path)

        # _upload_file called with basename, full path, ApiBlobType.COMPETITION_SOLUTION
        # (avoids the generic InboxFiles bucket so the source blob doesn't linger).
        args, _ = mock_upload.call_args
        self.assertEqual(args[0], "solution.csv")
        self.assertEqual(args[1], self.solution_path)
        self.assertEqual(args[2], ApiBlobType.COMPETITION_SOLUTION)

        # The blob token flows through to CreateCompetitionSolution
        request = mock_kaggle.competitions.competition_api_client.create_competition_solution.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(request.blob_token, "blob-tok-1")

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_upload_failure_raises(self, mock_client, mock_upload):
        mock_upload.return_value = None
        self._patch_client(mock_client)

        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_solution(competition_name="my-comp", path=self.solution_path)
        self.assertIn("upload failed", str(ctx.exception))

    def test_missing_path_raises(self):
        # Empty string flows into os.path.exists("") -> False, so we surface the
        # Invalid-path error. (Argparse's required=True catches this at the CLI
        # boundary; the guard here is for direct Python callers.)
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_solution(competition_name="my-comp", path="")
        self.assertIn("Invalid path", str(ctx.exception))

    def test_directory_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_solution(competition_name="my-comp", path=self.tmp)
        self.assertIn("single CSV file", str(ctx.exception))

    def test_nonexistent_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_solution(
                competition_name="my-comp", path=os.path.join(self.tmp, "does-not-exist.csv")
            )
        self.assertIn("Invalid path", str(ctx.exception))

    @patch.object(KaggleApi, "competition_create_solution")
    def test_cli_positional_competition(self, mock_create):
        with redirect_stdout(io.StringIO()):
            self.api.competition_create_solution_cli(competition="my-comp", path=self.solution_path)
        mock_create.assert_called_once_with(competition_name="my-comp", path=self.solution_path, quiet=False)

    @patch.object(KaggleApi, "competition_create_solution")
    def test_cli_dash_c_option(self, mock_create):
        with redirect_stdout(io.StringIO()):
            self.api.competition_create_solution_cli(competition_opt="my-comp", path=self.solution_path)
        mock_create.assert_called_once_with(competition_name="my-comp", path=self.solution_path, quiet=False)

    @patch.object(KaggleApi, "competition_create_solution")
    def test_cli_falls_back_to_config(self, mock_create):
        self.api.config_values = {self.api.CONFIG_NAME_COMPETITION: "from-config"}
        with redirect_stdout(io.StringIO()):
            self.api.competition_create_solution_cli(path=self.solution_path)
        mock_create.assert_called_once_with(competition_name="from-config", path=self.solution_path, quiet=False)

    def test_cli_missing_competition_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_solution_cli(path=self.solution_path)
        self.assertIn("No competition specified", str(ctx.exception))

    def test_cli_missing_path_raises(self):
        with self.assertRaises(ValueError):
            self.api.competition_create_solution_cli(competition="my-comp")


if __name__ == "__main__":
    unittest.main()
