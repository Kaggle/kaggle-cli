# coding=utf-8
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import CompetitionDatabundleType


def _mock_upload_file(token):
    uf = MagicMock()
    uf.token = token
    return uf


def _mock_response(url="https://kaggle.com/c/my-comp", db_id=1, dbv_id=42):
    r = MagicMock()
    r.url = url
    r.databundle_id = db_id
    r.databundle_version_id = dbv_id
    return r


class TestCompetitionDataUpdate(unittest.TestCase):
    """Tests for competition_data_update and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.tmp = tempfile.mkdtemp()
        # Top-level files.
        for name in ("train.csv", "test.csv"):
            with open(os.path.join(self.tmp, name), "w") as f:
                f.write("a,b\n1,2\n")
        # Nested files (recursion target).
        os.makedirs(os.path.join(self.tmp, "images", "cats"))
        with open(os.path.join(self.tmp, "images", "cats", "cat1.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        with open(os.path.join(self.tmp, "images", "dog.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        # Hidden entries (should be skipped by default).
        with open(os.path.join(self.tmp, ".DS_Store"), "wb") as f:
            f.write(b"\x00")
        os.makedirs(os.path.join(self.tmp, ".git"))
        with open(os.path.join(self.tmp, ".git", "config"), "w") as f:
            f.write("[core]\n")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_client(self, mock_client, response=None):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.create_competition_data.return_value = (
            response or _mock_response()
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_directory_recurses_and_preserves_paths(self, mock_client, mock_upload):
        # One token per file; order matches sorted rel path order.
        mock_upload.side_effect = [_mock_upload_file(f"tok-{i}") for i in range(10)]
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_data_update(
            competition_name="my-comp",
            path=self.tmp,
            version_notes="first version",
        )

        request = mock_kaggle.competitions.competition_api_client.create_competition_data.call_args[0][0]
        names = sorted(f.name for f in request.files)
        self.assertEqual(
            names,
            ["images/cats/cat1.png", "images/dog.png", "test.csv", "train.csv"],
        )
        # Names passed to _upload_file must be the same relative paths (with
        # forward slashes) that appear in the API request.
        called_names = sorted(call.args[0] for call in mock_upload.call_args_list)
        self.assertEqual(called_names, names)

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_single_file_uploads_as_is(self, mock_client, mock_upload):
        archive = os.path.join(self.tmp, "bundle.zip")
        with open(archive, "wb") as f:
            f.write(b"PK\x03\x04")
        mock_upload.return_value = _mock_upload_file("tok-zip")
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_data_update(
            competition_name="my-comp",
            path=archive,
            version_notes="single archive",
        )

        mock_upload.assert_called_once()
        # Basename becomes the API entry name; full path is uploaded.
        self.assertEqual(mock_upload.call_args.args[0], "bundle.zip")
        self.assertEqual(mock_upload.call_args.args[1], archive)
        request = mock_kaggle.competitions.competition_api_client.create_competition_data.call_args[0][0]
        self.assertEqual(len(request.files), 1)
        self.assertEqual(request.files[0].name, "bundle.zip")
        self.assertEqual(request.files[0].token, "tok-zip")

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_skips_hidden_by_default(self, mock_client, mock_upload):
        mock_upload.side_effect = [_mock_upload_file(f"tok-{i}") for i in range(10)]
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_data_update(
            competition_name="my-comp",
            path=self.tmp,
            version_notes="notes",
        )

        request = mock_kaggle.competitions.competition_api_client.create_competition_data.call_args[0][0]
        names = sorted(f.name for f in request.files)
        # Hidden files (.DS_Store) and hidden dirs (.git/) are not included.
        self.assertEqual(
            names,
            ["images/cats/cat1.png", "images/dog.png", "test.csv", "train.csv"],
        )
        for n in names:
            self.assertFalse(any(part.startswith(".") for part in n.split("/")))

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_include_hidden_uploads_them(self, mock_client, mock_upload):
        mock_upload.side_effect = [_mock_upload_file(f"tok-{i}") for i in range(10)]
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_data_update(
            competition_name="my-comp",
            path=self.tmp,
            version_notes="notes",
            include_hidden=True,
        )

        request = mock_kaggle.competitions.competition_api_client.create_competition_data.call_args[0][0]
        names = sorted(f.name for f in request.files)
        self.assertIn(".DS_Store", names)
        self.assertIn(".git/config", names)

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_rerun_sets_databundle_type(self, mock_client, mock_upload):
        mock_upload.side_effect = [_mock_upload_file(f"tok-{i}") for i in range(10)]
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_data_update(
            competition_name="my-comp",
            path=self.tmp,
            version_notes="rerun data",
            rerun=True,
        )

        request = mock_kaggle.competitions.competition_api_client.create_competition_data.call_args[0][0]
        self.assertEqual(
            request.competition_databundle_type,
            CompetitionDatabundleType.COMPETITION_DATABUNDLE_TYPE_RERUN,
        )

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_default_databundle_type_unspecified(self, mock_client, mock_upload):
        mock_upload.side_effect = [_mock_upload_file(f"tok-{i}") for i in range(10)]
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_data_update(
            competition_name="my-comp",
            path=self.tmp,
            version_notes="notes",
        )

        request = mock_kaggle.competitions.competition_api_client.create_competition_data.call_args[0][0]
        self.assertEqual(
            request.competition_databundle_type,
            CompetitionDatabundleType.COMPETITION_DATABUNDLE_TYPE_UNSPECIFIED,
        )

    @patch.object(KaggleApi, "_upload_file", return_value=None)
    def test_update_all_uploads_fail_raises(self, mock_upload):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_data_update(
                competition_name="my-comp",
                path=self.tmp,
                version_notes="notes",
            )
        self.assertIn("All file uploads failed", str(ctx.exception))

    def test_update_empty_directory_raises(self):
        empty = tempfile.mkdtemp()
        try:
            with self.assertRaises(ValueError) as ctx:
                self.api.competition_data_update(
                    competition_name="my-comp",
                    path=empty,
                    version_notes="notes",
                )
            self.assertIn("No files found", str(ctx.exception))
        finally:
            os.rmdir(empty)

    def test_update_missing_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_data_update(
                competition_name="my-comp",
                path="/tmp/does-not-exist-9999",
                version_notes="notes",
            )
        self.assertIn("Invalid path", str(ctx.exception))

    def test_update_blank_notes_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_data_update(
                competition_name="my-comp",
                path=self.tmp,
                version_notes="   ",
            )
        self.assertIn("version notes are required", str(ctx.exception))

    @patch.object(KaggleApi, "competition_data_update")
    def test_cli_forwards_args(self, mock_update):
        mock_update.return_value = _mock_response()

        self.api.competition_data_update_cli(
            competition="my-comp",
            path=self.tmp,
            version_notes="notes",
            rerun=True,
            include_hidden=True,
        )

        kwargs = mock_update.call_args.kwargs
        self.assertEqual(kwargs["competition_name"], "my-comp")
        self.assertEqual(kwargs["path"], self.tmp)
        self.assertEqual(kwargs["version_notes"], "notes")
        self.assertTrue(kwargs["rerun"])
        self.assertTrue(kwargs["include_hidden"])

    @patch.object(KaggleApi, "competition_data_update")
    def test_cli_include_hidden_defaults_false(self, mock_update):
        mock_update.return_value = _mock_response()

        self.api.competition_data_update_cli(
            competition="my-comp",
            path=self.tmp,
            version_notes="notes",
        )

        self.assertFalse(mock_update.call_args.kwargs["include_hidden"])

    def test_cli_missing_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_data_update_cli(competition="my-comp", version_notes="notes")
        self.assertIn("-p/--path is required", str(ctx.exception))

    def test_cli_missing_notes_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_data_update_cli(competition="my-comp", path=self.tmp)
        self.assertIn("version notes are required", str(ctx.exception))

    def test_cli_missing_competition_raises(self):
        self.api.config_values = {}
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_data_update_cli(path=self.tmp, version_notes="notes")
        self.assertIn("No competition specified", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
