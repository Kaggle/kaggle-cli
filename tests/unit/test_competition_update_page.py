# coding=utf-8
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


def _mock_updated_page(name="rules", is_published=False):
    page = MagicMock()
    page.name = name
    page.is_published = is_published
    return page


class TestCompetitionUpdatePage(unittest.TestCase):
    """Tests for competition_update_page and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
        self._tmp.write("# Updated rules\nNew content.\n")
        self._tmp.close()
        self.content_path = self._tmp.name

    def tearDown(self):
        os.unlink(self.content_path)

    def _patch_client(self, mock_client, returned_page=None):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.update_competition_page.return_value = (
            returned_page or _mock_updated_page()
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_update_content_only_sets_content_mask(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_page(
            competition_name="my-comp",
            page_name="rules",
            content_path=self.content_path,
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_page.call_args[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(request.page_name, "rules")
        self.assertEqual(request.page.content, "# Updated rules\nNew content.\n")
        self.assertEqual(list(request.update_mask.paths), ["content"])

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_rename_only_sets_name_mask(self, mock_client):
        mock_kaggle = self._patch_client(mock_client, _mock_updated_page(name="new-rules"))

        self.api.competition_update_page(
            competition_name="my-comp",
            page_name="rules",
            new_name="new-rules",
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_page.call_args[0][0]
        self.assertEqual(request.page.name, "new-rules")
        self.assertEqual(list(request.update_mask.paths), ["name"])

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_multiple_fields_set_combined_mask(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_page(
            competition_name="my-comp",
            page_name="rules",
            content_path=self.content_path,
            mime_type="text/markdown",
            post_title="Rules",
            is_published=True,
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_page.call_args[0][0]
        self.assertEqual(
            sorted(request.update_mask.paths),
            sorted(["content", "mime_type", "post_title", "is_published"]),
        )
        self.assertEqual(request.page.content, "# Updated rules\nNew content.\n")
        self.assertEqual(request.page.mime_type, "text/markdown")
        self.assertEqual(request.page.post_title, "Rules")
        self.assertTrue(request.page.is_published)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_unpublish_sets_is_published_false(self, mock_client):
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_update_page(
            competition_name="my-comp",
            page_name="rules",
            is_published=False,
        )

        request = mock_kaggle.competitions.competition_api_client.update_competition_page.call_args[0][0]
        self.assertEqual(list(request.update_mask.paths), ["is_published"])
        self.assertFalse(request.page.is_published)

    def test_no_fields_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_page(competition_name="my-comp", page_name="rules")
        self.assertIn("Nothing to update", str(ctx.exception))

    def test_missing_content_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_page(
                competition_name="my-comp",
                page_name="rules",
                content_path="/tmp/does-not-exist-9999.md",
            )
        self.assertIn("Content file not found", str(ctx.exception))

    @patch.object(KaggleApi, "competition_update_page")
    def test_cli_forwards_publish_flag(self, mock_update):
        mock_update.return_value = _mock_updated_page(is_published=True)

        self.api.competition_update_page_cli(
            competition="my-comp",
            page_name="rules",
            file_path=self.content_path,
            publish=True,
        )

        kwargs = mock_update.call_args.kwargs
        self.assertEqual(kwargs["competition_name"], "my-comp")
        self.assertEqual(kwargs["page_name"], "rules")
        self.assertEqual(kwargs["content_path"], self.content_path)
        self.assertTrue(kwargs["is_published"])

    @patch.object(KaggleApi, "competition_update_page")
    def test_cli_forwards_unpublish_flag(self, mock_update):
        mock_update.return_value = _mock_updated_page(is_published=False)

        self.api.competition_update_page_cli(
            competition="my-comp",
            page_name="rules",
            unpublish=True,
        )

        self.assertFalse(mock_update.call_args.kwargs["is_published"])

    @patch.object(KaggleApi, "competition_update_page")
    def test_cli_default_is_published_none(self, mock_update):
        mock_update.return_value = _mock_updated_page()

        self.api.competition_update_page_cli(
            competition="my-comp",
            page_name="rules",
            file_path=self.content_path,
        )

        self.assertIsNone(mock_update.call_args.kwargs["is_published"])

    def test_cli_publish_and_unpublish_mutually_exclusive(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_page_cli(
                competition="my-comp",
                page_name="rules",
                publish=True,
                unpublish=True,
            )
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_cli_missing_competition_raises(self):
        self.api.config_values = {}
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_page_cli(page_name="rules", file_path=self.content_path)
        self.assertIn("No competition specified", str(ctx.exception))

    def test_cli_missing_page_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_update_page_cli(competition="my-comp", file_path=self.content_path)
        self.assertIn("--page-name is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
