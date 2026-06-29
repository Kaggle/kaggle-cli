# coding=utf-8
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


def _mock_created_page(name="rules", is_published=False):
    page = MagicMock()
    page.name = name
    page.is_published = is_published
    return page


class TestCompetitionCreatePage(unittest.TestCase):
    """Tests for competition_create_page and its CLI wrapper."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
        self._tmp.write("# Rules\nYou must follow these rules.\n")
        self._tmp.close()
        self.content_path = self._tmp.name

    def tearDown(self):
        os.unlink(self.content_path)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_create_page_sends_expected_request(self, mock_client):
        created = _mock_created_page(name="rules", is_published=True)
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.create_competition_page.return_value = created
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.competition_create_page(
            competition_name="my-comp",
            page_name="rules",
            content_path=self.content_path,
            mime_type="text/markdown",
            post_title="Competition Rules",
            publish=True,
        )

        self.assertIs(result, created)
        call = mock_kaggle.competitions.competition_api_client.create_competition_page.call_args
        request = call[0][0]
        self.assertEqual(request.competition_name, "my-comp")
        self.assertEqual(request.page.name, "rules")
        self.assertEqual(request.page.content, "# Rules\nYou must follow these rules.\n")
        self.assertEqual(request.page.mime_type, "text/markdown")
        self.assertEqual(request.page.post_title, "Competition Rules")
        self.assertTrue(request.page.is_published)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_create_page_defaults_unpublished_and_omits_optionals(self, mock_client):
        mock_kaggle = MagicMock()
        mock_kaggle.competitions.competition_api_client.create_competition_page.return_value = _mock_created_page(
            name="description", is_published=False
        )
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        self.api.competition_create_page(
            competition_name="my-comp",
            page_name="description",
            content_path=self.content_path,
        )

        request = mock_kaggle.competitions.competition_api_client.create_competition_page.call_args[0][0]
        self.assertEqual(request.page.name, "description")
        self.assertFalse(request.page.is_published)
        # mime_type / post_title left unset so the server applies its defaults.
        self.assertEqual(request.page.mime_type, "")
        self.assertEqual(request.page.post_title, "")

    def test_create_page_missing_content_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_page(
                competition_name="my-comp",
                page_name="rules",
                content_path="/tmp/does-not-exist-12345.md",
            )
        self.assertIn("Content file not found", str(ctx.exception))

    @patch.object(KaggleApi, "competition_create_page")
    def test_cli_invokes_api_with_resolved_args(self, mock_create):
        mock_create.return_value = _mock_created_page(name="rules", is_published=True)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.competition_create_page_cli(
                competition="my-comp",
                page_name="rules",
                file_path=self.content_path,
                mime_type="text/markdown",
                publish=True,
            )
        finally:
            sys.stdout = sys.__stdout__

        mock_create.assert_called_once_with(
            competition_name="my-comp",
            page_name="rules",
            content_path=self.content_path,
            mime_type="text/markdown",
            post_title=None,
            publish=True,
        )
        output = captured.getvalue()
        self.assertIn('Page "rules" created on competition "my-comp"', output)
        self.assertIn("published", output)

    @patch.object(KaggleApi, "competition_create_page")
    def test_cli_uses_competition_opt_flag(self, mock_create):
        mock_create.return_value = _mock_created_page(name="rules", is_published=False)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.competition_create_page_cli(
                competition_opt="my-comp",
                page_name="rules",
                file_path=self.content_path,
            )
        finally:
            sys.stdout = sys.__stdout__

        self.assertEqual(mock_create.call_args.kwargs["competition_name"], "my-comp")
        self.assertIn("staged (unpublished)", captured.getvalue())

    def test_cli_missing_competition_raises(self):
        self.api.config_values = {}
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_page_cli(
                page_name="rules",
                file_path=self.content_path,
            )
        self.assertIn("No competition specified", str(ctx.exception))

    def test_cli_missing_page_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_page_cli(
                competition="my-comp",
                file_path=self.content_path,
            )
        self.assertIn("--page-name is required", str(ctx.exception))

    def test_cli_missing_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_page_cli(
                competition="my-comp",
                page_name="rules",
            )
        self.assertIn("-f/--file is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
