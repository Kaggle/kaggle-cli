# coding=utf-8
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import CompetitionPrivacy
from kagglesdk.competitions.types.competition import RewardTypeId


_MINIMAL_META = {
    "title": "My Awesome Comp",
    "slug": "my-awesome-comp",
    "briefDescription": "Test the things.",
    "privacy": "PUBLIC",
}


def _write_meta(folder, overrides=None):
    meta = dict(_MINIMAL_META)
    if overrides:
        meta.update(overrides)
    path = os.path.join(folder, KaggleApi.COMPETITION_METADATA_FILE)
    with open(path, "w") as f:
        json.dump(meta, f)
    return path


class TestCompetitionInitialize(unittest.TestCase):
    """Tests for competition_initialize (template writer)."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_template_with_required_placeholders(self):
        path = self.api.competition_initialize(self.tmp)

        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            meta = json.load(f)
        self.assertEqual(meta["title"], "INSERT_TITLE_HERE")
        self.assertEqual(meta["slug"], "INSERT_SLUG_HERE")
        self.assertEqual(meta["briefDescription"], "INSERT_BRIEF_DESCRIPTION_HERE")
        self.assertEqual(meta["privacy"], "PUBLIC")
        self.assertIn("reward", meta)
        self.assertIsNone(meta["reward"])

    def test_rejects_nonexistent_folder(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_initialize("/tmp/this-folder-does-not-exist-xyz123")
        self.assertIn("Invalid folder", str(ctx.exception))


class TestCompetitionCreateNew(unittest.TestCase):
    """Tests for competition_create_new (metadata → API)."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_client(self, mock_client, returned_url="https://kaggle.com/c/my-awesome-comp"):
        mock_kaggle = MagicMock()
        response = MagicMock()
        response.url = returned_url
        mock_kaggle.competitions.competition_api_client.create_competition.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        return mock_kaggle

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_minimal_metadata_sends_required_fields(self, mock_client):
        _write_meta(self.tmp)
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_create_new(self.tmp)

        request = mock_kaggle.competitions.competition_api_client.create_competition.call_args[0][0]
        self.assertEqual(request.title, "My Awesome Comp")
        self.assertEqual(request.slug, "my-awesome-comp")
        self.assertEqual(request.brief_description, "Test the things.")
        self.assertEqual(request.privacy, CompetitionPrivacy.PUBLIC)
        # Optional fields not present in JSON should be left at their proto defaults.
        self.assertEqual(request.clone_competition_id, 0)
        self.assertEqual(request.license_id, 0)
        self.assertIsNone(request.reward)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_optional_fields_forwarded(self, mock_client):
        _write_meta(
            self.tmp,
            {
                "disableKernels": True,
                "hackathon": True,
                "restrictLinkToEmailList": True,
                "cloneCompetitionId": 1234,
                "cloneExcludeCompetitionData": True,
                "clonePageNames": ["rules", "evaluation"],
                "licenseId": 7,
                "organizationId": 42,
                "numPrizes": 5,
                "privacy": "limited",  # case-insensitive
            },
        )
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_create_new(self.tmp)

        request = mock_kaggle.competitions.competition_api_client.create_competition.call_args[0][0]
        self.assertTrue(request.disable_kernels)
        self.assertTrue(request.hackathon)
        self.assertTrue(request.restrict_link_to_email_list)
        self.assertEqual(request.clone_competition_id, 1234)
        self.assertTrue(request.clone_exclude_competition_data)
        self.assertEqual(list(request.clone_page_names), ["rules", "evaluation"])
        self.assertEqual(request.license_id, 7)
        self.assertEqual(request.organization_id, 42)
        self.assertEqual(request.num_prizes, 5)
        self.assertEqual(request.privacy, CompetitionPrivacy.LIMITED)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_reward_parsed_into_proto(self, mock_client):
        _write_meta(
            self.tmp,
            {"reward": {"id": "USD", "quantity": 25000, "clarification": "Split top 5"}},
        )
        mock_kaggle = self._patch_client(mock_client)

        self.api.competition_create_new(self.tmp)

        request = mock_kaggle.competitions.competition_api_client.create_competition.call_args[0][0]
        self.assertIsNotNone(request.reward)
        self.assertEqual(request.reward.id, RewardTypeId.USD)
        self.assertEqual(request.reward.quantity, 25000)
        self.assertEqual(request.reward.clarification, "Split top 5")

    def test_rejects_default_title(self):
        _write_meta(self.tmp, {"title": "INSERT_TITLE_HERE"})
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_new(self.tmp)
        self.assertIn("Default title detected", str(ctx.exception))

    def test_rejects_default_slug(self):
        _write_meta(self.tmp, {"slug": "INSERT_SLUG_HERE"})
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_new(self.tmp)
        self.assertIn("Default slug detected", str(ctx.exception))

    def test_rejects_default_brief_description(self):
        _write_meta(self.tmp, {"briefDescription": "INSERT_BRIEF_DESCRIPTION_HERE"})
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_new(self.tmp)
        self.assertIn("Default briefDescription", str(ctx.exception))

    def test_rejects_invalid_privacy(self):
        _write_meta(self.tmp, {"privacy": "SECRET"})
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_new(self.tmp)
        self.assertIn("Invalid privacy 'SECRET'", str(ctx.exception))

    def test_rejects_invalid_reward_id(self):
        _write_meta(self.tmp, {"reward": {"id": "DOGECOIN", "quantity": 1}})
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_new(self.tmp)
        self.assertIn("Invalid reward.id 'DOGECOIN'", str(ctx.exception))

    def test_missing_metadata_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.api.competition_create_new(self.tmp)
        self.assertIn("Metadata file not found", str(ctx.exception))

    def test_missing_required_field_raises(self):
        bad = {k: v for k, v in _MINIMAL_META.items() if k != "slug"}
        path = os.path.join(self.tmp, KaggleApi.COMPETITION_METADATA_FILE)
        with open(path, "w") as f:
            json.dump(bad, f)
        with self.assertRaises(Exception):
            self.api.competition_create_new(self.tmp)


if __name__ == "__main__":
    unittest.main()
