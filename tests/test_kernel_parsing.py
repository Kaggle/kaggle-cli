# coding=utf-8
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "../src")


from kaggle.api.kaggle_api_extended import KaggleApi


class TestKernelParsing(unittest.TestCase):

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def test_validate_kernel_string_valid(self):
        # Valid formats
        self.api.validate_kernel_string("owner/slug-name")
        self.api.validate_kernel_string("owner/slug-name/1")
        self.api.validate_kernel_string("owner/slug-name/123")
        self.api.validate_kernel_string("owner/slug-name/notanint")
        self.api.validate_kernel_string("owner/slug-name/1.2")

    def test_validate_kernel_string_invalid_format(self):
        # Invalid formats
        with self.assertRaises(ValueError):
            self.api.validate_kernel_string("noslash")
        with self.assertRaises(ValueError):
            self.api.validate_kernel_string("owner/slug/version/extra")
        with self.assertRaises(ValueError):
            self.api.validate_kernel_string("/slug")
        with self.assertRaises(ValueError):
            self.api.validate_kernel_string("owner/")
        with self.assertRaises(ValueError):
            self.api.validate_kernel_string("owner/slug/")

    def test_validate_kernel_string_invalid_slug_length(self):
        # Slug must be >= 5 chars
        with self.assertRaises(ValueError):
            self.api.validate_kernel_string("owner/slug")  # slug is 4 chars
        self.api.validate_kernel_string("owner/slug5")  # slug is 5 chars

    def test_parse_kernel_string_three_parts(self):
        owner, slug, version = self.api.parse_kernel_string("owner/slug-name/123")
        self.assertEqual(owner, "owner")
        self.assertEqual(slug, "slug-name")
        self.assertEqual(version, "123")

    def test_parse_kernel_string_two_parts(self):
        owner, slug, version = self.api.parse_kernel_string("owner/slug-name")
        self.assertEqual(owner, "owner")
        self.assertEqual(slug, "slug-name")
        self.assertEqual(version, None)

    @patch.object(KaggleApi, "get_config_value")
    def test_parse_kernel_string_one_part(self, mock_get_config):
        mock_get_config.return_value = "default-owner"
        owner, slug, version = self.api.parse_kernel_string("slug-name")
        self.assertEqual(owner, "default-owner")
        self.assertEqual(slug, "slug-name")
        self.assertEqual(version, None)
        mock_get_config.assert_called_once_with(KaggleApi.CONFIG_NAME_USER)

    @patch.object(KaggleApi, "get_config_value")
    def test_parse_kernel_string_one_part_no_config(self, mock_get_config):
        mock_get_config.return_value = None
        owner, slug, version = self.api.parse_kernel_string("slug-name")
        self.assertEqual(owner, "")
        self.assertEqual(slug, "slug-name")
        self.assertEqual(version, None)


if __name__ == "__main__":
    unittest.main()
