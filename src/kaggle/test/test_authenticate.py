from kaggle.api.kaggle_api_extended import KaggleApi

# python -m unittest tests.test_authenticate

import os
import unittest
from unittest.mock import patch


class TestAuthenticate(unittest.TestCase):

    def setUp(self):
        print("setup             class:%s" % self)

    def tearDown(self):
        print("teardown          class:TestStuff")

    # Environment

    def test_environment_variables(self):
        os.environ["KAGGLE_USERNAME"] = "dinosaur"
        os.environ["KAGGLE_KEY"] = "xxxxxxxxxxxx"
        api = KaggleApi()

        # We haven't authenticated yet
        self.assertTrue("key" not in api.config_values)
        self.assertTrue("username" not in api.config_values)
        api.authenticate()

        # Should be set from the environment
        self.assertEqual(api.config_values["key"], "xxxxxxxxxxxx")
        self.assertEqual(api.config_values["username"], "dinosaur")

    # Configuration Actions

    def test_config_actions(self):
        api = KaggleApi()

        self.assertTrue(api.config_dir.endswith("kaggle"))
        self.assertEqual(api.get_config_value("doesntexist"), None)

    @patch("kaggle.api.kaggle_api_extended.KaggleApi.read_config_file")
    @patch("kaggle.api.kaggle_api_extended.KaggleApi._authenticate_with_oauth_creds")
    def test_oauth_fallback_when_legacy_config_has_no_credentials(self, mock_oauth, mock_read_config):
        username_env = os.environ.pop("KAGGLE_USERNAME", None)
        key_env = os.environ.pop("KAGGLE_KEY", None)

        try:
            api = KaggleApi()
            mock_read_config.return_value = {"proxy": "http://myproxy"}

            def fake_oauth():
                api.config_values["token"] = "oauth_token"
                api.config_values["username"] = "oauth_user"
                api.config_values["auth_method"] = "oauth"
                return True

            mock_oauth.side_effect = fake_oauth

            api.authenticate()

            self.assertEqual(api.config_values["token"], "oauth_token")
            self.assertEqual(api.config_values["username"], "oauth_user")
            self.assertEqual(api.config_values["auth_method"], "oauth")
            self.assertEqual(api.config_values.get("proxy"), "http://myproxy")

        finally:
            if username_env is not None:
                os.environ["KAGGLE_USERNAME"] = username_env
            if key_env is not None:
                os.environ["KAGGLE_KEY"] = key_env


if __name__ == "__main__":
    unittest.main()
