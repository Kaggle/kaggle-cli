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

    @patch.object(KaggleApi, "_load_config")
    @patch.object(KaggleApi, "_authenticate_with_access_token")
    @patch.object(KaggleApi, "_authenticate_with_legacy_apikey")
    @patch.object(KaggleApi, "_authenticate_with_oauth_creds")
    @patch.object(KaggleApi, "_authenticate_anonymously")
    def test_authenticate_call_sequence_and_fallback(self, mock_anon, mock_oauth, mock_legacy, mock_access, mock_load):
        api = KaggleApi()

        # Access token succeeds
        mock_access.return_value = True
        api.authenticate()
        mock_load.assert_called_once()
        mock_access.assert_called_once()
        mock_legacy.assert_not_called()

        # Legacy key succeeds
        mock_load.reset_mock()
        mock_access.reset_mock()
        mock_access.return_value = False
        mock_legacy.return_value = True
        api.authenticate()
        mock_access.assert_called_once()
        mock_legacy.assert_called_once()
        mock_oauth.assert_not_called()

        # OAuth credentials succeeds
        mock_legacy.reset_mock()
        mock_legacy.return_value = False
        mock_oauth.return_value = True
        api.authenticate()
        mock_legacy.assert_called_once()
        mock_oauth.assert_called_once()
        mock_anon.assert_not_called()

        # Anonymous fallback succeeds
        mock_oauth.reset_mock()
        mock_oauth.return_value = False
        mock_anon.return_value = True
        api.authenticate()
        mock_oauth.assert_called_once()
        mock_anon.assert_called_once()

    def test_authenticate_anonymously_detects_logged_out_commands(self):
        api = KaggleApi()
        with patch("sys.argv", ["kaggle", "datasets", "download", "some/dataset"]):
            self.assertTrue(api._authenticate_anonymously())
        with patch("sys.argv", ["kaggle", "datasets", "status", "some/dataset"]):
            self.assertFalse(api._authenticate_anonymously())

    def test_legacy_apikey_does_not_handle_anonymous_fallback(self):
        api = KaggleApi()
        api.config_values = {}
        with patch("sys.argv", ["kaggle", "datasets", "download", "some/dataset"]):
            self.assertFalse(api._authenticate_with_legacy_apikey())


if __name__ == "__main__":
    unittest.main()
