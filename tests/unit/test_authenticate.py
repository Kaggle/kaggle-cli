from kaggle.api.kaggle_api_extended import KaggleApi

# python -m unittest tests.test_authenticate

from contextlib import redirect_stdout
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from requests.exceptions import HTTPError
from requests.models import Response


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

    def test_kgat_token_in_kaggle_key_env_var_is_rejected(self):
        api = KaggleApi()
        token = "KGAT_0123456789abcdef0123456789abcdef"
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "dinosaur", "KAGGLE_KEY": token}, clear=False):
            api.config_values = {"username": "dinosaur", "key": token}
            captured = io.StringIO()
            with redirect_stdout(captured):
                result = api._authenticate_with_legacy_apikey()

        self.assertFalse(result)
        self.assertNotIn("key", api.config_values)
        output = captured.getvalue()
        self.assertIn("KAGGLE_KEY", output)
        self.assertIn("KAGGLE_API_TOKEN", output)
        self.assertIn("access_token", output)

    def test_kgat_token_in_kaggle_json_key_is_rejected(self):
        api = KaggleApi()
        token = "KGAT_0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as tmpdir:
            api.config = os.path.join(tmpdir, "kaggle.json")
            with open(api.config, "w", encoding="utf-8") as f:
                f.write(f'{{"username": "dinosaur", "key": "{token}"}}')
            os.chmod(api.config, 0o600)

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("KAGGLE_KEY", None)
                api._load_config()
                captured = io.StringIO()
                with redirect_stdout(captured):
                    result = api._authenticate_with_legacy_apikey()

        self.assertFalse(result)
        self.assertNotIn("key", api.config_values)
        output = captured.getvalue()
        self.assertIn("kaggle.json", output)
        self.assertIn("KAGGLE_API_TOKEN", output)
        self.assertIn("access_token", output)

    def test_raw_kgat_token_in_kaggle_json_file_prints_error(self):
        api = KaggleApi()
        token = "KGAT_0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as tmpdir:
            api.config = os.path.join(tmpdir, "kaggle.json")
            with open(api.config, "w", encoding="utf-8") as f:
                f.write(token)
            os.chmod(api.config, 0o600)

            captured = io.StringIO()
            with redirect_stdout(captured):
                config_data = api.read_config_file(quiet=True)

        self.assertEqual(config_data, {})
        output = captured.getvalue()
        self.assertIn("KGAT_", output)
        self.assertIn("kaggle.json", output)
        self.assertIn("access_token", output)

    def test_malformed_json_in_kaggle_json_file_prints_error(self):
        api = KaggleApi()
        with tempfile.TemporaryDirectory() as tmpdir:
            api.config = os.path.join(tmpdir, "kaggle.json")
            with open(api.config, "w", encoding="utf-8") as f:
                f.write("not-valid-json")
            os.chmod(api.config, 0o600)

            captured = io.StringIO()
            with redirect_stdout(captured):
                config_data = api.read_config_file(quiet=True)

        self.assertEqual(config_data, {})
        output = captured.getvalue()
        self.assertIn("kaggle.json", output)
        self.assertIn("JSON", output)

    @patch.object(KaggleApi, "_introspect_token")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_legacy_json_in_access_token_is_rejected(self, mock_get_token, mock_introspect):
        mock_get_token.return_value = (
            '{"username": "dinosaur", "key": "0123456789abcdef0123456789abcdef"}',
            "access_token",
        )
        api = KaggleApi()
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = api._authenticate_with_access_token()

        self.assertFalse(result)
        mock_introspect.assert_not_called()
        output = captured.getvalue()
        self.assertIn("kaggle.json", output)
        self.assertIn("access_token", output)

    @patch.object(KaggleApi, "_introspect_token")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_legacy_hex_key_in_access_token_is_rejected(self, mock_get_token, mock_introspect):
        mock_get_token.return_value = ("0123456789abcdef0123456789abcdef", "KAGGLE_API_TOKEN")
        api = KaggleApi()
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = api._authenticate_with_access_token()

        self.assertFalse(result)
        mock_introspect.assert_not_called()
        output = captured.getvalue()
        self.assertIn("legacy API key", output)
        self.assertIn("KAGGLE_API_TOKEN", output)
        self.assertIn("KGAT_", output)

    @patch.object(KaggleApi, "_introspect_token")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_malformed_kgat_token_in_access_token_is_rejected(self, mock_get_token, mock_introspect):
        mock_get_token.return_value = ("KGAT_too_short", "KAGGLE_API_TOKEN")
        api = KaggleApi()
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = api._authenticate_with_access_token()

        self.assertFalse(result)
        mock_introspect.assert_not_called()
        output = captured.getvalue()
        self.assertIn("Malformed", output)
        self.assertIn("KGAT_", output)
        self.assertIn("32 hexadecimal", output)

    @patch.object(KaggleApi, "_introspect_token", return_value="dinosaur")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_valid_kgat_token_in_access_token_succeeds(self, mock_get_token, mock_introspect):
        token = "KGAT_0123456789abcdef0123456789abcdef"
        mock_get_token.return_value = (token, "KAGGLE_API_TOKEN")
        api = KaggleApi()
        api.config_values = {}
        result = api._authenticate_with_access_token()

        self.assertTrue(result)
        mock_introspect.assert_called_once_with(token)
        self.assertEqual(api.config_values["token"], token)
        self.assertEqual(api.config_values["username"], "dinosaur")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_introspect_token_handles_401_without_raising(self, mock_build_client):
        response = Response()
        response.status_code = 401
        mock_kaggle = MagicMock()
        mock_kaggle.security.oauth_client.introspect_token.side_effect = HTTPError(response=response)
        mock_build_client.return_value.__enter__.return_value = mock_kaggle
        mock_build_client.return_value.__exit__.return_value = False

        api = KaggleApi()
        self.assertIsNone(api._introspect_token("KGAT_0123456789abcdef0123456789abcdef"))

    def test_kgat_token_in_kaggle_json_token_field_is_rejected(self):
        api = KaggleApi()
        token = "KGAT_0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as tmpdir:
            api.config = os.path.join(tmpdir, "kaggle.json")
            with open(api.config, "w", encoding="utf-8") as f:
                f.write(f'{{"token": "{token}"}}')
            os.chmod(api.config, 0o600)

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("KAGGLE_KEY", None)
                os.environ.pop("KAGGLE_API_TOKEN", None)
                api._load_config()
                captured = io.StringIO()
                with redirect_stdout(captured):
                    result = api._authenticate_with_legacy_apikey()

        self.assertFalse(result)
        output = captured.getvalue()
        self.assertIn("kaggle.json", output)
        self.assertIn("KAGGLE_API_TOKEN", output)

    @patch.object(KaggleApi, "_introspect_token")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_json_wrapped_kgat_token_in_access_token_is_rejected(self, mock_get_token, mock_introspect):
        mock_get_token.return_value = (
            '{"token": "KGAT_0123456789abcdef0123456789abcdef"}',
            "access_token",
        )
        api = KaggleApi()
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = api._authenticate_with_access_token()

        self.assertFalse(result)
        mock_introspect.assert_not_called()
        output = captured.getvalue()
        self.assertIn("JSON", output)
        self.assertIn("KGAT_", output)

    @patch.object(KaggleApi, "_introspect_token")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_refresh_token_in_access_token_is_rejected(self, mock_get_token, mock_introspect):
        mock_get_token.return_value = ("KGRT_0123456789abcdef0123456789abcdef", "KAGGLE_API_TOKEN")
        api = KaggleApi()
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = api._authenticate_with_access_token()

        self.assertFalse(result)
        mock_introspect.assert_not_called()
        output = captured.getvalue()
        self.assertIn("refresh token", output)
        self.assertIn("KGAT_", output)

    @patch.object(KaggleApi, "_introspect_token")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_env_assignment_syntax_in_access_token_is_rejected(self, mock_get_token, mock_introspect):
        mock_get_token.return_value = (
            "KAGGLE_API_TOKEN=KGAT_0123456789abcdef0123456789abcdef",
            "access_token",
        )
        api = KaggleApi()
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = api._authenticate_with_access_token()

        self.assertFalse(result)
        mock_introspect.assert_not_called()
        output = captured.getvalue()
        self.assertIn("KEY=VALUE", output)

    @patch.object(KaggleApi, "_introspect_token", return_value="dinosaur")
    @patch("kaggle.api.kaggle_api_extended.get_access_token_from_env")
    def test_oauth_bearer_token_in_access_token_succeeds(self, mock_get_token, mock_introspect):
        oauth_token = "CfDJ8N2_fake_encrypted_oauth_access_token_payload"
        mock_get_token.return_value = (oauth_token, "KAGGLE_API_TOKEN")
        api = KaggleApi()
        api.config_values = {}
        result = api._authenticate_with_access_token()

        self.assertTrue(result)
        mock_introspect.assert_called_once_with(oauth_token)
        self.assertEqual(api.config_values["token"], oauth_token)
        self.assertEqual(api.config_values["username"], "dinosaur")

    def test_valid_kaggle_json_file_succeeds(self):
        api = KaggleApi()
        key = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as tmpdir:
            api.config = os.path.join(tmpdir, "kaggle.json")
            with open(api.config, "w", encoding="utf-8") as f:
                f.write(f'{{"username": "dinosaur", "key": "{key}"}}')
            os.chmod(api.config, 0o600)

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("KAGGLE_USERNAME", None)
                os.environ.pop("KAGGLE_KEY", None)
                os.environ.pop("KAGGLE_API_TOKEN", None)
                api._load_config()
                result = api._authenticate_with_legacy_apikey()

        self.assertTrue(result)
        self.assertEqual(api.config_values["username"], "dinosaur")
        self.assertEqual(api.config_values["key"], key)


if __name__ == "__main__":
    unittest.main()
