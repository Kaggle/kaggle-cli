import unittest
from unittest.mock import patch, mock_open
import json
from kagglesdk.kaggle_http_client import _get_apikey_creds


class TestHttpClient(unittest.TestCase):
    @patch("os.path.exists")
    def test_get_apikey_creds_missing_username(self, mock_exists):
        mock_exists.return_value = True
        bad_json = '{"competition": "titanic"}'
        with patch("builtins.open", mock_open(read_data=bad_json)):
            # This should return None, not raise KeyError
            creds = _get_apikey_creds()
            self.assertIsNone(creds)

    @patch("os.path.exists")
    def test_get_apikey_creds_invalid_json(self, mock_exists):
        mock_exists.return_value = True
        bad_json = "{invalid json}"
        with patch("builtins.open", mock_open(read_data=bad_json)):
            # This should return None, not raise JSONDecodeError
            creds = _get_apikey_creds()
            self.assertIsNone(creds)

    @patch("os.path.exists")
    def test_get_apikey_creds_valid(self, mock_exists):
        mock_exists.return_value = True
        good_json = '{"username": "testuser", "key": "testkey"}'
        with patch("builtins.open", mock_open(read_data=good_json)):
            creds = _get_apikey_creds()
            self.assertEqual(creds, ("testuser", "testkey"))


if __name__ == "__main__":
    unittest.main()
