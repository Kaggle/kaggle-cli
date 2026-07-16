# coding=utf-8
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "../../src")

from kaggle.api.kaggle_api_extended import KaggleApi


class TestHelpOrVersionAuth(unittest.TestCase):
    """Tests for help/version detection used during authenticate()."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def _assert_help_or_version(self, argv, expected):
        with patch.object(sys, "argv", argv):
            api_command = " ".join(argv[1:])
            self.assertEqual(self.api._is_help_or_version_command(api_command), expected)

    def test_top_level_version_and_help(self):
        self._assert_help_or_version(["kaggle", "-v"], True)
        self._assert_help_or_version(["kaggle", "--version"], True)
        self._assert_help_or_version(["kaggle", "-h"], True)
        self._assert_help_or_version(["kaggle", "--help"], True)

    def test_subcommand_csv_flag_is_not_version(self):
        self._assert_help_or_version(["kaggle", "quota", "-v"], False)
        self._assert_help_or_version(["kaggle", "datasets", "list", "-v"], False)

    def test_subcommand_help_still_skips_auth(self):
        self._assert_help_or_version(["kaggle", "quota", "-h"], True)
        self._assert_help_or_version(["kaggle", "datasets", "list", "--help"], True)

    def test_quota_csv_flag_does_not_allow_logged_out(self):
        with patch.object(sys, "argv", ["kaggle", "quota", "-v"]):
            self.assertFalse(self.api._command_allows_logged_out("quota -v"))

    def _assert_anonymous(self, argv, expected):
        with patch.object(sys, "argv", argv):
            self.assertEqual(self.api._authenticate_anonymously(), expected)

    # --- CLI invocation: help/version must bypass authentication ---

    def test_cli_help_and_version_bypass_auth(self):
        # Installed console entry point (`kaggle` / `kaggle.exe`).
        self._assert_anonymous(["kaggle", "-h"], True)
        self._assert_anonymous(["kaggle", "--help"], True)
        self._assert_anonymous(["kaggle", "-v"], True)
        self._assert_anonymous(["kaggle", "--version"], True)
        self._assert_anonymous(["/usr/local/bin/kaggle", "-h"], True)
        self._assert_anonymous(["C:\\Python\\Scripts\\kaggle.exe", "--version"], True)

    def test_python_dash_m_kaggle_help_bypasses_auth(self):
        # `python -m kaggle` runs kaggle/__main__.py.
        self._assert_anonymous(["/site-packages/kaggle/__main__.py", "-h"], True)
        self._assert_anonymous(["C:\\site-packages\\kaggle\\__main__.py", "--version"], True)

    # --- Host script importing the library: never bypass auth via host args ---

    def test_plain_python_script_does_not_bypass_auth(self):
        # `python my_script.py` (no flags) never allowed anonymous anyway.
        self._assert_anonymous(["my_script.py"], False)

    def test_host_script_help_version_flags_do_not_bypass_auth(self):
        # `python my_script.py -h/--help/-v/--version` must NOT be read as a
        # Kaggle CLI help/version command.
        self._assert_anonymous(["my_script.py", "-h"], False)
        self._assert_anonymous(["my_script.py", "--help"], False)
        self._assert_anonymous(["my_script.py", "-v"], False)
        self._assert_anonymous(["my_script.py", "--version"], False)
        self._assert_anonymous(["/home/user/my_script.py", "--help"], False)

    def test_normal_import_without_argv_flags_does_not_bypass_auth(self):
        # A plain `import kaggle` from an interpreter/host with unrelated argv.
        self._assert_anonymous(["/home/user/app.py", "serve", "--port", "8080"], False)

    def test_other_module_dash_m_is_not_kaggle_cli(self):
        # A different `python -m otherpkg` whose __main__.py imports kaggle.
        self._assert_anonymous(["/site-packages/otherpkg/__main__.py", "-h"], False)


if __name__ == "__main__":
    unittest.main()
