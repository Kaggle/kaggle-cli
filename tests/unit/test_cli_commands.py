# coding=utf-8
import pytest


# Task 1.1: Quota
def test_quota_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["quota"])
    assert func.__name__ == "quota_view_cli"
    assert kwargs.get("csv_display") is False
    assert kwargs.get("output_format") is None


def test_quota_parser_csv_succeeds(parser):
    func, kwargs = parser.dispatch(["quota", "--csv"])
    assert func.__name__ == "quota_view_cli"
    assert kwargs["csv_display"] is True
    assert kwargs.get("output_format") is None


def test_quota_parser_format_json_succeeds(parser):
    func, kwargs = parser.dispatch(["quota", "--format", "json"])
    assert func.__name__ == "quota_view_cli"
    assert kwargs["csv_display"] is False
    assert kwargs["output_format"] == "json"


# Task 1.2: Config
def test_config_view_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["config", "view"])
    assert func.__name__ == "print_config_values"
    assert kwargs == {}


def test_config_set_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["config", "set", "-n", "proxy", "-v", "http://proxy"])
    assert func.__name__ == "set_config_value"
    assert kwargs["name"] == "proxy"
    assert kwargs["value"] == "http://proxy"


def test_config_unset_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["config", "unset", "-n", "proxy"])
    assert func.__name__ == "unset_config_value"
    assert kwargs["name"] == "proxy"


# Task 1.3: Auth
def test_auth_login_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["auth", "login"])
    assert func.__name__ == "auth_login_cli"
    assert kwargs["no_launch_browser"] is False
    assert kwargs["force"] is False


def test_auth_login_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["auth", "login", "--no-launch-browser", "--force"])
    assert func.__name__ == "auth_login_cli"
    assert kwargs["no_launch_browser"] is True
    assert kwargs["force"] is True


def test_auth_print_access_token_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["auth", "print-access-token"])
    assert func.__name__ == "auth_print_access_token"
    assert kwargs["expiration_duration"] is None


def test_auth_print_access_token_parser_with_expiration_succeeds(parser):
    func, kwargs = parser.dispatch(["auth", "print-access-token", "--expiration", "6h"])
    assert func.__name__ == "auth_print_access_token"
    assert kwargs["expiration_duration"] == "6h"


def test_auth_revoke_token_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["auth", "revoke"])
    assert func.__name__ == "auth_revoke_token"
    assert kwargs["reason"] is None


def test_auth_revoke_token_parser_with_reason_succeeds(parser):
    func, kwargs = parser.dispatch(["auth", "revoke", "--reason", "compromised"])
    assert func.__name__ == "auth_revoke_token"
    assert kwargs["reason"] == "compromised"


# Task 1.4: Files
def test_files_upload_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["files", "upload", "file1.zip", "file2.txt"])
    assert func.__name__ == "files_upload_cli"
    assert kwargs["local_paths"] == ["file1.zip", "file2.txt"]
    assert kwargs["inbox_path"] == ""
    assert kwargs["no_resume"] is False
    assert kwargs["no_compress"] is False


def test_files_upload_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["files", "upload", "file1.zip", "-i", "my_inbox", "--no-resume", "--no-compress"])
    assert func.__name__ == "files_upload_cli"
    assert kwargs["local_paths"] == ["file1.zip"]
    assert kwargs["inbox_path"] == "my_inbox"
    assert kwargs["no_resume"] is True
    assert kwargs["no_compress"] is True
