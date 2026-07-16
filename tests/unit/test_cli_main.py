# coding=utf-8
import io
import sys
from unittest.mock import MagicMock, patch
import pytest
from requests.exceptions import HTTPError


@pytest.fixture
def mock_api(monkeypatch):
    from kaggle.api.kaggle_api_extended import KaggleApi

    mock_api = MagicMock(spec=KaggleApi)
    mock_api._authenticated = True

    import kaggle

    monkeypatch.setattr(kaggle, "api", mock_api)

    import kaggle.cli

    monkeypatch.setattr(kaggle.cli, "api", mock_api)
    return mock_api


def test_main_valid_command_succeeds(monkeypatch, mock_api):
    monkeypatch.setattr(sys, "argv", ["kaggle", "quota"])
    mock_api.quota_view_cli.return_value = "mocked quota output"

    from kaggle.cli import main

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    main()

    assert "mocked quota output" in captured.getvalue()
    mock_api.quota_view_cli.assert_called_once()
    mock_api.authenticate.assert_not_called()


def test_main_unauthenticated_calls_authenticate(monkeypatch, mock_api):
    mock_api._authenticated = False
    monkeypatch.setattr(sys, "argv", ["kaggle", "quota"])
    mock_api.quota_view_cli.return_value = "output"

    from kaggle.cli import main

    with patch("sys.stdout", new=io.StringIO()):
        main()

    mock_api.authenticate.assert_called_once()


def test_main_with_warning_disable_flag_succeeds(monkeypatch, mock_api):
    monkeypatch.setattr(sys, "argv", ["kaggle", "-W", "quota"])
    mock_api.quota_view_cli.return_value = "output"

    from kaggle.cli import main
    from kaggle.api.kaggle_api_extended import KaggleApi

    KaggleApi.already_printed_version_warning = False

    with patch("sys.stdout", new=io.StringIO()):
        main()

    assert KaggleApi.already_printed_version_warning is True


def test_main_http_error_401_exits_with_error(monkeypatch, mock_api):
    monkeypatch.setattr(sys, "argv", ["kaggle", "quota"])

    response = MagicMock()
    response.status_code = 401
    error = HTTPError(response=response)
    mock_api.quota_view_cli.side_effect = error

    from kaggle.cli import main

    captured_out = io.StringIO()
    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", captured_err)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    assert "Authentication required" in captured_out.getvalue()


def test_main_http_error_generic_exits_with_error(monkeypatch, mock_api):
    monkeypatch.setattr(sys, "argv", ["kaggle", "quota"])

    response = MagicMock()
    response.status_code = 500
    error = HTTPError("Internal Server Error", response=response)
    mock_api.quota_view_cli.side_effect = error

    from kaggle.cli import main

    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_err)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    assert "Internal Server Error" in captured_err.getvalue()


def test_main_value_error_exits_with_error(monkeypatch, mock_api):
    monkeypatch.setattr(sys, "argv", ["kaggle", "quota"])
    mock_api.quota_view_cli.side_effect = ValueError("Invalid argument")

    from kaggle.cli import main

    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_err)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    assert "Invalid argument" in captured_err.getvalue()


def test_main_keyboard_interrupt_prints_cancelled_message(monkeypatch, mock_api):
    monkeypatch.setattr(sys, "argv", ["kaggle", "quota"])
    mock_api.quota_view_cli.side_effect = KeyboardInterrupt()

    from kaggle.cli import main

    captured_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)

    main()

    assert "User cancelled operation" in captured_out.getvalue()


def test_main_api_exception_exits_with_error(monkeypatch, mock_api):
    monkeypatch.setattr(sys, "argv", ["kaggle", "quota"])
    mock_api.quota_view_cli.side_effect = IOError("API Error")

    from kaggle.cli import main

    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_err)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    assert "API Error" in captured_err.getvalue()


def test_parse_body_valid_json_returns_dict():
    from kaggle.cli import __parse_body

    assert __parse_body('{"a": 1}') == {"a": 1}


def test_parse_body_invalid_json_returns_empty_dict():
    from kaggle.cli import __parse_body

    assert __parse_body("invalid json") == {}
