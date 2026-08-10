# coding=utf-8
from unittest.mock import MagicMock

import pytest
from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionResponse


def test_kernels_cancel_calls_the_supported_session_endpoint(api):
    """The CLI adapter must not depend on the undocumented editor API."""
    response = ApiCancelKernelSessionResponse()
    kaggle_client = MagicMock()
    kaggle_client.kernels.kernels_api_client.cancel_kernel_session.return_value = response
    api.build_kaggle_client.return_value.__enter__.return_value = kaggle_client

    result = api.kernels_cancel(123)

    assert result is response
    request = kaggle_client.kernels.kernels_api_client.cancel_kernel_session.call_args.args[0]
    assert request.kernel_session_id == 123


def test_kernels_cancel_rejects_a_non_positive_session_id(api):
    """Fail locally before a malformed cancellation request reaches Kaggle."""
    with pytest.raises(ValueError, match="positive"):
        api.kernels_cancel(0)

    api.build_kaggle_client.assert_not_called()


def test_kernels_cancel_cli_reports_the_server_error(api, capsys, monkeypatch):
    """The operator receives the cancel endpoint's actionable error text."""
    response = ApiCancelKernelSessionResponse()
    response.error_message = "The session is already complete."
    monkeypatch.setattr(api, "kernels_cancel", MagicMock(return_value=response))

    api.kernels_cancel_cli(123)

    assert capsys.readouterr().out == "Cancel failed: The session is already complete.\n"


def test_kernels_cancel_cli_confirms_the_requested_session(api, capsys, monkeypatch):
    """Success output remains scoped to the numeric session the user selected."""
    monkeypatch.setattr(api, "kernels_cancel", MagicMock(return_value=ApiCancelKernelSessionResponse()))

    api.kernels_cancel_cli(123)

    assert capsys.readouterr().out == "Cancellation requested for kernel session 123.\n"
