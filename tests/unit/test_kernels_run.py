# coding=utf-8
"""Tests for `kernels run` (kernels_run / _poll_kernel_session).

Covers the session polling helper (terminal states, transient 404s, the
stale-baseline guard, timeouts), the push-then-wait flow, output download
wiring, argument validation, and the owner/slug resolution fallbacks.

The Kaggle API only reports a kernel's *latest* session (version-pinned
status/output lookups are not supported by the backend), so the poll guards
against reading the previous session's terminal status right after a push by
comparing against a pre-push baseline status.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from kaggle.api.kaggle_api_extended import KaggleApi, KernelRunResult
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusResponse
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus


def _api():
    """A bare KaggleApi instance without running __init__/authenticate."""
    api = KaggleApi.__new__(KaggleApi)
    api.config_values = {"username": "testuser"}
    api.already_printed_version_warning = True
    return api


def _status(status, failure_message=None):
    response = ApiGetKernelSessionStatusResponse()
    response.status = status
    if failure_message is not None:
        response.failure_message = failure_message
    return response


def _http_error(code):
    return HTTPError(response=MagicMock(status_code=code))


def _no_sleep():
    """Patch the adaptive sleep so polls advance instantly."""
    return patch.object(KaggleApi, "_adaptive_sleep", side_effect=lambda current, cap, verbose=False: cap)


def _push_response(version_number=7, url="https://www.kaggle.com/code/testuser/my-kernel", error=None):
    response = MagicMock()
    response.error = error
    response.version_number = version_number
    response.url = url
    return response


# --------------------------------------------------------------------------
# _poll_kernel_session
# --------------------------------------------------------------------------
def test_poll_completes_after_queued_and_running():
    api = _api()
    statuses = [
        _status(KernelWorkerStatus.QUEUED),
        _status(KernelWorkerStatus.RUNNING),
        _status(KernelWorkerStatus.COMPLETE),
    ]
    with patch.object(KaggleApi, "kernels_status", side_effect=statuses) as status_mock, _no_sleep():
        response = api._poll_kernel_session("testuser/my-kernel", wait=0, poll_interval=30, quiet=True)
    assert response.status == KernelWorkerStatus.COMPLETE
    assert status_mock.call_count == 3


def test_poll_error_raises_with_failure_message():
    api = _api()
    statuses = [
        _status(KernelWorkerStatus.RUNNING),
        _status(KernelWorkerStatus.ERROR, failure_message="Boom"),
    ]
    with patch.object(KaggleApi, "kernels_status", side_effect=statuses), _no_sleep():
        with pytest.raises(ValueError, match="Boom"):
            api._poll_kernel_session("testuser/my-kernel", wait=0, poll_interval=30, quiet=True)


def test_poll_error_without_message_uses_fallback():
    api = _api()
    with patch.object(KaggleApi, "kernels_status", side_effect=[_status(KernelWorkerStatus.ERROR)]), _no_sleep():
        with pytest.raises(ValueError, match="No error message provided"):
            api._poll_kernel_session("testuser/my-kernel", wait=0, poll_interval=30, quiet=True)


def test_poll_cancel_acknowledged_raises():
    api = _api()
    statuses = [
        _status(KernelWorkerStatus.CANCEL_REQUESTED),
        _status(KernelWorkerStatus.CANCEL_ACKNOWLEDGED),
    ]
    with patch.object(KaggleApi, "kernels_status", side_effect=statuses), _no_sleep():
        with pytest.raises(ValueError, match="cancelled"):
            api._poll_kernel_session("testuser/my-kernel", wait=0, poll_interval=30, quiet=True)


def test_poll_transient_404_treated_as_pending():
    api = _api()
    statuses = [_http_error(404), _status(KernelWorkerStatus.RUNNING), _status(KernelWorkerStatus.COMPLETE)]
    with patch.object(KaggleApi, "kernels_status", side_effect=statuses) as status_mock, _no_sleep():
        response = api._poll_kernel_session("testuser/my-kernel", wait=0, poll_interval=30, quiet=True)
    assert response.status == KernelWorkerStatus.COMPLETE
    assert status_mock.call_count == 3


def test_poll_non_404_http_error_propagates():
    api = _api()
    with patch.object(KaggleApi, "kernels_status", side_effect=[_http_error(500)]), _no_sleep():
        with pytest.raises(HTTPError):
            api._poll_kernel_session("testuser/my-kernel", wait=0, poll_interval=30, quiet=True)


def test_poll_timeout_raises_with_recovery_hint():
    api = _api()
    with patch.object(KaggleApi, "kernels_status", return_value=_status(KernelWorkerStatus.RUNNING)), _no_sleep():
        with pytest.raises(ValueError, match="kaggle kernels status testuser/my-kernel"):
            api._poll_kernel_session("testuser/my-kernel", wait=1e-9, poll_interval=30, quiet=True)


def test_poll_stale_baseline_terminal_is_not_accepted():
    """A COMPLETE identical to the pre-push status must not be trusted until
    an active status proves the new session is the one being observed."""
    api = _api()
    statuses = [
        _status(KernelWorkerStatus.COMPLETE),  # stale: previous session
        _status(KernelWorkerStatus.COMPLETE),  # still stale
        _status(KernelWorkerStatus.RUNNING),  # new session visible
        _status(KernelWorkerStatus.COMPLETE),  # this run's result
    ]
    with patch.object(KaggleApi, "kernels_status", side_effect=statuses) as status_mock, _no_sleep():
        response = api._poll_kernel_session(
            "testuser/my-kernel",
            wait=0,
            poll_interval=30,
            quiet=True,
            baseline_status=KernelWorkerStatus.COMPLETE,
        )
    assert response.status == KernelWorkerStatus.COMPLETE
    assert status_mock.call_count == 4


def test_poll_stale_guard_expires_after_grace():
    api = _api()
    api._SESSION_START_GRACE_SECONDS = 0  # disable the guard window
    with patch.object(KaggleApi, "kernels_status", side_effect=[_status(KernelWorkerStatus.COMPLETE)]), _no_sleep():
        response = api._poll_kernel_session(
            "testuser/my-kernel",
            wait=0,
            poll_interval=30,
            quiet=True,
            baseline_status=KernelWorkerStatus.COMPLETE,
        )
    assert response.status == KernelWorkerStatus.COMPLETE


def test_poll_terminal_differing_from_baseline_accepted_immediately():
    api = _api()
    statuses = [_status(KernelWorkerStatus.ERROR, failure_message="new failure")]
    with patch.object(KaggleApi, "kernels_status", side_effect=statuses), _no_sleep():
        with pytest.raises(ValueError, match="new failure"):
            api._poll_kernel_session(
                "testuser/my-kernel",
                wait=0,
                poll_interval=30,
                quiet=True,
                baseline_status=KernelWorkerStatus.COMPLETE,
            )


def test_poll_without_baseline_accepts_first_terminal():
    api = _api()
    with patch.object(KaggleApi, "kernels_status", side_effect=[_status(KernelWorkerStatus.COMPLETE)]), _no_sleep():
        response = api._poll_kernel_session("testuser/my-kernel", wait=0, poll_interval=30, quiet=True)
    assert response.status == KernelWorkerStatus.COMPLETE


# --------------------------------------------------------------------------
# kernels_run
# --------------------------------------------------------------------------
def test_run_happy_path_with_output():
    api = _api()
    with (
        patch.object(KaggleApi, "_kernel_ref_from_folder", return_value=("testuser", "my-kernel")),
        patch.object(KaggleApi, "kernels_status", return_value=_status(KernelWorkerStatus.COMPLETE)),
        patch.object(KaggleApi, "kernels_push", return_value=_push_response()) as push_mock,
        patch.object(KaggleApi, "_poll_kernel_session", return_value=_status(KernelWorkerStatus.COMPLETE)) as poll_mock,
        patch.object(KaggleApi, "kernels_output", return_value=(["/tmp/out/sub.csv"], None)) as output_mock,
    ):
        result = api.kernels_run("/tmp/folder", output_path="/tmp/out", quiet=True)

    push_mock.assert_called_once_with("/tmp/folder", None, None)
    # The pre-push status is passed to the poll as the stale-detection baseline.
    assert poll_mock.call_args.kwargs["baseline_status"] == KernelWorkerStatus.COMPLETE
    assert poll_mock.call_args.args[0] == "testuser/my-kernel"
    output_mock.assert_called_once()
    assert output_mock.call_args.args[0] == "testuser/my-kernel"
    assert output_mock.call_args.args[1] == "/tmp/out"
    assert isinstance(result, KernelRunResult)
    assert result.status == KernelWorkerStatus.COMPLETE
    assert result.version_number == 7
    assert result.ref == "testuser/my-kernel"
    assert result.output_files == ["/tmp/out/sub.csv"]
    assert result.elapsed_seconds is not None


def test_run_without_output_path_skips_download():
    api = _api()
    with (
        patch.object(KaggleApi, "_kernel_ref_from_folder", return_value=("testuser", "my-kernel")),
        patch.object(KaggleApi, "kernels_status", return_value=_status(KernelWorkerStatus.COMPLETE)),
        patch.object(KaggleApi, "kernels_push", return_value=_push_response()),
        patch.object(KaggleApi, "_poll_kernel_session", return_value=_status(KernelWorkerStatus.COMPLETE)),
        patch.object(KaggleApi, "kernels_output") as output_mock,
    ):
        result = api.kernels_run("/tmp/folder", quiet=True)
    output_mock.assert_not_called()
    assert result.output_files == []


def test_run_push_error_short_circuits():
    api = _api()
    with (
        patch.object(KaggleApi, "_kernel_ref_from_folder", return_value=("testuser", "my-kernel")),
        patch.object(KaggleApi, "kernels_status", return_value=_status(KernelWorkerStatus.COMPLETE)),
        patch.object(KaggleApi, "kernels_push", return_value=_push_response(error="bad metadata")),
        patch.object(KaggleApi, "_poll_kernel_session") as poll_mock,
    ):
        with pytest.raises(ValueError, match="bad metadata"):
            api.kernels_run("/tmp/folder", quiet=True)
    poll_mock.assert_not_called()


def test_run_baseline_read_failure_is_tolerated():
    api = _api()
    with (
        patch.object(KaggleApi, "_kernel_ref_from_folder", return_value=("testuser", "my-kernel")),
        patch.object(KaggleApi, "kernels_status", side_effect=_http_error(404)),
        patch.object(KaggleApi, "kernels_push", return_value=_push_response()),
        patch.object(KaggleApi, "_poll_kernel_session", return_value=_status(KernelWorkerStatus.COMPLETE)) as poll_mock,
    ):
        api.kernels_run("/tmp/folder", quiet=True)
    assert poll_mock.call_args.kwargs["baseline_status"] is None


def test_run_resolves_ref_from_push_url_when_metadata_has_no_id():
    api = _api()
    push = _push_response(url="https://www.kaggle.com/code/owner2/slug2")
    with (
        patch.object(KaggleApi, "_kernel_ref_from_folder", return_value=None),
        patch.object(KaggleApi, "kernels_status") as status_mock,
        patch.object(KaggleApi, "kernels_push", return_value=push),
        patch.object(KaggleApi, "_poll_kernel_session", return_value=_status(KernelWorkerStatus.COMPLETE)) as poll_mock,
    ):
        result = api.kernels_run("/tmp/folder", quiet=True)
    # No metadata id: no baseline read happens before the push.
    status_mock.assert_not_called()
    assert poll_mock.call_args.args[0] == "owner2/slug2"
    assert result.ref == "owner2/slug2"


def test_run_wait_timeout_defaults_to_twelve_hours():
    api = _api()
    for wait_timeout in (None, 0):
        with (
            patch.object(KaggleApi, "_kernel_ref_from_folder", return_value=("testuser", "my-kernel")),
            patch.object(KaggleApi, "kernels_status", return_value=_status(KernelWorkerStatus.COMPLETE)),
            patch.object(KaggleApi, "kernels_push", return_value=_push_response()),
            patch.object(
                KaggleApi, "_poll_kernel_session", return_value=_status(KernelWorkerStatus.COMPLETE)
            ) as poll_mock,
        ):
            api.kernels_run("/tmp/folder", wait_timeout=wait_timeout, quiet=True)
        assert poll_mock.call_args.args[1] == KaggleApi._DEFAULT_WAIT_TIMEOUT


def test_run_file_pattern_requires_output():
    with pytest.raises(ValueError, match="--file-pattern requires --output"):
        _api().kernels_run("/tmp/folder", file_pattern="csv$")


def test_run_force_requires_output():
    with pytest.raises(ValueError, match="--force requires --output"):
        _api().kernels_run("/tmp/folder", force=True)


def test_run_poll_interval_minimum_enforced():
    with pytest.raises(ValueError, match="--poll-interval must be at least"):
        _api().kernels_run("/tmp/folder", poll_interval=1)


def test_run_negative_wait_timeout_rejected():
    with pytest.raises(ValueError, match="--wait-timeout cannot be negative"):
        _api().kernels_run("/tmp/folder", wait_timeout=-1)


# --------------------------------------------------------------------------
# _kernel_ref_from_folder / _kernel_ref_from_push
# --------------------------------------------------------------------------
def _write_metadata(tmp_path, meta):
    path = os.path.join(str(tmp_path), "kernel-metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f)
    return str(tmp_path)


def test_kernel_ref_from_folder_reads_id(tmp_path):
    folder = _write_metadata(tmp_path, {"id": "someuser/some-kernel"})
    assert _api()._kernel_ref_from_folder(folder) == ("someuser", "some-kernel")


def test_kernel_ref_from_folder_placeholder_returns_none(tmp_path):
    folder = _write_metadata(tmp_path, {"id": "someuser/INSERT_KERNEL_SLUG_HERE"})
    assert _api()._kernel_ref_from_folder(folder) is None


def test_kernel_ref_from_folder_missing_id_returns_none(tmp_path):
    folder = _write_metadata(tmp_path, {"id_no": 12345})
    assert _api()._kernel_ref_from_folder(folder) is None


def test_kernel_ref_from_push_falls_back_to_url(tmp_path):
    folder = _write_metadata(tmp_path, {"id_no": 12345})
    response = MagicMock()
    response.url = "https://www.kaggle.com/code/owner3/slug3"
    assert _api()._kernel_ref_from_push(folder, response) == ("owner3", "slug3")


def test_kernel_ref_from_push_without_any_ref_raises(tmp_path):
    folder = _write_metadata(tmp_path, {"id_no": 12345})
    response = MagicMock()
    response.url = ""
    with pytest.raises(ValueError, match="owner/kernel-slug"):
        _api()._kernel_ref_from_push(folder, response)


# --------------------------------------------------------------------------
# kernels_run_cli
# --------------------------------------------------------------------------
def test_run_cli_defaults_to_cwd_and_prints_output_hint(capsys):
    api = _api()
    result = KernelRunResult(status=KernelWorkerStatus.COMPLETE, version_number=7, url="u", ref="testuser/my-kernel")
    with patch.object(KaggleApi, "kernels_run", return_value=result) as run_mock:
        api.kernels_run_cli()
    assert run_mock.call_args.args[0] == os.getcwd()
    out = capsys.readouterr().out
    assert "kaggle kernels output testuser/my-kernel" in out


def test_run_cli_propagates_failure():
    api = _api()
    with patch.object(KaggleApi, "kernels_run", side_effect=ValueError("Kernel run failed")):
        with pytest.raises(ValueError, match="Kernel run failed"):
            api.kernels_run_cli(folder="/tmp/folder")
