# coding=utf-8
"""Tests for `competitions submit --wait` and `competitions submission <ref>`.

Covers the parser wiring, the submission polling helper (_poll_submission),
the submit-with-wait flow, the single-submission display command, and the
end-to-end CLI exit codes (0 on success, 1 on scoring failure).
"""

import io
import sys
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiSubmission
from kagglesdk.competitions.types.submission_status import SubmissionStatus


def _api():
    """A bare KaggleApi instance without running __init__/authenticate."""
    api = KaggleApi.__new__(KaggleApi)
    api.config_values = {"username": "testuser"}
    api.already_printed_version_warning = True
    return api


def _submission(status, ref=12345, public="0.987", private="0.812", desc="my run", error=None):
    s = ApiSubmission()
    s.ref = ref
    s.status = status
    if public is not None:
        s.public_score = public
    if private is not None:
        s.private_score = private
    if desc is not None:
        s.description = desc
    if error is not None:
        s.error_description = error
    return s


def _http_404():
    return HTTPError(response=MagicMock(status_code=404))


# --------------------------------------------------------------------------
# Parser tests
# --------------------------------------------------------------------------
def test_submit_parser_wait_flag_bare(parser):
    func, kwargs = parser.dispatch(["competitions", "submit", "my-comp", "-f", "sub.csv", "-m", "msg", "--wait"])
    assert func.__name__ == "competition_submit_cli"
    # const: bare --wait parses to 0; competition_submit_cli resolves 0 to the 12h default.
    assert kwargs["wait"] == 0
    assert kwargs["poll_interval"] == 60


def test_submit_parser_wait_with_timeout(parser):
    func, kwargs = parser.dispatch(["competitions", "submit", "my-comp", "-f", "sub.csv", "-m", "msg", "--wait", "300"])
    assert func.__name__ == "competition_submit_cli"
    assert kwargs["wait"] == 300
    assert kwargs["poll_interval"] == 60


def test_submit_parser_poll_interval(parser):
    func, kwargs = parser.dispatch(
        ["competitions", "submit", "my-comp", "-f", "sub.csv", "-m", "msg", "--wait", "--poll-interval", "5"]
    )
    assert func.__name__ == "competition_submit_cli"
    assert kwargs["wait"] == 0
    assert kwargs["poll_interval"] == 5


def test_submit_parser_no_wait_default(parser):
    func, kwargs = parser.dispatch(["competitions", "submit", "my-comp", "-f", "sub.csv", "-m", "msg"])
    assert func.__name__ == "competition_submit_cli"
    assert kwargs["wait"] is None  # absent → do not wait (backward compatible)
    assert kwargs["poll_interval"] == 60


def test_submission_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "submission", "12345"])
    assert func.__name__ == "competition_submission_cli"
    assert kwargs["submission_ref"] == "12345"


# --------------------------------------------------------------------------
# _poll_submission behavior
# --------------------------------------------------------------------------
def test_poll_submission_completes_and_prints_score():
    api = _api()
    api._adaptive_sleep = MagicMock(return_value=60)  # no real sleeping
    api.get_submission = MagicMock(
        side_effect=[
            _submission(SubmissionStatus.PENDING),
            _submission(SubmissionStatus.PENDING),
            _submission(SubmissionStatus.COMPLETE, public="0.987"),
        ]
    )

    out = io.StringIO()
    with redirect_stdout(out):
        result = api._poll_submission(12345, wait=0, poll_interval=60)

    assert result.status == SubmissionStatus.COMPLETE
    assert api.get_submission.call_count == 3  # polled until terminal
    assert api._adaptive_sleep.call_count == 2  # slept between the two PENDING polls
    printed = out.getvalue()
    assert "Public score: 0.987" in printed
    assert printed.count("Submission status: PENDING...") == 2


def test_poll_submission_error_raises_with_description():
    api = _api()
    api._adaptive_sleep = MagicMock(return_value=60)
    api.get_submission = MagicMock(
        side_effect=[
            _submission(SubmissionStatus.PENDING),
            _submission(SubmissionStatus.ERROR, error="Evaluation Exception: wrong number of rows"),
        ]
    )

    with pytest.raises(ValueError) as ctx:
        api._poll_submission(12345, wait=0, poll_interval=60, quiet=True)

    msg = str(ctx.value)
    assert "failed to score" in msg
    assert "wrong number of rows" in msg


def test_poll_submission_timeout_raises_with_resume_hint():
    api = _api()
    api._adaptive_sleep = MagicMock(return_value=60)
    api.get_submission = MagicMock(return_value=_submission(SubmissionStatus.PENDING))

    # start_time=1000, then elapsed check sees 2000 → exceeds wait=10 → timeout.
    with patch("kaggle.api.kaggle_api_extended.time.time", side_effect=[1000.0, 2000.0]):
        with pytest.raises(ValueError) as ctx:
            api._poll_submission(12345, wait=10, poll_interval=60, quiet=True)

    msg = str(ctx.value)
    assert "Timed out after 10s" in msg
    assert "Submission ref: 12345" in msg
    assert "kaggle competitions submission 12345" in msg


def test_poll_submission_transient_404_treated_as_pending():
    api = _api()
    api._adaptive_sleep = MagicMock(return_value=60)
    api.get_submission = MagicMock(
        side_effect=[
            _http_404(),  # eventual consistency: not visible yet
            _http_404(),
            _submission(SubmissionStatus.PENDING),
            _submission(SubmissionStatus.COMPLETE),
        ]
    )

    result = api._poll_submission(12345, wait=0, poll_interval=60, quiet=True)

    assert result.status == SubmissionStatus.COMPLETE
    assert api.get_submission.call_count == 4  # kept polling through the 404s


def test_poll_submission_non_404_http_error_propagates():
    api = _api()
    api._adaptive_sleep = MagicMock(return_value=60)
    api.get_submission = MagicMock(side_effect=HTTPError(response=MagicMock(status_code=500)))

    with pytest.raises(HTTPError):
        api._poll_submission(12345, wait=0, poll_interval=60, quiet=True)


def test_poll_submission_keyboard_interrupt_prints_hint_and_reraises():
    api = _api()
    api.get_submission = MagicMock(return_value=_submission(SubmissionStatus.PENDING))
    api._adaptive_sleep = MagicMock(side_effect=KeyboardInterrupt())

    out = io.StringIO()
    with redirect_stdout(out):
        with pytest.raises(KeyboardInterrupt):
            api._poll_submission(12345, wait=0, poll_interval=60, quiet=True)

    printed = out.getvalue()
    assert "Stopped waiting for submission 12345" in printed
    assert "kaggle competitions submission 12345" in printed


# --------------------------------------------------------------------------
# competition_submit_cli integration with --wait
# --------------------------------------------------------------------------
def test_submit_cli_prints_ref_without_wait():
    api = _api()
    api.competition_submit = MagicMock(return_value=MagicMock(ref=999, message="Successfully submitted"))
    api._poll_submission = MagicMock()

    out = io.StringIO()
    with redirect_stdout(out):
        result = api.competition_submit_cli(file_name="sub.csv", message="m", competition="comp", quiet=True)

    assert "Submission ref: 999" in out.getvalue()
    api._poll_submission.assert_not_called()  # no --wait → do not poll
    assert result == "Successfully submitted"


def test_submit_cli_wait_default_timeout_is_12h():
    # A bare --wait (wait=0) is resolved to the 12h maximum notebook runtime, not infinite.
    api = _api()
    api.competition_submit = MagicMock(return_value=MagicMock(ref=12345, message="Successfully submitted"))
    api._poll_submission = MagicMock()

    out = io.StringIO()
    with redirect_stdout(out):
        result = api.competition_submit_cli(
            file_name="sub.csv", message="m", competition="comp", quiet=False, wait=0, poll_interval=60
        )

    assert "Submission ref: 12345" in out.getvalue()
    # 0 → 12h (43200s); poll_interval passed through unchanged.
    api._poll_submission.assert_called_once_with(12345, 43200, 60, quiet=False)
    assert result == "Successfully submitted"


def test_submit_cli_wait_explicit_timeout_passed_through():
    # An explicit positive timeout is forwarded verbatim (no 12h substitution).
    api = _api()
    api.competition_submit = MagicMock(return_value=MagicMock(ref=12345, message="Successfully submitted"))
    api._poll_submission = MagicMock()

    with redirect_stdout(io.StringIO()):
        api.competition_submit_cli(
            file_name="sub.csv", message="m", competition="comp", quiet=False, wait=300, poll_interval=60
        )

    api._poll_submission.assert_called_once_with(12345, 300, 60, quiet=False)


def test_submit_cli_poll_interval_below_minimum_fails_fast():
    api = _api()
    api.competition_submit = MagicMock()

    # 3s is below the 5s minimum poll interval.
    with pytest.raises(ValueError) as ctx:
        api.competition_submit_cli(file_name="sub.csv", message="m", competition="comp", wait=0, poll_interval=3)

    assert "--poll-interval must be at least 5s" in str(ctx.value)
    api.competition_submit.assert_not_called()  # validated before submitting


def test_submit_cli_poll_interval_at_minimum_is_accepted():
    # The 5s boundary is valid and reaches polling.
    api = _api()
    api.competition_submit = MagicMock(return_value=MagicMock(ref=12345, message="Successfully submitted"))
    api._poll_submission = MagicMock()

    with redirect_stdout(io.StringIO()):
        api.competition_submit_cli(
            file_name="sub.csv", message="m", competition="comp", quiet=False, wait=600, poll_interval=5
        )

    api._poll_submission.assert_called_once_with(12345, 600, 5, quiet=False)


def test_submit_cli_wait_propagates_scoring_error():
    api = _api()
    api.competition_submit = MagicMock(return_value=MagicMock(ref=12345, message="Successfully submitted"))
    api._poll_submission = MagicMock(side_effect=ValueError("Submission 12345 failed to score."))

    with pytest.raises(ValueError):
        api.competition_submit_cli(file_name="sub.csv", message="m", competition="comp", wait=0, poll_interval=60)


# --------------------------------------------------------------------------
# competition_submission_cli display command
# --------------------------------------------------------------------------
def test_submission_cli_displays_fields():
    api = _api()
    api.get_submission = MagicMock(
        return_value=_submission(SubmissionStatus.COMPLETE, ref=12345, public="0.987", private="0.812", desc="my run")
    )

    out = io.StringIO()
    with redirect_stdout(out):
        api.competition_submission_cli("12345")

    printed = out.getvalue()
    for label in ("Submission Ref", "Status", "Public Score", "Private Score", "Description", "Submission Date"):
        assert label in printed
    assert "COMPLETE" in printed
    assert "12345" in printed
    assert "0.987" in printed
    assert "0.812" in printed
    assert "my run" in printed
    api.get_submission.assert_called_once_with("12345")


def test_submission_cli_shows_the_scoring_error():
    api = _api()
    api.get_submission = MagicMock(
        return_value=_submission(
            SubmissionStatus.ERROR,
            public=None,
            private=None,
            error="Evaluation Exception: Submission must have 418 rows",
        )
    )

    out = io.StringIO()
    with redirect_stdout(out):
        api.competition_submission_cli("12345")

    printed = out.getvalue()
    assert "Error:" in printed
    assert "Evaluation Exception: Submission must have 418 rows" in printed


def test_submission_cli_aligns_a_multi_line_scoring_error():
    api = _api()
    error = (
        "ERROR: Unexpected Column:  'NotAnId' (Line 1, Column 1)\n"
        "ERROR: Required column 'PassengerId' could not be found"
    )
    api.get_submission = MagicMock(
        return_value=_submission(SubmissionStatus.ERROR, public=None, private=None, error=error)
    )

    out = io.StringIO()
    with redirect_stdout(out):
        api.competition_submission_cli("12345")

    lines = out.getvalue().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("Error:"))
    indent = lines[first].index("ERROR: Unexpected")
    # The continuation line starts in the same column as the first, not at the margin.
    assert lines[first + 1].startswith(" " * indent)
    assert lines[first + 1].strip() == "ERROR: Required column 'PassengerId' could not be found"


def test_submission_cli_omits_the_error_row_when_there_is_none():
    api = _api()
    api.get_submission = MagicMock(return_value=_submission(SubmissionStatus.COMPLETE))

    out = io.StringIO()
    with redirect_stdout(out):
        api.competition_submission_cli("12345")

    printed = out.getvalue()
    assert "Error:" not in printed
    assert "COMPLETE" in printed


def test_submission_cli_ignores_a_blank_error_description():
    api = _api()
    api.get_submission = MagicMock(return_value=_submission(SubmissionStatus.ERROR, error="   "))

    out = io.StringIO()
    with redirect_stdout(out):
        api.competition_submission_cli("12345")

    assert "Error:" not in out.getvalue()


def test_submission_cli_missing_ref_fails():
    api = _api()
    with pytest.raises(ValueError) as ctx:
        api.competition_submission_cli(None)
    assert "A submission ref must be specified" in str(ctx.value)


# --------------------------------------------------------------------------
# End-to-end CLI exit codes via main()
# --------------------------------------------------------------------------
def _run_main(monkeypatch, api, argv):
    """Run kaggle.cli.main() with the given argv; return the process exit code."""
    import kaggle.cli as cli

    monkeypatch.setattr(cli, "api", api)
    monkeypatch.setattr(api, "authenticate", MagicMock())
    monkeypatch.setattr(api, "_authenticated", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["kaggle", *argv])
    try:
        cli.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def test_cli_submit_wait_success_exits_zero(monkeypatch, api):
    api.competition_submit = MagicMock(return_value=MagicMock(ref=12345, message="Successfully submitted"))
    api.get_submission = MagicMock(return_value=_submission(SubmissionStatus.COMPLETE, public="0.987"))
    api._adaptive_sleep = MagicMock(return_value=60)

    out = io.StringIO()
    with redirect_stdout(out):
        code = _run_main(
            monkeypatch,
            api,
            ["competitions", "submit", "comp", "-f", "sub.csv", "-m", "msg", "--wait"],
        )

    assert code == 0
    assert "Public score: 0.987" in out.getvalue()


def test_cli_submit_wait_scoring_error_exits_one(monkeypatch, api):
    api.competition_submit = MagicMock(return_value=MagicMock(ref=12345, message="Successfully submitted"))
    api.get_submission = MagicMock(return_value=_submission(SubmissionStatus.ERROR, error="Evaluation failed"))
    api._adaptive_sleep = MagicMock(return_value=60)

    err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with patch("sys.stderr", err):
            code = _run_main(
                monkeypatch,
                api,
                ["competitions", "submit", "comp", "-f", "sub.csv", "-m", "msg", "--wait"],
            )

    assert code == 1
    assert "failed to score" in err.getvalue()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
