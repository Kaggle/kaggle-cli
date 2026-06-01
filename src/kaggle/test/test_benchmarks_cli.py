"""Tests for ``kaggle benchmarks tasks`` CLI commands.

Organized by command (matching the spec):
  TestPush      – ``kaggle benchmarks tasks push <task> -f <file>``
  TestRun       – ``kaggle benchmarks tasks run <task> [-m ...] [--wait]``
  TestList      – ``kaggle benchmarks tasks list [--name-regex] [--status]``
  TestStatus    – ``kaggle benchmarks tasks status <task> [-m ...]``
  TestDownload  – ``kaggle benchmarks tasks download <task> [-m ...] [-o ...] [-s]``
  TestLog       – ``kaggle benchmarks tasks log <task> [-m ...]``
  TestDelete    – ``kaggle benchmarks tasks delete <task> [-y]``
  TestCliArgParsing – argparse wiring for all subcommands
"""

import argparse
import io
import os
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_proxy_api_service import ApiCreateDefaultModelProxyTokenResponse
from kagglesdk.benchmarks.types.benchmark_enums import (
    BenchmarkTaskRunState,
    BenchmarkTaskVersionCreationState,
)

# Short aliases for verbose enum members used throughout the tests.
QUEUED = BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_QUEUED
RUNNING = BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_RUNNING
COMPLETED = BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED
ERRORED = BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_ERRORED

RUN_QUEUED = BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_QUEUED
RUN_RUNNING = BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_RUNNING
RUN_COMPLETED = BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_COMPLETED
RUN_ERRORED = BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_ERRORED

DEFAULT_TASK_CONTENT = '@task(name="my-task")\ndef evaluate(): pass\n'


# ---- Fixtures & helpers ----


@pytest.fixture
def api():
    """A KaggleApi with mocked auth and client — no network calls."""
    a = KaggleApi()
    a.authenticate = MagicMock()
    mock_client = MagicMock()
    a.build_kaggle_client = MagicMock()
    a.build_kaggle_client.return_value.__enter__.return_value = mock_client
    # Expose internals so helpers can wire up responses.
    a._mock_client = mock_client
    a._mock_benchmarks = mock_client.benchmarks.benchmark_tasks_api_client
    return a


@pytest.fixture
def mock_token(api):
    """Pre-wire the model proxy token response for auth/init tests."""
    api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
        _make_token_response()
    )


def _write_task_file(tmp_path, content=DEFAULT_TASK_CONTENT, name="task.py"):
    """Write *content* to a .py file under *tmp_path* and return its path str."""
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def _mock_jupytext():
    """Return ``(mock_jupytext_module, context_manager)``."""
    jt = MagicMock()
    notebook = MagicMock()
    notebook.metadata = {}
    jt.reads.return_value = notebook
    jt.writes.return_value = '{"cells": []}'
    return jt, patch.dict("sys.modules", {"jupytext": jt})


def _push(api, task, filepath):
    """Call ``benchmarks_tasks_push_cli`` with jupytext mocked.

    Returns the mock jupytext module so callers can assert on calls.
    """
    jt, ctx = _mock_jupytext()
    with ctx:
        api.benchmarks_tasks_push_cli(task, filepath)
    return jt


def _make_task(slug="my-task", state=COMPLETED, create_time="2026-04-06 10:00:00", url=None, version_number=1):
    t = MagicMock()
    t.slug.task_slug = slug
    t.slug.version_number = version_number
    t.creation_state = state
    t.create_time = create_time
    t.url = url if url is not None else f"/benchmarks/{slug}"
    t.creation_error_message = ""
    return t


def _make_run_result(scheduled=True, skipped_reason=None):
    r = MagicMock()
    r.run_scheduled = scheduled
    r.benchmark_task_version_id = 1
    r.benchmark_model_version_id = 10
    r.run_skipped_reason = skipped_reason
    return r


def _make_run(
    model="gemini-pro",
    state=RUN_COMPLETED,
    run_id=1,
    start_time=None,
    end_time=None,
    error_message=None,
):
    r = MagicMock()
    r.model_version_slug = model
    r.state = state
    r.id = run_id
    r.start_time = start_time
    r.end_time = end_time
    r.error_message = error_message
    return r


def _setup_create_response(api, task_slug="my-task"):
    resp = MagicMock()
    resp.slug.task_slug = task_slug
    resp.url = f"https://kaggle.com/benchmarks/{task_slug}"
    resp.error = None
    api._mock_benchmarks.create_benchmark_task.return_value = resp


def _setup_completed_task(api, slug="my-task"):
    task = _make_task(slug=slug, state=COMPLETED)
    api._mock_benchmarks.get_benchmark_task.return_value = task


def _setup_batch_schedule(api, results):
    resp = MagicMock()
    resp.results = results
    api._mock_benchmarks.batch_schedule_benchmark_task_runs.return_value = resp


def _setup_available_models(api, slugs):
    models = []
    for s in slugs:
        m = MagicMock()
        m.version.slug = s
        m.display_name = s.title()
        models.append(m)
    resp = MagicMock()
    resp.benchmark_models = models
    resp.next_page_token = ""
    api._mock_client.benchmarks.benchmarks_api_client.list_benchmark_models.return_value = resp


def _setup_paginated_response(mock, attr_name, items, paginated_responses=None):
    """Wire up a paginated API mock.

    If *paginated_responses* is provided, it should be a list of
    (items_list, next_page_token) tuples for multi-page scenarios.
    Otherwise a single-page response is created from *items*.
    """
    if paginated_responses:
        side_effects = []
        for page_items, token in paginated_responses:
            resp = MagicMock()
            setattr(resp, attr_name, page_items)
            resp.next_page_token = token
            side_effects.append(resp)
        mock.side_effect = side_effects
    else:
        resp = MagicMock()
        setattr(resp, attr_name, items)
        resp.next_page_token = ""
        mock.return_value = resp


def _setup_list_response(api, tasks, **kwargs):
    _setup_paginated_response(api._mock_benchmarks.list_benchmark_tasks, "tasks", tasks, **kwargs)


def _setup_runs_response(api, runs, **kwargs):
    _setup_paginated_response(api._mock_benchmarks.list_benchmark_task_runs, "runs", runs, **kwargs)


# ============================================================
# Push
# ============================================================


class TestPush:
    """``kaggle benchmarks tasks push <task> -f <file>``"""

    # -- Input validation (before any server call) --

    @pytest.mark.parametrize(
        "task, filename, content, expected_error",
        [
            ("my-task", None, None, "does not exist"),
            ("my-task", "task.txt", "hello", "must be a Python"),
            ("any-task", "task.py", "def f(): pass\n", "No @task decorators"),
            ("wrong", "task.py", '@task(name="real")\ndef f(llm): pass\n', "not found"),
            ("any-task", "task.py", "def broken(\n", "No @task decorators"),
        ],
        ids=[
            "missing_file",
            "wrong_extension",
            "no_decorators",
            "wrong_name",
            "syntax_error",
        ],
    )
    def test_push_rejects_invalid_input(self, api, tmp_path, task, filename, content, expected_error):
        if filename is None:
            filepath = "/nonexistent/task.py"
        else:
            filepath = _write_task_file(tmp_path, content, name=filename)
        with pytest.raises(ValueError, match=expected_error):
            api.benchmarks_tasks_push_cli(task, filepath)

    # -- Happy path --

    @pytest.mark.parametrize(
        "content, task_name, expected_slug",
        [
            ('@task(name="my-task")\ndef evaluate(): pass\n', "my-task", "my-task"),
            ("@task\ndef my_task(llm): pass\n", "My Task", "my-task"),
            ("@task\nasync def my_task(llm): pass\n", "My Task", "my-task"),
        ],
        ids=["explicit_name", "title_cased", "async_function"],
    )
    def test_push_creates_task(self, api, tmp_path, capsys, content, task_name, expected_slug):
        """Push converts .py -> ipynb via jupytext and creates the task."""
        filepath = _write_task_file(tmp_path, content)
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=404))
        _setup_create_response(api, task_name)

        jt = _push(api, task_name, filepath)

        # Verify jupytext conversion happened
        jt.reads.assert_called_once()
        jt.writes.assert_called_once()
        request = api._mock_benchmarks.create_benchmark_task.call_args[0][0]
        assert request.text == '{"cells": []}'
        assert request.slug == expected_slug

        captured = capsys.readouterr()
        output = captured.out
        assert f"Pushed {expected_slug}" in output
        assert "Task Details:" in output
        assert "Model Output:" in output
        assert f"kaggle b t run {expected_slug}" in output
        # When the original name differs from the slug, a normalization warning is printed to stderr.
        if task_name != expected_slug:
            assert f"normalized to slug '{expected_slug}'" in captured.err

    @pytest.mark.parametrize("status_code", [403, 404], ids=["forbidden", "not_found"])
    def test_push_creates_new_task_without_prompting(self, api, tmp_path, capsys, status_code):
        """A 403/404 means a new task -- push proceeds without confirmation."""
        filepath = _write_task_file(tmp_path)
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=status_code))
        _setup_create_response(api)
        _push(api, "my-task", filepath)
        assert "Pushed my-task" in capsys.readouterr().out

    def test_push_prefixes_relative_url(self, api, tmp_path, capsys):
        """If url starts with '/', prefix https://www.kaggle.com."""
        filepath = _write_task_file(tmp_path)
        resp = MagicMock()
        resp.url = "/benchmarks/my-task"
        resp.error = None
        api._mock_benchmarks.create_benchmark_task.return_value = resp
        _setup_completed_task(api)
        _push(api, "my-task", filepath)
        assert "https://www.kaggle.com/benchmarks/my-task" in capsys.readouterr().out

    # -- Server edge cases --

    @pytest.mark.parametrize("state", [QUEUED, RUNNING], ids=["queued", "running"])
    def test_push_rejects_pending_task_without_wait(self, api, tmp_path, state):
        """Push without --wait rejects when task is pending, with a --wait hint."""
        filepath = _write_task_file(tmp_path)
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task(state=state)
        with pytest.raises(ValueError, match="creation is still pending") as exc_info:
            _push(api, "my-task", filepath)
        assert "--wait" in str(exc_info.value)

    @pytest.mark.parametrize("state", [QUEUED, RUNNING], ids=["queued", "running"])
    def test_push_wait_monitors_pending_then_pushes(self, api, capsys, tmp_path, state):
        """Push --wait with a pending task waits for existing creation, then pushes new version."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")

        # Call 1: initial check → pending; Call 2: poll existing → completed;
        # Call 3: poll new version after push → completed
        api._mock_benchmarks.get_benchmark_task.side_effect = [
            _make_task(state=state),
            _make_task(state=COMPLETED),
            _make_task(state=COMPLETED),
        ]

        jt, ctx = _mock_jupytext()
        with ctx, patch("time.sleep"):
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=0)

        output = capsys.readouterr().out
        assert "already being created" in output
        assert "Pushed new version of my-task" in output
        # Verify the create API was still called (new version pushed)
        api._mock_benchmarks.create_benchmark_task.assert_called_once()

    def test_push_propagates_server_error(self, api, tmp_path):
        """Non-403/404 HTTP errors (e.g. 500) are re-raised, not swallowed."""
        filepath = _write_task_file(tmp_path)
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=500))
        with pytest.raises(HTTPError):
            _push(api, "my-task", filepath)

    def test_push_handles_api_error(self, api, tmp_path):
        """Push raises ValueError when response contains error_message."""
        filepath = _write_task_file(tmp_path)
        _setup_completed_task(api)

        resp = MagicMock()
        resp.error = "Some backend error"
        api._mock_benchmarks.create_benchmark_task.return_value = resp

        with pytest.raises(ValueError, match=r"Failed to push task\. Error: Some backend error"):
            _push(api, "my-task", filepath)

    def test_push_wait_polls_until_completion(self, api, capsys, tmp_path):
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")

        api._mock_benchmarks.get_benchmark_task.side_effect = [
            _make_task(state=COMPLETED),
            _make_task(state=QUEUED),
            _make_task(state=COMPLETED),
        ]

        with patch("time.sleep"):
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=0)

        output = capsys.readouterr().out
        assert "Status" in output
        assert "Completed" in output
        assert "Model Output:" in output

    def test_push_adaptive_polling(self, api, tmp_path):
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")
        api._mock_benchmarks.get_benchmark_task.side_effect = [
            _make_task(state=COMPLETED),
            _make_task(state=QUEUED),
            _make_task(state=QUEUED),
            _make_task(state=QUEUED),
            _make_task(state=COMPLETED),
        ]
        with patch("time.sleep") as mock_sleep:
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=0, poll_interval=10)
        assert mock_sleep.call_count == 3
        # Starts at 5s (ADAPTIVE_POLL_START), grows by 1.5x, caps at poll_interval (10)
        mock_sleep.assert_any_call(5)
        mock_sleep.assert_any_call(7)
        mock_sleep.assert_any_call(10)

    def test_push_large_poll_interval_adaptive_growth(self, api, tmp_path):
        """When poll_interval > 60s, polling still starts at 5s and grows adaptively to poll_interval."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")
        api._mock_benchmarks.get_benchmark_task.side_effect = [
            _make_task(state=COMPLETED),
            _make_task(state=QUEUED),
            _make_task(state=QUEUED),
            _make_task(state=QUEUED),
            _make_task(state=COMPLETED),
        ]
        with patch("time.sleep") as mock_sleep:
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=0, poll_interval=90)
        intervals = [call[0][0] for call in mock_sleep.call_args_list]
        assert mock_sleep.call_count == 3
        # Starts at 5s, grows by 1.5x: 5 -> 7 -> 10, all below 90 cap
        assert intervals == [5, 7, 10]

    def test_push_verbose_prints_sleep_info(self, api, capsys, tmp_path):
        """Verbose flag causes adaptive sleep durations to be printed."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")
        api._mock_benchmarks.get_benchmark_task.side_effect = [
            _make_task(state=COMPLETED),
            _make_task(state=QUEUED),
            _make_task(state=COMPLETED),
        ]
        with patch("time.sleep"):
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=0, poll_interval=10, verbose=True)
        output = capsys.readouterr().out
        assert "Adaptive polling sleep: 5s" in output

    def test_push_adaptive_polling_caps_at_poll_interval(self, api, tmp_path):
        """Adaptive polling does not exceed the user's poll_interval."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")
        # With poll_interval=10: 5 -> 7 -> 10 -> 10 -> 10 -> 10 -> 10
        api._mock_benchmarks.get_benchmark_task.side_effect = [
            _make_task(state=COMPLETED),  # initial check
            *[_make_task(state=QUEUED) for _ in range(7)],
            _make_task(state=COMPLETED),
        ]
        with patch("time.sleep") as mock_sleep:
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=0, poll_interval=10)
        intervals = [call[0][0] for call in mock_sleep.call_args_list]
        # All intervals should be <= poll_interval (10)
        assert all(i <= 10 for i in intervals), f"Intervals exceeded poll_interval cap: {intervals}"
        # The last few should be exactly 10 (capped)
        assert intervals[-1] == 10

    def test_push_wait_times_out(self, api, capsys, tmp_path):
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")

        api._mock_benchmarks.get_benchmark_task.side_effect = [
            _make_task(state=COMPLETED),
            _make_task(state=QUEUED),
            _make_task(state=QUEUED),
        ]

        with patch("time.sleep"), patch("time.time", side_effect=[1000, 1060]):
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=30)

        output = capsys.readouterr().out
        assert "Timed out after 30s waiting for task creation." in output

    @pytest.mark.parametrize("interval", [0, -1], ids=["zero", "negative"])
    def test_push_rejects_non_positive_poll_interval(self, api, tmp_path, interval):
        """Push raises ValueError when poll_interval is 0 or negative."""
        filepath = _write_task_file(tmp_path)
        with pytest.raises(ValueError, match="--poll-interval must be a positive integer"):
            api.benchmarks_tasks_push_cli("my-task", filepath, wait=0, poll_interval=interval)

    # -- Kaggle dataset attachment --

    def test_push_with_kaggle_datasets(self, api, tmp_path, capsys):
        """Push attaches Kaggle datasets via BenchmarkTaskOptions."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")
        resp = api._mock_benchmarks.create_benchmark_task.return_value
        resp.options = MagicMock()
        resp.options.dataset_data_sources = ["user/dataset-one", "user/dataset-two"]
        resp.invalid_dataset_sources = []

        jt, ctx = _mock_jupytext()
        with ctx:
            api.benchmarks_tasks_push_cli("my-task", filepath, kaggle_datasets=["user/dataset-one", "user/dataset-two"])

        request = api._mock_benchmarks.create_benchmark_task.call_args[0][0]
        assert request.options is not None
        assert request.options.dataset_data_sources == ["user/dataset-one", "user/dataset-two"]
        output = capsys.readouterr().out
        assert "Attached Kaggle dataset(s)" in output
        # Attach message must appear below both Task Details and Model Output (compare URL)
        task_idx = output.index("Task Details:")
        model_idx = output.index("Model Output:")
        attach_idx = output.index("Attached Kaggle dataset(s)")
        assert task_idx < model_idx < attach_idx

    def test_push_with_invalid_kaggle_dataset_warns(self, api, tmp_path, capsys):
        """Push warns about invalid/unresolvable Kaggle datasets."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")
        resp = api._mock_benchmarks.create_benchmark_task.return_value
        resp.options = MagicMock()
        resp.options.dataset_data_sources = ["user/valid"]
        resp.invalid_dataset_sources = ["user/nonexistent"]

        jt, ctx = _mock_jupytext()
        with ctx:
            api.benchmarks_tasks_push_cli("my-task", filepath, kaggle_datasets=["user/valid", "user/nonexistent"])

        captured = capsys.readouterr()
        assert "could not be resolved" in captured.err
        assert "user/nonexistent" in captured.err

    def test_push_without_kaggle_datasets_sends_no_options(self, api, tmp_path, capsys):
        """Push without --kaggle-dataset does not set options on the request."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")
        _push(api, "my-task", filepath)
        request = api._mock_benchmarks.create_benchmark_task.call_args[0][0]
        # When no datasets are specified, options should not be set
        assert not hasattr(request, "options") or request.options is None

    def test_push_warns_when_removing_previously_attached_datasets(self, api, tmp_path, capsys):
        """Re-pushing without --kaggle-dataset warns when previous version had datasets."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")

        # Previous version had datasets attached
        prev_task = _make_task(slug="my-task", state=COMPLETED)
        prev_task.options = MagicMock()
        prev_task.options.dataset_data_sources = ["user/important-data"]
        api._mock_benchmarks.get_benchmark_task.return_value = prev_task

        jt, ctx = _mock_jupytext()
        with ctx:
            api.benchmarks_tasks_push_cli("my-task", filepath)  # no kaggle_datasets

        captured = capsys.readouterr()
        assert "previous version" in captured.err
        assert "user/important-data" in captured.err
        assert "will detach" in captured.err

    def test_push_no_warning_when_re_push_keeps_datasets(self, api, tmp_path, capsys):
        """Re-pushing WITH --kaggle-dataset does NOT warn about detaching."""
        filepath = _write_task_file(tmp_path)
        _setup_create_response(api, "my-task")

        prev_task = _make_task(slug="my-task", state=COMPLETED)
        prev_task.options = MagicMock()
        prev_task.options.dataset_data_sources = ["user/important-data"]
        api._mock_benchmarks.get_benchmark_task.return_value = prev_task

        resp = api._mock_benchmarks.create_benchmark_task.return_value
        resp.options = MagicMock()
        resp.options.dataset_data_sources = ["user/important-data"]
        resp.invalid_dataset_sources = []

        jt, ctx = _mock_jupytext()
        with ctx:
            api.benchmarks_tasks_push_cli("my-task", filepath, kaggle_datasets=["user/important-data"])

        captured = capsys.readouterr()
        assert "will detach" not in captured.err


# ============================================================
# Run
# ============================================================


class TestRun:
    """``kaggle benchmarks tasks run <task> [-m ...] [--wait]``"""

    # -- Pre-conditions --

    def test_run_rejects_non_completed_task(self, api):
        """Run errors when the task creation state is not COMPLETED."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task(state=QUEUED)
        with pytest.raises(ValueError, match="not ready to run"):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        api._mock_benchmarks.batch_schedule_benchmark_task_runs.assert_not_called()

    @pytest.mark.parametrize("interval", [0, -1], ids=["zero", "negative"])
    def test_run_rejects_non_positive_poll_interval(self, api, interval):
        """Run raises ValueError when poll_interval is 0 or negative."""
        with pytest.raises(ValueError, match="--poll-interval must be a positive integer"):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], poll_interval=interval)

    def test_run_errored_task_includes_task_info(self, api):
        """ERRORED task error message includes task info."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task(state=ERRORED)
        with pytest.raises(ValueError, match="Task Info:"):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])

    @pytest.mark.parametrize("status_code", [403, 404], ids=["forbidden", "not_found"])
    def test_run_task_not_found(self, api, status_code):
        """Run gives friendly error when task doesn't exist (403/404)."""
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=status_code))
        with pytest.raises(ValueError, match="not found"):
            api.benchmarks_tasks_run_cli("no-such-task", ["gemini-pro"])

    # -- Model scheduling --

    @pytest.mark.parametrize(
        "models",
        [["gemini-pro"], ["gemini-pro", "gemma-2b"]],
        ids=["single_model", "multiple_models"],
    )
    def test_run_schedules_models(self, api, capsys, models):
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result() for _ in models])
        api.benchmarks_tasks_run_cli("my-task", models)
        output = capsys.readouterr().out
        assert "Submitted run(s) for task 'my-task'" in output
        assert "Next steps:" in output
        assert "Check run status:" in output
        assert "kaggle b t status my-task" in output
        for m in models:
            assert f"{m}: Scheduled" in output

    def test_run_reports_skipped_with_reason(self, api, capsys):
        """Skipped runs print the backend-provided reason."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result(scheduled=False, skipped_reason="Already running")])
        api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        output = capsys.readouterr().out
        assert "gemini-pro: Skipped" in output
        assert "Already running" in output

    @pytest.mark.parametrize("reason", [None, ""], ids=["none", "empty"])
    def test_run_skipped_empty_reason_does_not_crash(self, api, capsys, reason):
        """Empty/None skip reason renders without crashing."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result(scheduled=False, skipped_reason=reason)])
        api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        output = capsys.readouterr().out
        assert "gemini-pro: Skipped" in output

    def test_run_no_status_hint_when_waiting(self, api, capsys):
        """When --wait is used, the status hint should not appear."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.return_value = MagicMock(
            runs=[_make_run(state=RUN_COMPLETED)], next_page_token=""
        )
        with patch("time.sleep"):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0)
        output = capsys.readouterr().out
        assert "Check run status" not in output

    # -- Interactive model selection --

    def test_run_prompts_model_selection(self, api):
        """No model specified -> user picks from a numbered list."""
        _setup_completed_task(api)
        _setup_available_models(api, ["gemini-pro", "gemma-2b"])
        _setup_batch_schedule(api, [_make_run_result()])
        with patch("builtins.input", return_value="1"):
            api.benchmarks_tasks_run_cli("my-task")
        request = api._mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemini-pro"]

    def test_run_selects_multiple_models_by_number(self, api):
        """Comma-separated indices pick multiple models."""
        _setup_completed_task(api)
        _setup_available_models(api, ["gemini-pro", "gemma-2b"])
        _setup_batch_schedule(api, [])
        with patch("builtins.input", return_value="1,2"):
            api.benchmarks_tasks_run_cli("my-task")
        request = api._mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemini-pro", "gemma-2b"]

    def test_run_rejects_empty_model_list(self, api):
        """No models available on server -> ValueError."""
        _setup_completed_task(api)
        _setup_available_models(api, [])
        with pytest.raises(ValueError, match="No benchmark models available"):
            api.benchmarks_tasks_run_cli("my-task")

    def test_run_rejects_invalid_model_selection(self, api):
        """Bad input during interactive model selection -> ValueError."""
        _setup_completed_task(api)
        _setup_available_models(api, ["gemini-pro"])
        with patch("builtins.input", return_value="abc"):
            with pytest.raises(ValueError, match="is not a valid choice"):
                api.benchmarks_tasks_run_cli("my-task")

    def test_run_eof_without_models_raises(self, api):
        """Closed stdin (EOF) -> clear error instead of hanging on input()."""
        _setup_completed_task(api)
        _setup_available_models(api, ["gemini-pro"])
        with patch("builtins.input", side_effect=EOFError), pytest.raises(ValueError, match="-m/--model"):
            api.benchmarks_tasks_run_cli("my-task")

    def test_run_model_selection_table_header(self, api, capsys):
        """Interactive model picker prints summary + Model/Slug/Modality header + underline."""
        _setup_completed_task(api)
        _setup_available_models(api, ["gemini-pro", "gemma-2b"])
        _setup_batch_schedule(api, [_make_run_result()])
        with patch("builtins.input", return_value="1"):
            api.benchmarks_tasks_run_cli("my-task")
        output = capsys.readouterr().out
        assert "Showing 1-2 of 2 models available" in output
        assert "Model" in output and "Slug" in output and "Modality" in output
        # Per-column unicode underlines
        assert "─" * len("Model") in output

    def test_run_model_selection_pagination(self, api, capsys):
        """When >page_size models, the [Page X/Y] indicator appears and 'n' advances."""
        _setup_completed_task(api)
        _setup_available_models(api, [f"model-{i}" for i in range(25)])
        _setup_batch_schedule(api, [_make_run_result()])
        # 'n' moves to page 2, then '21' selects model index 21.
        with patch("builtins.input", side_effect=["n", "21"]):
            api.benchmarks_tasks_run_cli("my-task")
        output = capsys.readouterr().out
        assert "Showing 1-20 of 25 models available" in output
        assert "Showing 21-25 of 25 models available" in output
        assert "[Page 1/2]" in output
        assert "[Page 2/2]" in output
        # Verify the selection landed on the right (1-indexed) slug.
        request = api._mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["model-20"]

    def test_run_accepts_piped_model_selection(self, api):
        """Piped stdin with a selection still schedules the chosen model."""
        _setup_completed_task(api)
        _setup_available_models(api, ["gemini-pro", "gemma-2b"])
        _setup_batch_schedule(api, [_make_run_result()])
        with patch("builtins.input", return_value="2"):
            api.benchmarks_tasks_run_cli("my-task")
        request = api._mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemma-2b"]

    # -- Wait / polling --

    def test_run_wait_polls_until_completion(self, api, capsys):
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.side_effect = [
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_COMPLETED)], next_page_token=""),
        ]
        with patch("time.sleep"):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0)
        output = capsys.readouterr().out
        assert "Waiting for run(s) to complete" in output
        assert "All runs completed" in output
        assert "gemini-pro: COMPLETED" in output

    def test_run_adaptive_polling(self, api):
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.side_effect = [
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_COMPLETED)], next_page_token=""),
        ]
        with patch("time.sleep") as mock_sleep:
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0, poll_interval=10)
        assert mock_sleep.call_count == 3
        # Starts at 5s (ADAPTIVE_POLL_START), grows by 1.5x, caps at poll_interval (10)
        mock_sleep.assert_any_call(5)
        mock_sleep.assert_any_call(7)
        mock_sleep.assert_any_call(10)

    def test_run_large_poll_interval_adaptive_growth(self, api):
        """When poll_interval > 60s, polling still starts at 5s and grows adaptively to poll_interval."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.side_effect = [
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_COMPLETED)], next_page_token=""),
        ]
        with patch("time.sleep") as mock_sleep:
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0, poll_interval=90)
        intervals = [call[0][0] for call in mock_sleep.call_args_list]
        assert mock_sleep.call_count == 3
        # Starts at 5s, grows by 1.5x: 5 -> 7 -> 10, all below 90 cap
        assert intervals == [5, 7, 10]

    def test_run_verbose_prints_sleep_info(self, api, capsys):
        """Verbose flag causes adaptive sleep durations to be printed."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.side_effect = [
            MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token=""),
            MagicMock(runs=[_make_run(state=RUN_COMPLETED)], next_page_token=""),
        ]
        with patch("time.sleep"):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0, poll_interval=10, verbose=True)
        output = capsys.readouterr().out
        assert "Adaptive polling sleep: 5s" in output

    def test_run_adaptive_polling_caps_at_poll_interval(self, api):
        """Adaptive polling does not exceed the user's poll_interval."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        # With poll_interval=10: 5 -> 7 -> 10 -> 10 -> 10 -> 10 -> 10
        api._mock_benchmarks.list_benchmark_task_runs.side_effect = [
            *[MagicMock(runs=[_make_run(state=RUN_RUNNING)], next_page_token="") for _ in range(7)],
            MagicMock(runs=[_make_run(state=RUN_COMPLETED)], next_page_token=""),
        ]
        with patch("time.sleep") as mock_sleep:
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0, poll_interval=10)
        intervals = [call[0][0] for call in mock_sleep.call_args_list]
        # All intervals should be <= poll_interval (10)
        assert all(i <= 10 for i in intervals), f"Intervals exceeded poll_interval cap: {intervals}"
        # The last few should be exactly 10 (capped)
        assert intervals[-1] == 10

    def test_run_wait_times_out(self, api, capsys):
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.return_value = MagicMock(
            runs=[_make_run(state=RUN_RUNNING)], next_page_token=""
        )
        with patch("time.sleep"), patch("time.time", side_effect=[1000, 1060]):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=30)
        output = capsys.readouterr().out
        assert "Timed out after 30s waiting for runs." in output

    def test_run_wait_shows_errored_runs(self, api, capsys):
        """ERRORED runs display with ERRORED label and raise ValueError."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.return_value = MagicMock(
            runs=[_make_run(state=RUN_ERRORED)], next_page_token=""
        )
        with patch("time.sleep"), pytest.raises(ValueError, match="run\(s\) failed"):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0)
        assert "gemini-pro: ERRORED" in capsys.readouterr().out

    def test_run_invalid_model_gives_friendly_error(self, api):
        """Invalid model name returns a friendly error instead of raw 404."""
        _setup_completed_task(api)
        api._mock_benchmarks.batch_schedule_benchmark_task_runs.side_effect = HTTPError(
            response=MagicMock(status_code=404)
        )
        with pytest.raises(ValueError, match="model names may be invalid"):
            api.benchmarks_tasks_run_cli("my-task", ["nonexistent-model"])


# ============================================================
# List
# ============================================================


class TestList:
    """``kaggle benchmarks tasks list [--name-regex <pattern>] [--status <status>]``"""

    def test_list_all(self, api, capsys):
        _setup_list_response(api, [_make_task()])
        api.benchmarks_tasks_list_cli()
        output = capsys.readouterr().out
        assert "Task" in output
        assert "Status" in output
        assert "my-task" in output
        assert "Showing 1-1 of 1 tasks" in output

    def test_list_retries_on_429_and_succeeds(self, api, capsys):
        """When list receives 429, it retries and succeeds, printing retry log to stderr."""
        api._mock_benchmarks.list_benchmark_tasks.side_effect = [
            HTTPError(response=MagicMock(status_code=429, headers={})),
            MagicMock(tasks=[_make_task()], next_page_token=""),
        ]
        with patch("time.sleep") as mock_sleep:
            api.benchmarks_tasks_list_cli()
        mock_sleep.assert_called_once()
        captured = capsys.readouterr()
        assert "Request failed:" in captured.err
        assert "Will retry in" in captured.err
        assert "Request failed:" not in captured.out
        assert "Task" in captured.out
        assert "my-task" in captured.out

    def test_list_with_name_regex_filter(self, api, capsys):
        _setup_list_response(api, [_make_task(slug="math-task")])
        api.benchmarks_tasks_list_cli(name_regex="math.*")
        request = api._mock_benchmarks.list_benchmark_tasks.call_args[0][0]
        assert request.regex_filter == "math.*"
        assert "math-task" in capsys.readouterr().out

    def test_list_with_status_filter(self, api, capsys):
        _setup_list_response(api, [_make_task()])
        api.benchmarks_tasks_list_cli(status="completed")
        request = api._mock_benchmarks.list_benchmark_tasks.call_args[0][0]
        assert request.status_filter == "completed"

    def test_list_pagination(self, api, capsys):
        """List fetches all pages of tasks."""
        _setup_list_response(
            api,
            tasks=[],
            paginated_responses=[
                ([_make_task(slug="task-1")], "page2"),
                ([_make_task(slug="task-2")], ""),
            ],
        )
        api.benchmarks_tasks_list_cli()
        output = capsys.readouterr().out
        assert "task-1" in output
        assert "task-2" in output

    @pytest.mark.parametrize("tasks", [[], None], ids=["empty_list", "none"])
    def test_list_empty(self, api, capsys, tasks):
        """Unfiltered empty list prints the actionable push hint, not an empty table."""
        _setup_list_response(api, tasks)
        api.benchmarks_tasks_list_cli()
        output = capsys.readouterr().out
        assert "No tasks found. Use 'kaggle b t push' to create one." in output
        assert "my-task" not in output

    @pytest.mark.parametrize(
        "kwargs",
        [{"name_regex": "no-match.*"}, {"status": "completed"}, {"name_regex": "x", "status": "completed"}],
        ids=["regex", "status", "both"],
    )
    def test_list_empty_with_filter(self, api, capsys, kwargs):
        """Filtered empty list says 'matching the given filters' rather than the push hint."""
        _setup_list_response(api, [])
        api.benchmarks_tasks_list_cli(**kwargs)
        output = capsys.readouterr().out
        assert "No tasks found matching the given filters." in output
        assert "kaggle b t push" not in output

    def test_list_table_format(self, api, capsys):
        """Table uses per-column unicode-line underlines spanning each column's full width."""
        _setup_list_response(api, [_make_task()])
        api.benchmarks_tasks_list_cli()
        output = capsys.readouterr().out
        # Column widths: max_task_len(>=40)/20/20.
        assert "─" * 40 in output  # Task column (min width 40)
        assert "─" * 20 in output  # Status / Created columns

    def test_list_pagination_prompts_for_navigation(self, api, capsys):
        """When >page_size tasks, an interactive [n/p/q] prompt drives paging."""
        tasks = [_make_task(slug=f"task-{i}") for i in range(25)]
        _setup_list_response(api, tasks)
        # 'n' advances to page 2, then 'q' exits.
        with patch("builtins.input", side_effect=["n", "q"]):
            api.benchmarks_tasks_list_cli()
        output = capsys.readouterr().out
        assert "Showing 1-20 of 25 tasks" in output
        assert "Showing 21-25 of 25 tasks" in output
        assert "[Page 1/2]" in output
        assert "[Page 2/2]" in output

    def test_list_page_size_overrides_default(self, api, capsys):
        """``--page-size 5`` overrides the default page size."""
        tasks = [_make_task(slug=f"task-{i}") for i in range(12)]
        _setup_list_response(api, tasks)
        with patch("builtins.input", side_effect=["q"]):
            api.benchmarks_tasks_list_cli(page_size=5)
        output = capsys.readouterr().out
        assert "Showing 1-5 of 12 tasks" in output
        assert "[Page 1/3]" in output

    def test_list_all_skips_interactive_pager(self, api, capsys):
        """``--all`` prints every task and never prompts for input."""
        tasks = [_make_task(slug=f"task-{i}") for i in range(25)]
        _setup_list_response(api, tasks)
        # No input mock — would raise if input() were called.
        api.benchmarks_tasks_list_cli(show_all=True)
        output = capsys.readouterr().out
        assert "Showing 1-25 of 25 tasks" in output
        assert "[Page" not in output
        assert "task-0" in output
        assert "task-24" in output


# ============================================================
# Status
# ============================================================


class TestStatus:
    """``kaggle benchmarks tasks status <task> [-m <model> ...]``"""

    def test_status_header(self, api, capsys):
        """Status prints Task/Status/Created/Public header."""
        task = _make_task()
        task.is_public = True
        api._mock_benchmarks.get_benchmark_task.return_value = task
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "Task:" in output
        assert "Version:" in output
        assert "Status:   Completed" in output
        assert "Created:" in output
        assert "Public:   True" in output
        assert "Task URL:" in output

    @pytest.mark.parametrize("status_code", [403, 404], ids=["forbidden", "not_found"])
    def test_status_task_not_found(self, api, status_code):
        """Status gives friendly error when task doesn't exist (403/404)."""
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=status_code))
        with pytest.raises(ValueError, match="not found"):
            api.benchmarks_tasks_status_cli("no-such-task")

    def test_status_no_runs_message(self, api, capsys):
        """No runs -> helpful message with run command hint."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "No runs yet" in output
        assert "kaggle b t run my-task" in output

    @pytest.mark.parametrize(
        "model_input, expected",
        [("gemini-3", ["gemini-3"]), (["gemini-3", "gpt-5"], ["gemini-3", "gpt-5"])],
        ids=["single", "multiple"],
    )
    def test_status_with_model_filter(self, api, capsys, model_input, expected):
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task", model=model_input)
        request = api._mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == expected

    def test_status_run_table(self, api, capsys):
        """Completed run renders with correct columns."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [_make_run(model="gemini-pro", run_id=42)])
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "gemini-pro" in output
        assert "https://www.kaggle.com/benchmarks/runs/42" not in output

    def test_status_errored_run_shows_error_message(self, api, capsys):
        """ERRORED runs show error in a dedicated section below the table."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(
            api,
            [_make_run(model="gemma-2b", state=RUN_ERRORED, run_id=43, error_message="OOM")],
        )
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "Errors:" in output
        assert "[gemma-2b]" in output
        assert "OOM" in output

    def test_status_errored_run_truncates_traceback(self, api, capsys):
        """Multi-line tracebacks collapse to the last meaningful line only."""
        traceback_msg = (
            "Traceback (most recent call last):\n"
            '  File "/benchmarks/src/tasks.py", line 127, in run\n'
            "    run.result = self.func(*args, **kwargs)\n"
            '  File "/tmp/ipykernel/162297423.py", line 6, in what_is_kaggle\n'
            '    response = llm.prompt("What is Kaggle?")\n'
            '  File "/openai/_base_client.py", line 1047, in request\n'
            "    raise self._make_status_error_from_response(err.response) from None\n"
            "openai.BadRequestError: Error code: 400 - max_tokens too large\n"
        )
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(
            api,
            [_make_run(model="gemma-2b", state=RUN_ERRORED, run_id=43, error_message=traceback_msg)],
        )
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        # Final exception line is rendered...
        assert "openai.BadRequestError: Error code: 400 - max_tokens too large" in output
        # ...stack-frame noise is suppressed
        assert "Traceback (most recent call last)" not in output
        assert "ipykernel" not in output
        assert "/openai/_base_client.py" not in output
        # Slug and exception sit on the same line — no newline between bracket and message
        assert "[gemma-2b]" in output
        assert "[gemma-2b]\n" not in output

    def test_status_shows_log_hint(self, api, capsys):
        """Status with runs shows a hint to use 'kaggle b t log' for details."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [_make_run(model="gemini-pro", run_id=42)])
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "View logs: kaggle b t log my-task [-m <model>]" in output

    def test_status_no_log_hint_without_runs(self, api, capsys):
        """Status without runs does NOT show the log hint."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "View logs:" not in output

    def test_status_pagination(self, api, capsys):
        """Status fetches all pages of runs."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(
            api,
            runs=[],
            paginated_responses=[
                ([_make_run(model="gemini-1", run_id=1)], "page2"),
                ([_make_run(model="gemini-2", run_id=2)], ""),
            ],
        )
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "gemini-1" in output
        assert "gemini-2" in output


# ============================================================
# Download
# ============================================================


class TestDownload:
    """``kaggle benchmarks tasks download <task> [-m <model> ...] [-o <dir>]``"""

    def _mock_download(self, api):
        """Mock download_file, zipfile.ZipFile, and os.remove for download tests."""
        _setup_completed_task(api)
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()

    @pytest.mark.parametrize("status_code", [403, 404], ids=["forbidden", "not_found"])
    def test_download_task_not_found(self, api, status_code):
        """Download gives friendly error when task doesn't exist (403/404)."""
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=status_code))
        with pytest.raises(ValueError, match="not found"):
            api.benchmarks_tasks_download_cli("no-such-task")

    def test_download_to_specific_output(self, api, capsys):
        _setup_runs_response(api, [_make_run()])
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", output="my_output_dir")
        output = capsys.readouterr().out
        assert "Downloading output runs for my-task" in output
        assert "gemini-pro" in output
        assert "Done" in output
        assert "my_output_dir" in output

    def test_download_default_output_path(self, api, capsys):
        """Default output is ./{task}/{model}/{run_id}.zip."""
        _setup_runs_response(api, [_make_run(run_id=1)])
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task")
        # download_file receives the .zip path
        call_args = api.download_file.call_args
        zippath = call_args[0][1]
        expected = os.path.join(".", "my-task", "1", "gemini-pro", "1.zip")
        assert zippath == expected

    def test_download_with_model_filter(self, api, capsys):
        _setup_runs_response(api, [_make_run()])
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", model="gemini-pro")
        request = api._mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemini-pro"]

    def test_download_skips_non_downloadable_runs(self, api, capsys):
        """QUEUED/RUNNING runs are silently skipped."""
        _setup_runs_response(
            api,
            [
                _make_run(model="queued-model", state=RUN_QUEUED, run_id=1),
                _make_run(model="running-model", state=RUN_RUNNING, run_id=2),
                _make_run(model="done-model", state=RUN_COMPLETED, run_id=3),
            ],
        )
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task")
        # Only the completed run should be downloaded
        assert api._mock_benchmarks.download_benchmark_task_run_output.call_count == 1
        output = capsys.readouterr().out
        assert "done-model" in output
        assert "queued-model" not in output
        assert "running-model" not in output

    def test_download_no_runs_shows_message(self, api, capsys):
        """No runs at all prints a helpful message with run command hint."""
        _setup_completed_task(api)
        _setup_runs_response(api, [])
        api.benchmarks_tasks_download_cli("my-task", model="nonexistent-model")
        output = capsys.readouterr().out
        assert "No runs found for task 'my-task'" in output
        assert "kaggle b t run my-task" in output

    def test_download_all_pending_shows_message(self, api, capsys):
        """All runs still in progress prints a status hint."""
        _setup_completed_task(api)
        _setup_runs_response(
            api,
            [
                _make_run(state=RUN_QUEUED, run_id=1),
                _make_run(state=RUN_RUNNING, run_id=2),
            ],
        )
        api.benchmarks_tasks_download_cli("my-task")
        output = capsys.readouterr().out
        assert "No downloadable runs yet" in output
        assert "2 run(s) still in progress" in output
        assert "kaggle b t status my-task" in output

    def test_download_skips_existing_output(self, api, capsys, tmp_path):
        """Already-downloaded runs render as Cached without making API calls."""
        _setup_runs_response(api, [_make_run(run_id=42)])
        self._mock_download(api)
        outdir = str(tmp_path / "out")
        # Pre-create the output directory with a file to simulate a previous download
        existing = os.path.join(outdir, "my-task", "1", "gemini-pro", "42")
        os.makedirs(existing)
        with open(os.path.join(existing, "result.run.json"), "w") as f:
            f.write("{}")

        api.benchmarks_tasks_download_cli("my-task", output=outdir)

        output = capsys.readouterr().out
        assert "gemini-pro" in output
        assert "Cached" in output
        assert "1 run(s) cached" in output
        # The File column shows the output directory, not a .zip path
        assert "gemini-pro/42/" in output
        assert "/42/42.zip" not in output
        # No download API call should have been made
        api._mock_benchmarks.download_benchmark_task_run_output.assert_not_called()

    def test_download_re_fetches_empty_existing_dir(self, api, capsys, tmp_path):
        """An empty leftover output directory should not count as cached; re-download instead."""
        _setup_runs_response(api, [_make_run(run_id=42)])
        self._mock_download(api)
        outdir = str(tmp_path / "out")
        # Empty leftover dir from a prior interrupted run
        existing = os.path.join(outdir, "my-task", "1", "gemini-pro", "42")
        os.makedirs(existing)

        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", output=outdir)

        output = capsys.readouterr().out
        assert "Cached" not in output
        # The empty dir triggers a fresh download
        api._mock_benchmarks.download_benchmark_task_run_output.assert_called_once()

    def test_download_cached_dir_without_source_prints_tip_when_s_passed(self, api, capsys, tmp_path):
        """When -s is passed but the cached dir lacks source notebooks, hint that -f -s is needed."""
        _setup_runs_response(api, [_make_run(run_id=42)])
        self._mock_download(api)
        outdir = str(tmp_path / "out")
        # Cached dir exists but has no __notebook__.ipynb / __notebook_source__.ipynb
        existing = os.path.join(outdir, "my-task", "1", "gemini-pro", "42")
        os.makedirs(existing)
        # Drop a placeholder file so the dir isn't empty, but not a source notebook
        with open(os.path.join(existing, "result.run.json"), "w") as f:
            f.write("{}")

        api.benchmarks_tasks_download_cli("my-task", output=outdir, include_source=True)
        output = capsys.readouterr().out

        assert "Cached" in output
        assert "1 cached run(s) lack source notebooks" in output
        assert "-f -s" in output
        # Without -f, the cached dir was not touched: no download API call
        api._mock_benchmarks.download_benchmark_task_run_output.assert_not_called()

    def test_download_cached_dir_with_source_does_not_print_tip(self, api, capsys, tmp_path):
        """When the cached dir already contains source notebooks, no tip is shown."""
        _setup_runs_response(api, [_make_run(run_id=42)])
        self._mock_download(api)
        outdir = str(tmp_path / "out")
        existing = os.path.join(outdir, "my-task", "1", "gemini-pro", "42")
        os.makedirs(existing)
        # Simulate a previous -s download by dropping the source notebook
        with open(os.path.join(existing, "__notebook__.ipynb"), "w") as f:
            f.write("{}")

        api.benchmarks_tasks_download_cli("my-task", output=outdir, include_source=True)
        output = capsys.readouterr().out

        assert "Cached" in output
        assert "lack source notebooks" not in output

    def test_download_summary_counts(self, api, capsys, tmp_path):
        """Download summary shows correct downloaded and cached counts."""
        _setup_runs_response(
            api,
            [_make_run(model="new-model", run_id=1), _make_run(model="old-model", run_id=2)],
        )
        self._mock_download(api)
        outdir = str(tmp_path / "out")
        # Pre-create only run 2 with a file to simulate a previous download
        existing = os.path.join(outdir, "my-task", "1", "old-model", "2")
        os.makedirs(existing)
        with open(os.path.join(existing, "result.run.json"), "w") as f:
            f.write("{}")

        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", output=outdir)

        output = capsys.readouterr().out
        assert "1 run(s) downloaded" in output
        assert "1 run(s) cached" in output

    def test_download_force_overwrites_existing_output(self, api, capsys, tmp_path):
        """Using force=True re-downloads and overwrites existing output."""
        _setup_runs_response(api, [_make_run(run_id=42)])
        self._mock_download(api)
        outdir = str(tmp_path / "out")
        # Pre-create the output directory to simulate a previous download
        existing = os.path.join(outdir, "my-task", "1", "gemini-pro", "42")
        os.makedirs(existing)

        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", output=outdir, force=True)

        output = capsys.readouterr().out
        assert "gemini-pro" in output
        assert "Done" in output
        assert "1 run(s) downloaded" in output
        # The download API call must have been made!
        api._mock_benchmarks.download_benchmark_task_run_output.assert_called_once()

    def test_download_includes_errored_runs(self, api, capsys):
        """ERRORED runs are also downloadable per spec."""
        _setup_runs_response(api, [_make_run(state=RUN_ERRORED)])
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task")
        assert api._mock_benchmarks.download_benchmark_task_run_output.call_count == 1

    def test_download_pagination(self, api, capsys):
        """Download fetches all pages of runs."""
        _setup_runs_response(
            api,
            runs=[],
            paginated_responses=[
                ([_make_run(model="gemini-1", run_id=1)], "page2"),
                ([_make_run(model="gemini-2", run_id=2)], ""),
            ],
        )
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task")
        assert api._mock_benchmarks.download_benchmark_task_run_output.call_count == 2

    def test_download_extracts_zip_and_cleans_up(self, api, capsys, tmp_path):
        """Download extracts zip into a directory and removes the zip file."""
        _setup_completed_task(api)
        _setup_runs_response(api, [_make_run(run_id=42)])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()

        outdir = str(tmp_path / "out")
        zip_path = os.path.join(outdir, "my-task", "1", "gemini-pro", "42.zip")

        # Make download_file create a real zip so extraction works
        def fake_download(response, outfile, http_client, quiet=False):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("output.txt", "hello world")
            with open(outfile, "wb") as f:
                f.write(buf.getvalue())

        api.download_file = MagicMock(side_effect=fake_download)

        api.benchmarks_tasks_download_cli("my-task", output=outdir)

        # Verify extraction happened: {output}/{task}/{version}/{model}/{run_id}/
        extracted_dir = os.path.join(outdir, "my-task", "1", "gemini-pro", "42")
        assert os.path.isdir(extracted_dir)
        assert os.path.isfile(os.path.join(extracted_dir, "output.txt"))
        with open(os.path.join(extracted_dir, "output.txt")) as f:
            assert f.read() == "hello world"
        # Verify zip was cleaned up
        assert not os.path.exists(zip_path)

    def test_download_model_slug_at_sign_fallback(self, api, capsys):
        """Model filter matches proxy-style slugs via @→- replacement.

        The server may return ``model_version_slug`` in the proxy format
        (e.g. ``anthropic/claude-sonnet-4-6@default``) while the user filters
        by the display slug (``claude-sonnet-4-6-default``).  The client-side
        fallback in ``_fetch_task_runs`` should still include such runs.
        """
        _setup_runs_response(
            api,
            [_make_run(model="anthropic/claude-sonnet-4-6@default", run_id=10)],
        )
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", model="claude-sonnet-4-6-default")
        # The run should NOT have been filtered out
        assert api._mock_benchmarks.download_benchmark_task_run_output.call_count == 1

    def test_download_bad_zip_keeps_file_and_continues(self, api, capsys, tmp_path):
        """Corrupt zip prints a warning, keeps the raw file, and continues."""
        _setup_completed_task(api)
        _setup_runs_response(
            api,
            [
                _make_run(model="bad-model", run_id=10),
                _make_run(model="good-model", run_id=11),
            ],
        )
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()

        outdir = str(tmp_path / "out")

        call_count = 0

        def fake_download(response, outfile, http_client, quiet=False):
            nonlocal call_count
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            call_count += 1
            if call_count == 1:
                # First download: write garbage (not a valid zip)
                with open(outfile, "wb") as f:
                    f.write(b"this is not a zip")
            else:
                # Second download: write a valid zip
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as zf:
                    zf.writestr("result.txt", "ok")
                with open(outfile, "wb") as f:
                    f.write(buf.getvalue())

        api.download_file = MagicMock(side_effect=fake_download)

        api.benchmarks_tasks_download_cli("my-task", output=outdir)

        output = capsys.readouterr().out
        # Bad zip: status row marks it failed, raw file kept
        assert "Bad zip" in output
        bad_zip_path = os.path.join(outdir, "my-task", "1", "bad-model", "10.zip")
        assert os.path.isfile(bad_zip_path)
        # Good zip: extracted successfully
        good_dir = os.path.join(outdir, "my-task", "1", "good-model", "11")
        assert os.path.isdir(good_dir)
        assert os.path.isfile(os.path.join(good_dir, "result.txt"))
        assert "Done" in output

    def test_download_version_zero_uses_zero(self, api, capsys):
        """When version_number is 0 (unset), directory uses 'unset'."""
        task = _make_task(version_number=0)
        api._mock_benchmarks.get_benchmark_task.return_value = task
        _setup_runs_response(api, [_make_run(run_id=1)])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task")
        zippath = api.download_file.call_args[0][1]
        expected = os.path.join(".", "my-task", "unset", "gemini-pro", "1.zip")
        assert zippath == expected

    def test_download_include_source_flag(self, api, capsys):
        """--include-source passes include_source=True to the SDK request."""
        _setup_runs_response(api, [_make_run()])
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", include_source=True)
        request = api._mock_benchmarks.download_benchmark_task_run_output.call_args[0][0]
        assert request.include_source is True

    def test_download_include_source_default_false(self, api, capsys):
        """Without --include-source, include_source defaults to False."""
        _setup_runs_response(api, [_make_run()])
        self._mock_download(api)
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task")
        request = api._mock_benchmarks.download_benchmark_task_run_output.call_args[0][0]
        assert request.include_source is False

    def test_download_bad_zip_with_force_replaces_output(self, api, capsys, tmp_path):
        """BadZipFile with --force keeps raw zip; existing output is NOT deleted."""
        _setup_completed_task(api)
        _setup_runs_response(api, [_make_run(model="bad-model", run_id=10)])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()

        outdir = str(tmp_path / "out")
        existing_dir = os.path.join(outdir, "my-task", "1", "bad-model", "10")
        os.makedirs(existing_dir)
        # Write a sentinel file in existing output
        sentinel = os.path.join(existing_dir, "old_result.txt")
        with open(sentinel, "w") as f:
            f.write("previous run")

        def fake_download(response, outfile, http_client, quiet=False):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, "wb") as f:
                f.write(b"this is not a zip")

        api.download_file = MagicMock(side_effect=fake_download)

        api.benchmarks_tasks_download_cli("my-task", output=outdir, force=True)

        output = capsys.readouterr().out
        assert "bad-model" in output
        assert "Bad zip" in output
        assert "Done: 0 runs downloaded." in output
        # Raw zip file is kept
        zip_path = os.path.join(outdir, "my-task", "1", "bad-model", "10.zip")
        assert os.path.isfile(zip_path)
        # Existing output directory is preserved (not deleted by --force)
        assert os.path.isfile(sentinel)

    def test_download_all_bad_zips_shows_zero_downloaded(self, api, capsys, tmp_path):
        """When every download is a BadZipFile, summary says '0 downloaded'."""
        _setup_completed_task(api)
        _setup_runs_response(api, [_make_run(model="bad-model", run_id=10)])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()

        outdir = str(tmp_path / "out")

        def fake_download(response, outfile, http_client, quiet=False):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, "wb") as f:
                f.write(b"this is not a zip")

        api.download_file = MagicMock(side_effect=fake_download)

        api.benchmarks_tasks_download_cli("my-task", output=outdir)

        output = capsys.readouterr().out
        assert "bad-model" in output
        assert "Bad zip" in output
        assert "Done: 0 runs downloaded." in output


# ============================================================
# Log
# ============================================================


class TestLog:
    """``kaggle benchmarks tasks log <task> [-m <model> ...]``"""

    def _mock_log_response(self, api, content="log output", content_type="application/json"):
        """Set up a mock log response."""
        _setup_completed_task(api)
        response = MagicMock()
        response.headers = {"Content-Type": content_type}
        response.text = content
        response.iter_lines.return_value = []
        api._mock_benchmarks.get_benchmark_task_run_logs.return_value = response
        return response

    @pytest.mark.parametrize("status_code", [403, 404], ids=["forbidden", "not_found"])
    def test_log_task_not_found(self, api, status_code):
        """Log gives friendly error when task doesn't exist (403/404)."""
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=status_code))
        with pytest.raises(ValueError, match="not found"):
            api.benchmarks_tasks_log_cli("no-such-task")

    def test_log_no_runs(self, api, capsys):
        """No runs prints a helpful message and returns."""
        _setup_completed_task(api)
        _setup_runs_response(api, [])
        api.benchmarks_tasks_log_cli("my-task")
        output = capsys.readouterr().out
        assert "No runs found" in output

    def test_log_no_runs_with_model_filter(self, api, capsys):
        """No runs for a specific model prints descriptive message and returns."""
        _setup_completed_task(api)
        _setup_runs_response(api, [])
        api.benchmarks_tasks_log_cli("my-task", model=["nonexistent-model"])
        output = capsys.readouterr().out
        assert "No runs found" in output
        assert "nonexistent-model" in output

    def test_log_single_run_with_header(self, api, capsys):
        """Single run prints logs with model header including state."""
        _setup_runs_response(api, [_make_run(model="gemini-pro", run_id=1)])
        self._mock_log_response(api, content="hello world")
        api.benchmarks_tasks_log_cli("my-task")
        output = capsys.readouterr().out
        assert "hello world" in output
        assert "═══ Logs for gemini-pro (Run 1) [COMPLETED] ═══" in output
        assert "═══ (" in output  # line count footer
        assert "Showed logs for 1 run(s) across 1 model(s)." in output

    def test_log_multiple_runs_with_headers(self, api, capsys):
        """Multiple runs print logs with model headers and a summary."""
        _setup_runs_response(
            api,
            [
                _make_run(model="gemini-pro", run_id=1),
                _make_run(model="claude-4", run_id=2),
            ],
        )
        self._mock_log_response(api, content="log output")
        api.benchmarks_tasks_log_cli("my-task")
        output = capsys.readouterr().out
        assert "═══ Logs for gemini-pro (Run 1) [COMPLETED] ═══" in output
        assert "═══ Logs for claude-4 (Run 2) [COMPLETED] ═══" in output
        assert "Showed logs for 2 run(s) across 2 model(s)." in output

    def test_log_with_model_filter(self, api, capsys):
        """Model filter is passed to _fetch_task_runs."""
        _setup_runs_response(api, [_make_run(model="gemini-pro")])
        self._mock_log_response(api, content="filtered logs")
        api.benchmarks_tasks_log_cli("my-task", model=["gemini-pro"])
        request = api._mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemini-pro"]
        output = capsys.readouterr().out
        assert "filtered logs" in output

    def test_log_sse_stream(self, api, capsys):
        """SSE responses are streamed line by line."""
        _setup_runs_response(api, [_make_run()])
        _setup_completed_task(api)
        response = MagicMock()
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = [
            b"data: Starting benchmark...",
            b"",
            b"data: Running task function...",
            b"event: done",
        ]
        api._mock_benchmarks.get_benchmark_task_run_logs.return_value = response
        api.benchmarks_tasks_log_cli("my-task")
        output = capsys.readouterr().out
        assert "Starting benchmark..." in output
        assert "Running task function..." in output

    def test_log_json_response(self, api, capsys):
        """JSON responses are printed as text."""
        _setup_runs_response(api, [_make_run()])
        self._mock_log_response(api, content='{"logs": "some data"}')
        api.benchmarks_tasks_log_cli("my-task")
        output = capsys.readouterr().out
        assert '{"logs": "some data"}' in output

    def test_log_queued_run_server_error(self, api, capsys):
        """A QUEUED run whose log endpoint returns 404 prints a friendly message."""
        _setup_completed_task(api)
        _setup_runs_response(api, [_make_run(model="gemini-pro", state=RUN_QUEUED, run_id=1)])
        api._mock_benchmarks.get_benchmark_task_run_logs.side_effect = HTTPError(response=MagicMock(status_code=404))
        api.benchmarks_tasks_log_cli("my-task")
        output = capsys.readouterr().out
        assert "No logs available" in output
        assert "404" in output
        assert "0 lines" in output

    def test_log_json_list_with_data_key(self, api, capsys):
        """JSON response containing list of {"data": ...} entries prints each entry."""
        import json

        _setup_runs_response(api, [_make_run()])
        log_entries = [
            {"data": "line one\n"},
            {"data": "line two\n"},
        ]
        content = json.dumps(log_entries)
        self._mock_log_response(api, content=content)
        api.benchmarks_tasks_log_cli("my-task")
        output = capsys.readouterr().out
        # JSON log content is printed as raw text from response.text
        assert "line one" in output
        assert "line two" in output


# ============================================================
# download_file (Content-Length handling)
# ============================================================


class TestDownloadFile:
    """Tests for ``download_file`` handling of Content-Length header."""

    def _make_response(self, content=b"test data", headers=None, url="http://example.com/file"):
        """Build a mock requests.Response-like object."""
        resp = MagicMock()
        default_headers = {"Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"}
        if headers:
            default_headers.update(headers)
        resp.headers = default_headers
        resp.url = url
        resp.request.method = "GET"
        resp.request.headers = {}
        resp.iter_content = MagicMock(return_value=iter([content]))
        # Make type().__name__ return something other than "HTTPResponse"
        type(resp).__name__ = "Response"
        return resp

    def test_download_file_with_content_length(self, api, tmp_path):
        """Normal download with Content-Length header works and verifies size."""
        content = b"hello world"
        resp = self._make_response(
            content=content,
            headers={"Content-Length": str(len(content))},
        )
        outfile = str(tmp_path / "out" / "test.txt")
        api.download_file(resp, outfile, MagicMock(), quiet=True)
        assert os.path.isfile(outfile)
        with open(outfile, "rb") as f:
            assert f.read() == content

    def test_download_file_missing_content_length(self, api, tmp_path):
        """Chunked response (no Content-Length) downloads without crashing."""
        content = b"chunked data"
        resp = self._make_response(
            content=content,
            headers={"Transfer-Encoding": "chunked"},
        )
        outfile = str(tmp_path / "out" / "chunked.bin")
        api.download_file(resp, outfile, MagicMock(), quiet=True)
        assert os.path.isfile(outfile)
        with open(outfile, "rb") as f:
            assert f.read() == content

    def test_download_file_missing_content_length_skips_size_check(self, api, tmp_path):
        """When Content-Length is absent, size verification is skipped (no ValueError)."""
        content = b"data"
        resp = self._make_response(content=content)
        # No Content-Length in headers at all
        outfile = str(tmp_path / "out" / "nosize.bin")
        api.download_file(resp, outfile, MagicMock(), quiet=True)
        # Should succeed without ValueError about size mismatch
        assert os.path.isfile(outfile)


# ============================================================
# Model Slug Normalization
# ============================================================


class TestModelSlugNormalization:
    """Tests for model slug normalization helpers."""

    # -- _normalize_model_slug --

    @pytest.mark.parametrize(
        "input_slug, expected",
        [
            # Already canonical — no change
            ("gemini-pro", "gemini-pro"),
            ("grok-4.3", "grok-4.3"),
            # Provider prefix stripped
            ("xai/grok-4.3", "grok-4.3"),
            ("google/gemini-2.5-pro", "gemini-2.5-pro"),
            # @ replaced with -
            ("claude-sonnet-4-6@default", "claude-sonnet-4-6-default"),
            # Both prefix and @
            ("anthropic/claude-sonnet-4-6@default", "claude-sonnet-4-6-default"),
        ],
        ids=[
            "canonical_plain",
            "canonical_with_dot",
            "strip_xai_prefix",
            "strip_google_prefix",
            "at_to_dash",
            "prefix_and_at",
        ],
    )
    def test_normalize_model_slug(self, input_slug, expected):
        assert KaggleApi._normalize_model_slug(input_slug) == expected

    # -- _normalize_model_list --

    def test_normalize_model_list_none(self):
        assert KaggleApi._normalize_model_list(None) == []

    def test_normalize_model_list_single_string(self):
        assert KaggleApi._normalize_model_list("xai/grok-4.3") == ["grok-4.3"]

    def test_normalize_model_list_list_of_strings(self):
        result = KaggleApi._normalize_model_list(["xai/grok-4.3", "anthropic/claude-sonnet-4-6@default", "gemini-pro"])
        assert result == ["grok-4.3", "claude-sonnet-4-6-default", "gemini-pro"]

    # -- End-to-end: run sends normalized slugs to the API --

    def test_run_normalizes_prefixed_model_for_api(self, api, capsys):
        """Running with 'xai/grok-4.3' should send 'grok-4.3' to the server."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api.benchmarks_tasks_run_cli("my-task", ["xai/grok-4.3"])
        request = api._mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["grok-4.3"]

    def test_run_normalizes_at_sign_model_for_api(self, api, capsys):
        """Running with 'anthropic/claude-sonnet-4-6@default' normalizes to 'claude-sonnet-4-6-default'."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api.benchmarks_tasks_run_cli("my-task", ["anthropic/claude-sonnet-4-6@default"])
        request = api._mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["claude-sonnet-4-6-default"]

    def test_status_normalizes_model_filter(self, api, capsys):
        """Status with prefixed model filter normalizes slugs for the API request."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task", model="google/gemini-2.5-pro")
        request = api._mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemini-2.5-pro"]

    def test_download_normalizes_model_filter(self, api, capsys):
        """Download with prefixed model filter normalizes slugs for the API request."""
        _setup_completed_task(api)
        _setup_runs_response(api, [_make_run()])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
        with patch("zipfile.ZipFile"), patch("os.remove"), patch("os.rename"):
            api.benchmarks_tasks_download_cli("my-task", model="xai/grok-4.3")
        request = api._mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["grok-4.3"]


# ============================================================
# Models
# ============================================================


class TestModels:
    """``kaggle benchmarks tasks models``"""

    def test_models_lists_available(self, api, capsys):
        _setup_available_models(api, ["gemini-pro", "gemma-2b"])
        api.benchmarks_tasks_models_cli()
        output = capsys.readouterr().out
        assert "Slug" in output
        assert "Display Name" in output
        assert "gemini-pro" in output
        assert "gemma-2b" in output

    def test_models_empty(self, api, capsys):
        _setup_available_models(api, [])
        api.benchmarks_tasks_models_cli()
        output = capsys.readouterr().out
        assert "No benchmark models available" in output


# ============================================================
# Delete
# ============================================================


class TestDelete:
    """``kaggle benchmarks tasks delete <task> [-y]``"""

    @pytest.mark.parametrize("no_confirm", [False, True], ids=["default", "yes_flag"])
    def test_delete_prints_stub_message(self, api, capsys, no_confirm):
        """Delete always prints stub message; -y flag is accepted but has no effect."""
        api.benchmarks_tasks_delete_cli("my-task", no_confirm=no_confirm)
        assert "Delete is not supported by the server yet." in capsys.readouterr().out


# ============================================================
# CLI Arg Parsing
# ============================================================


class TestCliArgParsing:
    """Tests that argparse wiring for ``kaggle benchmarks tasks`` is correct."""

    def setup_method(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        subparsers = self.parser.add_subparsers(title="commands", dest="command")
        subparsers.required = True
        from kaggle.cli import parse_benchmarks

        parse_benchmarks(subparsers)

    def _parse(self, arg_string):
        return self.parser.parse_args(arg_string.split())

    @pytest.mark.parametrize(
        "cmd, expected",
        [
            # push
            (
                "benchmarks tasks push my-task -f ./task.py",
                {"task": "my-task", "file": "./task.py"},
            ),
            ("b t push my-task -f ./task.py", {"task": "my-task", "file": "./task.py"}),
            (
                "benchmarks tasks push my-task -f ./task.py --wait",
                {"task": "my-task", "file": "./task.py", "wait": 0},
            ),
            (
                "benchmarks tasks push my-task -f ./task.py --wait 60",
                {"task": "my-task", "file": "./task.py", "wait": 60},
            ),
            # run
            (
                "benchmarks tasks run my-task",
                {"task": "my-task", "model": None, "wait": None},
            ),
            (
                "benchmarks tasks run my-task -m gemini-3 --wait",
                {"model": ["gemini-3"], "wait": 0},
            ),
            (
                "benchmarks tasks run my-task -m gemini-3 --wait 60",
                {"model": ["gemini-3"], "wait": 60},
            ),
            (
                "benchmarks tasks run my-task -m gemini-3 -m gpt-5 -m claude-4",
                {"model": ["gemini-3", "gpt-5", "claude-4"]},
            ),
            ("b t run my-task -m gemini-3", {"task": "my-task", "model": ["gemini-3"]}),
            # list
            ("benchmarks tasks list", {"name_regex": None, "status": None}),
            ("benchmarks tasks list --name-regex ^math", {"name_regex": "^math"}),
            ("benchmarks tasks list --status completed", {"status": "completed"}),
            (
                "benchmarks tasks list --name-regex ^math --status errored",
                {"name_regex": "^math", "status": "errored"},
            ),
            # status
            ("benchmarks tasks status my-task", {"task": "my-task", "model": None}),
            (
                "benchmarks tasks status my-task -m gemini-3 -m gpt-5",
                {"task": "my-task", "model": ["gemini-3", "gpt-5"]},
            ),
            # download
            (
                "benchmarks tasks download my-task",
                {"task": "my-task", "model": None, "output": None, "include_source": False},
            ),
            ("benchmarks tasks download my-task -o ./results", {"output": "./results", "include_source": False}),
            (
                "benchmarks tasks download my-task -m gemini-3 -o ./results",
                {"model": ["gemini-3"], "output": "./results", "include_source": False},
            ),
            (
                "benchmarks tasks download my-task --include-source",
                {"task": "my-task", "include_source": True},
            ),
            (
                "benchmarks tasks download my-task -s -m gemini-3",
                {"model": ["gemini-3"], "include_source": True},
            ),
            # log
            (
                "benchmarks tasks log my-task",
                {"task": "my-task", "model": None},
            ),
            (
                "benchmarks tasks log my-task -m gemini-3",
                {"task": "my-task", "model": ["gemini-3"]},
            ),
            (
                "benchmarks tasks log my-task -m gemini-3 -m claude-4",
                {"model": ["gemini-3", "claude-4"]},
            ),
            (
                "benchmarks tasks logs my-task",
                {"task": "my-task", "model": None},
            ),
            ("b t log my-task", {"task": "my-task", "model": None}),
            # delete
            (
                "benchmarks tasks delete my-task",
                {"task": "my-task", "no_confirm": False},
            ),
            ("benchmarks tasks delete my-task -y", {"no_confirm": True}),
            ("benchmarks tasks delete my-task --yes", {"no_confirm": True}),
            # auth
            ("benchmarks auth", {"no_confirm": False, "env_file": ".env"}),
            ("benchmarks auth -y", {"no_confirm": True}),
            ("benchmarks auth --env-file custom.env", {"env_file": "custom.env"}),
            # init
            (
                "benchmarks init",
                {"no_confirm": False, "env_file": ".env", "example_file": "example_task.py"},
            ),
            ("benchmarks init -y", {"no_confirm": True}),
            ("benchmarks init --env-file custom.env", {"env_file": "custom.env"}),
            ("benchmarks init --example-file my_task.py", {"example_file": "my_task.py"}),
            # publish
            (
                "benchmarks tasks publish my-task",
                {"task": "my-task", "publish_backing_notebook": True},
            ),
            (
                "benchmarks tasks publish my-task --no-publish-backing-notebook",
                {"task": "my-task", "publish_backing_notebook": False},
            ),
            ("b t publish my-task", {"task": "my-task", "publish_backing_notebook": True}),
            # push with --kaggle-dataset
            (
                "benchmarks tasks push my-task -f ./task.py -d user/dataset1 -d user/dataset2",
                {"task": "my-task", "file": "./task.py", "kaggle_datasets": ["user/dataset1", "user/dataset2"]},
            ),
            (
                "benchmarks tasks push my-task -f ./task.py --kaggle-dataset user/ds",
                {"kaggle_datasets": ["user/ds"]},
            ),
            (
                "benchmarks tasks push my-task -f ./task.py",
                {"kaggle_datasets": None},
            ),
        ],
    )
    def test_parse_success(self, cmd, expected):
        args = self._parse(cmd)
        for key, val in expected.items():
            assert getattr(args, key) == val

    @pytest.mark.parametrize(
        "cmd",
        [
            "benchmarks tasks push my-task",  # missing required -f
            "benchmarks tasks run my-task -m",  # -m requires at least one arg
            "benchmarks tasks status my-task -m",  # -m requires at least one arg
            "benchmarks tasks download my-task -m",  # -m requires at least one arg
            "benchmarks tasks log my-task -m",  # -m requires at least one arg
        ],
    )
    def test_parse_error(self, cmd):
        with pytest.raises(SystemExit):
            self._parse(cmd)

    @pytest.mark.parametrize(
        "cmd, expected",
        [
            (
                "benchmarks tasks download my-task --force",
                {"task": "my-task", "force": True},
            ),
            (
                "benchmarks tasks download my-task -f",
                {"force": True},
            ),
            (
                "benchmarks tasks download my-task",
                {"force": False},
            ),
        ],
        ids=["force_long", "force_short", "force_default"],
    )
    def test_parse_download_force(self, cmd, expected):
        args = self._parse(cmd)
        for key, val in expected.items():
            assert getattr(args, key) == val


# ============================================================
# Benchmarks Auth
# ============================================================


def _make_token_response(
    base_uri="https://mp-staging.kaggle.net/models/openapi", token="kaggle-benchmarks:cool-token", expiry_time=None
):
    from datetime import datetime

    if expiry_time is None:
        expiry_time = datetime(2026, 4, 17, 12, 0, 0)
    response = ApiCreateDefaultModelProxyTokenResponse()
    response.base_uri = base_uri
    response.token = token
    response.expiry_time = expiry_time
    return response


class TestBenchmarksAuth:
    """Tests for ``kaggle benchmarks auth``."""

    def test_writes_env_file_with_yes_flag(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        api.benchmarks_auth_cli(no_confirm=True, env_file=env_file)
        content = (tmp_path / ".env").read_text()
        assert "MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi\n" in content
        assert "MODEL_PROXY_API_KEY=kaggle-benchmarks:cool-token\n" in content
        assert "MODEL_PROXY_EXPIRY_TIME=2026-04-17T12:00:00Z\n" in content
        out = capsys.readouterr().out
        assert "API Key  (ends in ...oken)" in out
        assert "kaggle-benchmarks:cool-token" not in out
        assert "have been written to" in out

    def test_aborted_on_no_confirm(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        with patch("builtins.input", return_value="no"):
            api.benchmarks_auth_cli(no_confirm=False, env_file=env_file)
        assert not (tmp_path / ".env").exists()
        out = capsys.readouterr().out
        assert "The following configuration will be set:" in out
        assert "have been written to" not in out

    def test_confirmed_on_yes(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        with patch("builtins.input", return_value="yes"):
            api.benchmarks_auth_cli(no_confirm=False, env_file=env_file)
        assert (tmp_path / ".env").exists()
        out = capsys.readouterr().out
        assert "have been written to" in out

    def test_appends_to_existing_file(self, api, mock_token, capsys, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=hello\n")
        api.benchmarks_auth_cli(no_confirm=True, env_file=str(env_file))
        content = env_file.read_text()
        assert content.startswith("EXISTING_VAR=hello\n")
        assert "MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi\n" in content

    def test_custom_env_file(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / "custom.env")
        api.benchmarks_auth_cli(no_confirm=True, env_file=env_file)
        assert (tmp_path / "custom.env").exists()
        out = capsys.readouterr().out
        assert "custom.env" in out

    def test_friendly_error_on_404_lists_both_causes(self, api, tmp_path):
        """404 maps to a message listing both beta-access and stale-CLI as possibilities."""
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.side_effect = HTTPError(
            response=MagicMock(status_code=404)
        )
        with pytest.raises(ValueError) as excinfo:
            api.benchmarks_auth_cli(no_confirm=True, env_file=str(tmp_path / ".env"))
        msg = str(excinfo.value)
        assert "currently in beta" in msg
        assert "pip install --upgrade kaggle" in msg

    def test_friendly_error_on_403_lists_both_causes(self, api, tmp_path):
        """403 maps to a message listing both verification and stale-credentials as possibilities."""
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.side_effect = HTTPError(
            response=MagicMock(status_code=403)
        )
        with pytest.raises(ValueError) as excinfo:
            api.benchmarks_auth_cli(no_confirm=True, env_file=str(tmp_path / ".env"))
        msg = str(excinfo.value)
        assert "phone or identity verification" in msg
        assert "https://www.kaggle.com/settings" in msg
        assert "stale or invalid" in msg
        assert "https://www.kaggle.com/settings/api" in msg

    @pytest.mark.parametrize("status_code", [401, 429, 500, 503])
    def test_other_http_errors_propagate_unchanged(self, api, tmp_path, status_code):
        """Status codes outside 403/404 propagate as HTTPError (no misleading verification text)."""
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.side_effect = HTTPError(
            response=MagicMock(status_code=status_code)
        )
        with pytest.raises(HTTPError):
            api.benchmarks_auth_cli(no_confirm=True, env_file=str(tmp_path / ".env"))

    def test_http_error_without_response_propagates(self, api, tmp_path):
        """An HTTPError with no response object (e.g. network error) propagates raw."""
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.side_effect = HTTPError(
            response=None
        )
        with pytest.raises(HTTPError):
            api.benchmarks_auth_cli(no_confirm=True, env_file=str(tmp_path / ".env"))

    def test_non_http_errors_propagate_unchanged(self, api, tmp_path):
        """Non-HTTP exceptions (e.g. connection errors) propagate as-is, no error wrapping."""
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.side_effect = ConnectionError(
            "DNS lookup failed"
        )
        with pytest.raises(ConnectionError, match="DNS lookup failed"):
            api.benchmarks_auth_cli(no_confirm=True, env_file=str(tmp_path / ".env"))


# ============================================================
# Benchmarks Init
# ============================================================


class TestBenchmarksInit:
    """Tests for ``kaggle benchmarks init``."""

    def test_writes_all_vars(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        content = (tmp_path / ".env").read_text()
        assert "MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi\n" in content
        assert "MODEL_PROXY_API_KEY=kaggle-benchmarks:cool-token\n" in content
        assert "MODEL_PROXY_EXPIRY_TIME=2026-04-17T12:00:00Z\n" in content
        assert "LLM_DEFAULT=google/gemini-3-flash-preview\n" in content
        assert "LLM_DEFAULT_EVAL=google/gemini-3-flash-preview\n" in content
        assert (
            "LLMS_AVAILABLE=anthropic/claude-haiku-4-5@20251001,deepseek-ai/deepseek-v3.2,google/gemini-3-flash-preview,google/gemini-3.1-flash-lite-preview,openai/gpt-oss-120b,qwen/qwen3-next-80b-a3b-instruct,zai/glm-5\n"
            in content
        )
        out = capsys.readouterr().out
        assert "API Key  (ends in ...oken)" in out
        assert "Default LLM      google/gemini-3-flash-preview" in out
        assert "Environment initialized!" in out

    def test_writes_example_file(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        content = (tmp_path / "example_task.py").read_text()
        assert "import kaggle_benchmarks as kbench" in content
        assert "kaggle_benchmarks_reference.md" in content
        out = capsys.readouterr().out
        assert "example_task.py" in out
        assert "Starter template" in out

    def test_writes_reference_file(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        ref_file = tmp_path / "kaggle_benchmarks_reference.md"
        assert ref_file.exists()
        content = ref_file.read_text()
        assert "kaggle-benchmarks Task Syntax Reference" in content
        out = capsys.readouterr().out
        assert "kaggle_benchmarks_reference.md" in out
        assert "Syntax guide" in out

    def test_skips_reference_file_if_exists(self, api, mock_token, capsys, tmp_path):
        ref_file = tmp_path / "kaggle_benchmarks_reference.md"
        ref_file.write_text("existing content\n")
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=str(example_file))
        assert ref_file.read_text() == "existing content\n"
        out = capsys.readouterr().out
        assert "Environment initialized!" in out

    def test_skips_example_file_if_exists(self, api, mock_token, capsys, tmp_path):
        example_file = tmp_path / "example_task.py"
        example_file.write_text("existing content\n")
        env_file = str(tmp_path / ".env")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=str(example_file))
        assert example_file.read_text() == "existing content\n"
        out = capsys.readouterr().out
        assert "Environment initialized!" in out

    def test_custom_example_file(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "my_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        content = (tmp_path / "my_task.py").read_text()
        assert "import kaggle_benchmarks as kbench" in content

    def test_aborted_on_no_confirm(self, api, mock_token, capsys, tmp_path):
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        with patch("builtins.input", return_value="no"):
            api.benchmarks_init_cli(no_confirm=False, env_file=env_file, example_file=example_file)
        assert not (tmp_path / ".env").exists()

    def test_appends_to_existing_file(self, api, mock_token, capsys, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=hello\n")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=str(env_file), example_file=example_file)
        content = env_file.read_text()
        assert content.startswith("EXISTING_VAR=hello\n")
        assert "LLM_DEFAULT=google/gemini-3-flash-preview\n" in content

    def test_friendly_error_on_404_lists_both_causes(self, api, tmp_path):
        """404 maps to a message listing both beta-access and stale-CLI as possibilities."""
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.side_effect = HTTPError(
            response=MagicMock(status_code=404)
        )
        with pytest.raises(ValueError) as excinfo:
            api.benchmarks_init_cli(
                no_confirm=True,
                env_file=str(tmp_path / ".env"),
                example_file=str(tmp_path / "example_task.py"),
            )
        msg = str(excinfo.value)
        assert "currently in beta" in msg
        assert "pip install --upgrade kaggle" in msg

    def test_friendly_error_on_403_lists_both_causes(self, api, tmp_path):
        """403 maps to a message listing both verification and stale-credentials as possibilities."""
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.side_effect = HTTPError(
            response=MagicMock(status_code=403)
        )
        with pytest.raises(ValueError) as excinfo:
            api.benchmarks_init_cli(
                no_confirm=True,
                env_file=str(tmp_path / ".env"),
                example_file=str(tmp_path / "example_task.py"),
            )
        msg = str(excinfo.value)
        assert "phone or identity verification" in msg
        assert "https://www.kaggle.com/settings" in msg
        assert "stale or invalid" in msg
        assert "https://www.kaggle.com/settings/api" in msg


# ============================================================
# Upsert behavior (auth/init reruns)
# ============================================================


class TestBenchmarksUpsert:
    """End-to-end checks that ``kaggle benchmarks auth/init`` upserts the
    managed keys rather than appending duplicates. The line-by-line upsert
    semantics are owned by ``python-dotenv``; these tests only verify wiring
    and the user-facing contract."""

    def test_rerun_with_new_token_replaces_stale_value(self, api, tmp_path):
        # First run: original token.
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response(token="kaggle-benchmarks:first")
        )
        env_file = tmp_path / ".env"
        api.benchmarks_auth_cli(no_confirm=True, env_file=str(env_file))

        # Second run: refreshed token.
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response(token="kaggle-benchmarks:second")
        )
        api.benchmarks_auth_cli(no_confirm=True, env_file=str(env_file))

        content = env_file.read_text()
        assert "MODEL_PROXY_API_KEY=kaggle-benchmarks:second\n" in content
        assert "kaggle-benchmarks:first" not in content
        assert content.count("MODEL_PROXY_API_KEY=") == 1

    def test_preserves_user_added_keys_and_comments(self, api, mock_token, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# my notes\nMY_SECRET=hunter2\nMODEL_PROXY_URL=https://stale.example.com\n")
        api.benchmarks_auth_cli(no_confirm=True, env_file=str(env_file))
        content = env_file.read_text()
        assert "# my notes\n" in content
        assert "MY_SECRET=hunter2\n" in content
        assert "stale.example.com" not in content
        assert content.count("MODEL_PROXY_URL=") == 1

    def test_first_rerun_after_upgrade_refreshes_stale_duplicates(self, api, tmp_path):
        """Migration scenario: a user who ran the old append-based CLI ends up
        with several stacked ``MODEL_PROXY_API_KEY=`` lines in their ``.env``.
        The first rerun on the new code must rewrite every duplicate to the
        fresh value so no stale token lingers in the file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "MODEL_PROXY_API_KEY=stale-1\n"
            "OTHER=keep\n"
            "MODEL_PROXY_API_KEY=stale-2\n"
            "MODEL_PROXY_API_KEY=stale-3\n"
        )
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response(token="kaggle-benchmarks:fresh")
        )
        api.benchmarks_auth_cli(no_confirm=True, env_file=str(env_file))

        content = env_file.read_text()
        for stale in ("stale-1", "stale-2", "stale-3"):
            assert stale not in content
        from dotenv import dotenv_values

        assert dotenv_values(str(env_file))["MODEL_PROXY_API_KEY"] == "kaggle-benchmarks:fresh"
        assert "OTHER=keep\n" in content

    def test_init_rerun_is_idempotent(self, api, mock_token, tmp_path):
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        first = (tmp_path / ".env").read_text()
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        second = (tmp_path / ".env").read_text()
        assert first == second
        for key in (
            "MODEL_PROXY_URL=",
            "MODEL_PROXY_API_KEY=",
            "MODEL_PROXY_EXPIRY_TIME=",
            "LLM_DEFAULT=",
            "LLM_DEFAULT_EVAL=",
            "LLMS_AVAILABLE=",
        ):
            assert second.count(key) == 1


# ============================================================
# Expiry Formatting
# ============================================================


class TestFormatExpiry:
    """Tests for ``KaggleApi._format_expiry`` static helper."""

    _NOW = "2026-05-23T12:00:00"  # frozen "now" in UTC

    @pytest.fixture(autouse=True)
    def _freeze_now(self):
        from datetime import datetime as real_datetime, timezone

        frozen = real_datetime.fromisoformat(self._NOW).replace(tzinfo=timezone.utc)

        class FrozenDatetime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen if tz is None else frozen.astimezone(tz)

        with patch("kaggle.api.kaggle_api_extended.datetime", FrozenDatetime):
            yield

    @pytest.mark.parametrize(
        "expiry_iso, expected",
        [
            ("2026-05-23T11:00:00Z", "Expired"),
            ("2026-05-23T12:00:00Z", "Expired"),
            ("2026-05-23T12:00:30Z", "In 1 minute"),
            ("2026-05-23T12:05:00Z", "In 5 minutes"),
            ("2026-05-23T13:00:00Z", "In 1 hour"),
            ("2026-05-24T00:00:00Z", "In 12 hours"),
            ("2026-05-24T12:00:00Z", "In 1 day"),
            ("2026-05-30T12:00:00Z", "In 7 days"),
        ],
    )
    def test_relative_buckets(self, expiry_iso, expected):
        assert KaggleApi._format_expiry(expiry_iso) == expected

    def test_unparseable_falls_back_to_raw(self):
        assert KaggleApi._format_expiry("not-a-timestamp") == "not-a-timestamp"


# ============================================================
# Task Name Detection
# ============================================================


class TestGetTaskNamesFromFile:
    """Tests for ``_get_task_names_from_file`` static method."""

    @pytest.mark.parametrize(
        "source, expected",
        [
            # keyword arg
            ('@task(name="My Task")\ndef evaluate(): pass\n', ["My Task"]),
            # positional arg
            ('@task("My Task")\ndef evaluate(): pass\n', ["My Task"]),
            # no name — falls back to function name
            ("@task()\ndef my_eval(): pass\n", ["My Eval"]),
            # bare decorator (no parens)
            ("@task\ndef my_eval(): pass\n", ["My Eval"]),
            # module-qualified: benchmarks.task
            ('@benchmarks.task(name="Qualified")\ndef f(): pass\n', ["Qualified"]),
            # module-qualified positional
            ('@benchmarks.task("Qualified")\ndef f(): pass\n', ["Qualified"]),
            # async function
            ('@task(name="Async")\nasync def evaluate(): pass\n', ["Async"]),
            # non-constant name expression — falls back to function name
            ("@task(name=TASK_NAME)\ndef my_task(): pass\n", ["My Task"]),
            # non-constant positional arg — falls back to function name
            ("@task(TASK_NAME)\ndef my_task(): pass\n", ["My Task"]),
            # multiple tasks in one file
            (
                '@task("First")\ndef a(): pass\n\n@task(name="Second")\ndef b(): pass\n',
                ["First", "Second"],
            ),
            # syntax error → empty list
            ("def broken(\n", []),
            # no decorators at all
            ("def plain(): pass\n", []),
            # unrelated decorator
            ("@other_decorator\ndef f(): pass\n", []),
            # keyword takes priority when both name= and positional could exist
            # (not valid Python for task(), but tests the keyword-first logic)
            ('@task("Pos", name="Kw")\ndef f(): pass\n', ["Kw"]),
            # file with IPython line magic
            ('!pip install numpy\n@task("t1")\ndef f(): pass\n', ["t1"]),
            # file with cell magic and body
            (
                '%%writefile out.csv\n1,2,3\n4,5,6\n\n@task("t2")\ndef g(): pass\n',
                ["t2"],
            ),
            # file with Jupytext cell markers (# %%) should NOT be stripped
            (
                '# %%\nimport os\n\n# %%\n@task("t3")\ndef h(): pass\n',
                ["t3"],
            ),
        ],
        ids=[
            "keyword_name",
            "positional_name",
            "no_name_parens",
            "bare_decorator",
            "module_qualified_keyword",
            "module_qualified_positional",
            "async_function",
            "non_constant_keyword",
            "non_constant_positional",
            "multiple_tasks",
            "syntax_error",
            "no_decorators",
            "unrelated_decorator",
            "keyword_over_positional",
            "with_line_magic",
            "with_cell_magic",
            "with_jupytext_markers",
        ],
    )
    def test_task_name_detection(self, source, expected):
        assert KaggleApi._get_task_names_from_file(source) == expected


# ============================================================
# IPython Magic Stripping
# ============================================================


class TestStripIpythonMagics:
    """Tests for ``_strip_ipython_magics`` static method."""

    def test_strips_line_magic(self):
        source = "%matplotlib inline\nimport os\n"
        result = KaggleApi._strip_ipython_magics(source)
        assert "%matplotlib" not in result
        assert "import os" in result

    def test_strips_shell_escape(self):
        source = "!pip install numpy\nimport numpy\n"
        result = KaggleApi._strip_ipython_magics(source)
        assert "!pip" not in result
        assert "import numpy" in result

    def test_strips_cell_magic_with_body(self):
        source = "%%writefile out.csv\n1,2,3\n4,5,6\n\nimport os\n"
        result = KaggleApi._strip_ipython_magics(source)
        assert "%%writefile" not in result
        assert "1,2,3" not in result
        assert "import os" in result

    def test_preserves_jupytext_cell_markers(self):
        """``# %%`` markers are NOT magics and must be preserved."""
        source = "# %%\nimport os\n\n# %%\nx = 1\n"
        result = KaggleApi._strip_ipython_magics(source)
        assert result == source

    def test_preserves_line_count(self):
        """Stripped lines are replaced with blanks to keep line numbers stable."""
        source = "!pip install foo\nimport os\n%%writefile a.txt\nhello\n\nx = 1\n"
        assert source.count("\n") == KaggleApi._strip_ipython_magics(source).count("\n")

    def test_mixed_magics_and_code(self):
        """A realistic Jupytext percent-format file."""
        source = (
            "# %%\n"
            "!pip install kaggle_benchmarks\n"
            "\n"
            "# %%\n"
            "import kaggle_benchmarks as kb\n"
            "\n"
            "# %%\n"
            '@kb.task("ask")\n'
            "def my_task(llm):\n"
            "    pass\n"
            "\n"
            "my_task.run(kb.llm)\n"
            "\n"
            "# %%\n"
            "%%writefile a.csv\n"
            "1,2,3\n"
            "4,5,6\n"
        )
        result = KaggleApi._strip_ipython_magics(source)
        # Code is preserved
        assert "import kaggle_benchmarks" in result
        assert '@kb.task("ask")' in result
        assert "my_task.run" in result
        # Magics are gone
        assert "!pip" not in result
        assert "%%writefile" not in result
        assert "1,2,3" not in result
        # Line count preserved
        assert source.count("\n") == result.count("\n")

    def test_empty_source(self):
        assert KaggleApi._strip_ipython_magics("") == ""

    def test_no_magics(self):
        source = "import os\nx = 1\n"
        assert KaggleApi._strip_ipython_magics(source) == source


# ============================================================
# _truncate / _format_modalities helpers
# ============================================================


class TestTruncate:
    """Tests for ``KaggleApi._truncate``."""

    @pytest.mark.parametrize(
        "value, max_len, expected",
        [
            ("hello", 10, "hello"),  # short string passes through
            ("hello", 5, "hello"),  # exactly max length
            ("hello world", 5, "hell…"),  # truncated to max_len - 1 + ellipsis
            ("", 5, ""),  # empty string
            ("a" * 100, 3, "aa…"),  # very long input
        ],
    )
    def test_truncate(self, value, max_len, expected):
        assert KaggleApi._truncate(value, max_len) == expected


class TestFormatModalities:
    """Tests for ``KaggleApi._format_modalities``."""

    def _make_version(self, inputs=None, outputs=None):
        version = MagicMock()
        version.input_modalities = [MagicMock(name=f"MODALITY_{n}") for n in (inputs or [])]
        for mock_obj, name in zip(version.input_modalities, inputs or []):
            mock_obj.name = f"MODALITY_{name}"
        version.output_modalities = [MagicMock() for _ in (outputs or [])]
        for mock_obj, name in zip(version.output_modalities, outputs or []):
            mock_obj.name = f"MODALITY_{name}"
        return version

    def test_text_to_text(self):
        v = self._make_version(inputs=["TEXT"], outputs=["TEXT"])
        assert KaggleApi._format_modalities(v) == "Text-to-Text"

    def test_image_text_to_text_sorted_alphabetically(self):
        v = self._make_version(inputs=["TEXT", "IMAGE"], outputs=["TEXT"])
        assert KaggleApi._format_modalities(v) == "Image-Text-to-Text"

    def test_any_to_any_when_both_have_three_or_more_matching(self):
        v = self._make_version(
            inputs=["TEXT", "IMAGE", "AUDIO", "VIDEO"],
            outputs=["TEXT", "IMAGE", "AUDIO", "VIDEO"],
        )
        assert KaggleApi._format_modalities(v) == "Any-to-Any"

    def test_unspecified_modality_skipped(self):
        v = self._make_version(inputs=["TEXT", "UNSPECIFIED"], outputs=["TEXT"])
        assert KaggleApi._format_modalities(v) == "Text-to-Text"

    def test_missing_attributes_returns_empty(self):
        v = MagicMock(spec=[])  # spec=[] -> no attributes -> getattr returns None
        assert KaggleApi._format_modalities(v) == ""

    def test_non_iterable_modalities_tolerated(self):
        """Older API responses or test mocks may not have iterable modality lists."""
        v = MagicMock()
        v.input_modalities = MagicMock()  # truthy but not iterable as a list of enums
        v.input_modalities.__iter__ = MagicMock(side_effect=TypeError)
        v.output_modalities = MagicMock()
        v.output_modalities.__iter__ = MagicMock(side_effect=TypeError)
        assert KaggleApi._format_modalities(v) == ""


# Publish
# ============================================================


class TestPublish:
    """``kaggle benchmarks tasks publish <task> [--no-publish-backing-notebook]``"""

    def _setup_publish(self, api, slug="my-task", is_public=False, is_notebook_published=False):
        """Wire up get + publish mocks."""
        task = _make_task(slug=slug)
        task.is_public = is_public
        task.is_backing_notebook_published = is_notebook_published
        api._mock_benchmarks.get_benchmark_task.return_value = task

        resp = _make_task(slug=slug)
        resp.is_public = True
        resp.is_backing_notebook_published = is_notebook_published
        api._mock_benchmarks.publish_benchmark_task.return_value = resp
        return task, resp

    def test_publish_success(self, api, capsys):
        """Publish a task successfully."""
        self._setup_publish(api)
        api.benchmarks_tasks_publish_cli("my-task")
        output = capsys.readouterr().out
        assert "published successfully" in output
        assert "Task URL:" in output

    def test_publish_already_public(self, api, capsys):
        """Publishing an already-public task (without notebook) prints info and returns."""
        self._setup_publish(api, is_public=True)
        api.benchmarks_tasks_publish_cli("my-task", publish_backing_notebook=False)
        output = capsys.readouterr().out
        assert "already public" in output
        api._mock_benchmarks.publish_benchmark_task.assert_not_called()

    def test_publish_already_public_notebook_not_published(self, api, capsys):
        """Already-public task with unpublished notebook proceeds to publish notebook."""
        self._setup_publish(api, is_public=True, is_notebook_published=False)
        api.benchmarks_tasks_publish_cli("my-task", publish_backing_notebook=True)
        output = capsys.readouterr().out
        assert "already public" in output
        assert "Publishing the backing notebook" in output
        api._mock_benchmarks.publish_benchmark_task.assert_called_once()

    def test_publish_already_public_notebook_already_published(self, api, capsys):
        """Already-public task with already-published notebook returns early."""
        self._setup_publish(api, is_public=True, is_notebook_published=True)
        api.benchmarks_tasks_publish_cli("my-task", publish_backing_notebook=True)
        output = capsys.readouterr().out
        assert "already public" in output
        assert "already published" in output
        api._mock_benchmarks.publish_benchmark_task.assert_not_called()

    def test_publish_with_backing_notebook(self, api, capsys):
        """Publish task and its backing notebook (default behavior)."""
        task, resp = self._setup_publish(api)
        resp.is_backing_notebook_published = True
        api.benchmarks_tasks_publish_cli("my-task")
        request = api._mock_benchmarks.publish_benchmark_task.call_args[0][0]
        assert request.publish_backing_notebook is True
        output = capsys.readouterr().out
        assert "Backing notebook also published" in output

    def test_publish_without_backing_notebook(self, api, capsys):
        """Publish with --no-publish-backing-notebook skips notebook."""
        self._setup_publish(api)
        api.benchmarks_tasks_publish_cli("my-task", publish_backing_notebook=False)
        request = api._mock_benchmarks.publish_benchmark_task.call_args[0][0]
        assert request.publish_backing_notebook is False

    def test_publish_with_backing_notebook_no_notebook(self, api, capsys):
        """Publish when no backing notebook exists."""
        task, resp = self._setup_publish(api)
        resp.is_backing_notebook_published = False
        api.benchmarks_tasks_publish_cli("my-task", publish_backing_notebook=True)
        output = capsys.readouterr().out
        assert "No backing notebook is associated" in output

    @pytest.mark.parametrize("status_code", [403, 404], ids=["forbidden", "not_found"])
    def test_publish_task_not_found(self, api, status_code):
        """Publish gives friendly error when task doesn't exist."""
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=status_code))
        with pytest.raises(ValueError, match="not found"):
            api.benchmarks_tasks_publish_cli("no-such-task")

    def test_publish_normalizes_task_name(self, api, capsys):
        """Publish slugifies the task name."""
        self._setup_publish(api)
        api.benchmarks_tasks_publish_cli("My Task")
        request = api._mock_benchmarks.publish_benchmark_task.call_args[0][0]
        assert request.slug.task_slug == "my-task"

    def test_publish_server_error_propagates(self, api):
        """Non-403/404 server errors (e.g. 500) from publish propagate as HTTPError."""
        task = _make_task()
        task.is_public = False
        api._mock_benchmarks.get_benchmark_task.return_value = task
        api._mock_benchmarks.publish_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=500))
        with pytest.raises(HTTPError):
            api.benchmarks_tasks_publish_cli("my-task")
