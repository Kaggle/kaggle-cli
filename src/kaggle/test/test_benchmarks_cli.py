"""Tests for ``kaggle benchmarks tasks`` CLI commands.

Organized by command (matching the spec):
  TestPush      – ``kaggle benchmarks tasks push <task> -f <file>``
  TestRun       – ``kaggle benchmarks tasks run <task> [-m ...] [--wait]``
  TestList      – ``kaggle benchmarks tasks list [--name-regex] [--status]``
  TestStatus    – ``kaggle benchmarks tasks status <task> [-m ...]``
  TestDownload  – ``kaggle benchmarks tasks download <task> [-m ...] [-o ...]``
  TestDelete    – ``kaggle benchmarks tasks delete <task> [-y]``
  TestCliArgParsing – argparse wiring for all subcommands
"""

import argparse
import os
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


def _make_task(slug="my-task", state=COMPLETED, create_time="2026-04-06 10:00:00", url=None):
    t = MagicMock()
    t.slug.task_slug = slug
    t.creation_state = state
    t.create_time = create_time
    t.url = url if url is not None else f"/benchmarks/{slug}"
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
    resp.error_message = None
    resp.errorMessage = None
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


def _setup_list_response(api, tasks, paginated_responses=None):
    """Set up list tasks response.

    If *paginated_responses* is provided, it should be a list of
    (tasks_list, next_page_token) tuples for multi-page scenarios.
    Otherwise a single-page response is created from *tasks*.
    """
    if paginated_responses:
        side_effects = []
        for page_tasks, token in paginated_responses:
            resp = MagicMock()
            resp.tasks = page_tasks
            resp.next_page_token = token
            side_effects.append(resp)
        api._mock_benchmarks.list_benchmark_tasks.side_effect = side_effects
    else:
        resp = MagicMock()
        resp.tasks = tasks
        resp.next_page_token = ""
        api._mock_benchmarks.list_benchmark_tasks.return_value = resp


def _setup_runs_response(api, runs, paginated_responses=None):
    """Set up runs response.

    If *paginated_responses* is provided, it should be a list of
    (runs_list, next_page_token) tuples for multi-page scenarios.
    Otherwise a single-page response is created from *runs*.
    """
    if paginated_responses:
        side_effects = []
        for page_runs, token in paginated_responses:
            resp = MagicMock()
            resp.runs = page_runs
            resp.next_page_token = token
            side_effects.append(resp)
        api._mock_benchmarks.list_benchmark_task_runs.side_effect = side_effects
    else:
        resp = MagicMock()
        resp.runs = runs
        resp.next_page_token = ""
        api._mock_benchmarks.list_benchmark_task_runs.return_value = resp


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
            ("my-task", "task.txt", "hello", "must be a .py"),
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
        assert f"Task '{expected_slug}' pushed." in output
        assert "Task URL:" in output
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
        assert "Task 'my-task' pushed." in capsys.readouterr().out

    def test_push_prefixes_relative_url(self, api, tmp_path, capsys):
        """If url starts with '/', prefix https://www.kaggle.com."""
        filepath = _write_task_file(tmp_path)
        resp = MagicMock()
        resp.url = "/benchmarks/my-task"
        resp.error_message = None
        resp.errorMessage = None
        api._mock_benchmarks.create_benchmark_task.return_value = resp
        _setup_completed_task(api)
        _push(api, "my-task", filepath)
        assert "https://www.kaggle.com/benchmarks/my-task" in capsys.readouterr().out

    # -- Server edge cases --

    @pytest.mark.parametrize("state", [QUEUED, RUNNING], ids=["queued", "running"])
    def test_push_rejects_pending_task(self, api, tmp_path, state):
        """Push rejects when the task version is still being created."""
        filepath = _write_task_file(tmp_path)
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task(state=state)
        with pytest.raises(ValueError, match="currently being created"):
            _push(api, "my-task", filepath)

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
        resp.error_message = "Some backend error"
        api._mock_benchmarks.create_benchmark_task.return_value = resp

        with pytest.raises(ValueError, match="Failed to push task: Some backend error"):
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
        assert "Waiting for task to be processed" in output
        assert "Task 'my-task' creation completed." in output

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
        assert "Timed out waiting for task creation after 30 seconds" in output


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
        assert "To check status later, use: kaggle b t status" in output
        for m in models:
            assert f"{m}: Scheduled" in output

    def test_run_reports_skipped_with_reason(self, api, capsys):
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result(scheduled=False, skipped_reason="Already running")])
        api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        output = capsys.readouterr().out
        assert "gemini-pro: Skipped" in output
        assert "Already running" in output

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
        assert "To check status later" not in output

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

    def test_run_selects_all_models(self, api):
        _setup_completed_task(api)
        _setup_available_models(api, ["gemini-pro", "gemma-2b"])
        _setup_batch_schedule(api, [])
        with patch("builtins.input", return_value="all"):
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
            with pytest.raises(ValueError, match="Invalid selection"):
                api.benchmarks_tasks_run_cli("my-task")

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

    def test_run_wait_times_out(self, api, capsys):
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.return_value = MagicMock(
            runs=[_make_run(state=RUN_RUNNING)], next_page_token=""
        )
        with patch("time.sleep"), patch("time.time", side_effect=[1000, 1060]):
            api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=30)
        output = capsys.readouterr().out
        assert "Timed out waiting for runs after 30 seconds" in output

    def test_run_wait_shows_errored_runs(self, api, capsys):
        """ERRORED runs display with ERRORED label."""
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result()])
        api._mock_benchmarks.list_benchmark_task_runs.return_value = MagicMock(
            runs=[_make_run(state=RUN_ERRORED)], next_page_token=""
        )
        with patch("time.sleep"):
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
        assert "my-task" in output

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

    def test_list_empty(self, api, capsys):
        """Empty task list still prints the header."""
        _setup_list_response(api, [])
        api.benchmarks_tasks_list_cli()
        output = capsys.readouterr().out
        assert "Task" in output
        # No task rows
        assert "my-task" not in output

    def test_list_table_format(self, api, capsys):
        """Table uses 40/20/20 column widths and 80-char separator."""
        _setup_list_response(api, [_make_task()])
        api.benchmarks_tasks_list_cli()
        output = capsys.readouterr().out
        assert "-" * 80 in output


# ============================================================
# Status
# ============================================================


class TestStatus:
    """``kaggle benchmarks tasks status <task> [-m <model> ...]``"""

    def test_status_header(self, api, capsys):
        """Status prints Task/Status/Created header."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "Task:" in output
        assert "Status:" in output
        assert "Created:" in output
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

    def test_status_with_model_filter(self, api, capsys):
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task", model="gemini-3")
        request = api._mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemini-3"]

    def test_status_with_multiple_models_filter(self, api, capsys):
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [])
        api.benchmarks_tasks_status_cli("my-task", model=["gemini-3", "gpt-5"])
        request = api._mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        assert request.model_version_slugs == ["gemini-3", "gpt-5"]

    def test_status_run_table(self, api, capsys):
        """Completed run renders with correct columns."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(api, [_make_run(model="gemini-pro", run_id=42)])
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "gemini-pro" in output
        assert "https://www.kaggle.com/benchmarks/runs/42" not in output

    def test_status_errored_run_shows_error_message(self, api, capsys):
        """ERRORED runs with error_message append ' | Error: ...'."""
        api._mock_benchmarks.get_benchmark_task.return_value = _make_task()
        _setup_runs_response(
            api,
            [_make_run(model="gemma-2b", state=RUN_ERRORED, run_id=43, error_message="OOM")],
        )
        api.benchmarks_tasks_status_cli("my-task")
        output = capsys.readouterr().out
        assert "| Error: OOM" in output

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

    def test_download_to_specific_output(self, api, capsys):
        _setup_runs_response(api, [_make_run()])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
        api.benchmarks_tasks_download_cli("my-task", output="my_output_dir")
        output = capsys.readouterr().out
        assert "Downloading output for run" in output
        assert "Downloaded output for gemini-pro to" in output
        assert "my_output_dir" in output

    def test_download_default_output_path(self, api, capsys):
        """Default output directory is ./<task>/output."""
        _setup_runs_response(api, [_make_run(run_id=1)])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
        api.benchmarks_tasks_download_cli("my-task")
        # Check the outfile passed to download_file
        call_args = api.download_file.call_args
        outfile = call_args[0][1]
        expected = os.path.join(".", "my-task", "output", "gemini-pro_1")
        assert outfile == expected

    def test_download_with_model_filter(self, api, capsys):
        _setup_runs_response(api, [_make_run()])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
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
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
        api.benchmarks_tasks_download_cli("my-task")
        # Only the completed run should be downloaded
        assert api._mock_benchmarks.download_benchmark_task_run_output.call_count == 1
        output = capsys.readouterr().out
        assert "done-model" in output
        assert "queued-model" not in output
        assert "running-model" not in output

    def test_download_includes_errored_runs(self, api, capsys):
        """ERRORED runs are also downloadable per spec."""
        _setup_runs_response(api, [_make_run(state=RUN_ERRORED)])
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
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
        api._mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        api.download_file = MagicMock()
        api.benchmarks_tasks_download_cli("my-task")
        assert api._mock_benchmarks.download_benchmark_task_run_output.call_count == 2


# ============================================================
# Delete
# ============================================================


class TestDelete:
    """``kaggle benchmarks tasks delete <task> [-y]``"""

    def test_delete_prints_stub_message(self, api, capsys):
        api.benchmarks_tasks_delete_cli("my-task")
        assert "Delete is not supported by the server yet." in capsys.readouterr().out

    def test_delete_accepts_no_confirm_flag(self, api, capsys):
        """The -y flag is accepted but has no effect (stub)."""
        api.benchmarks_tasks_delete_cli("my-task", no_confirm=True)
        assert "Delete is not supported" in capsys.readouterr().out


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
                "benchmarks tasks run my-task -m gemini-3 gpt-5 claude-4",
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
                "benchmarks tasks status my-task -m gemini-3 gpt-5",
                {"task": "my-task", "model": ["gemini-3", "gpt-5"]},
            ),
            # download
            (
                "benchmarks tasks download my-task",
                {"task": "my-task", "model": None, "output": None},
            ),
            ("benchmarks tasks download my-task -o ./results", {"output": "./results"}),
            (
                "benchmarks tasks download my-task -m gemini-3 -o ./results",
                {"model": ["gemini-3"], "output": "./results"},
            ),
            # delete
            (
                "benchmarks tasks delete my-task",
                {"task": "my-task", "no_confirm": False},
            ),
            ("benchmarks tasks delete my-task -y", {"no_confirm": True}),
            ("benchmarks tasks delete my-task --yes", {"no_confirm": True}),
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
        ],
    )
    def test_parse_error(self, cmd):
        with pytest.raises(SystemExit):
            self._parse(cmd)

    def test_parse_benchmarks_auth(self):
        args = self._parse("benchmarks auth")
        assert args.no_confirm is False
        assert args.env_file == ".env"

    def test_parse_benchmarks_auth_yes(self):
        args = self._parse("benchmarks auth -y")
        assert args.no_confirm is True

    def test_parse_benchmarks_auth_env_file(self):
        args = self._parse("benchmarks auth --env-file custom.env")
        assert args.env_file == "custom.env"

    def test_parse_benchmarks_init(self):
        args = self._parse("benchmarks init")
        assert args.no_confirm is False
        assert args.env_file == ".env"

    def test_parse_benchmarks_init_yes(self):
        args = self._parse("benchmarks init -y")
        assert args.no_confirm is True

    def test_parse_benchmarks_init_env_file(self):
        args = self._parse("benchmarks init --env-file custom.env")
        assert args.env_file == "custom.env"


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

    def test_writes_env_file_with_yes_flag(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        api.benchmarks_auth_cli(no_confirm=True, env_file=env_file)
        content = (tmp_path / ".env").read_text()
        assert "MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi\n" in content
        assert "MODEL_PROXY_API_KEY=kaggle-benchmarks:cool-token\n" in content
        assert "MODEL_PROXY_EXPIRY_TIME=2026-04-17T12:00:00Z\n" in content
        out = capsys.readouterr().out
        assert "MODEL_PROXY_API_KEY=****************oken" in out
        assert "kaggle-benchmarks:cool-token" not in out
        assert "have been written to" in out

    def test_aborted_on_no_confirm(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        with patch("builtins.input", return_value="no"):
            api.benchmarks_auth_cli(no_confirm=False, env_file=env_file)
        assert not (tmp_path / ".env").exists()
        out = capsys.readouterr().out
        assert "MODEL_PROXY_URL" in out
        assert "have been written to" not in out

    def test_confirmed_on_yes(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        with patch("builtins.input", return_value="yes"):
            api.benchmarks_auth_cli(no_confirm=False, env_file=env_file)
        assert (tmp_path / ".env").exists()
        out = capsys.readouterr().out
        assert "have been written to" in out

    def test_appends_to_existing_file(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=hello\n")
        api.benchmarks_auth_cli(no_confirm=True, env_file=str(env_file))
        content = env_file.read_text()
        assert content.startswith("EXISTING_VAR=hello\n")
        assert "MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi\n" in content

    def test_custom_env_file(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / "custom.env")
        api.benchmarks_auth_cli(no_confirm=True, env_file=env_file)
        assert (tmp_path / "custom.env").exists()
        out = capsys.readouterr().out
        assert "custom.env" in out


# ============================================================
# Benchmarks Init
# ============================================================


class TestBenchmarksInit:
    """Tests for ``kaggle benchmarks init``."""

    def test_writes_all_vars(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        content = (tmp_path / ".env").read_text()
        assert "MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi\n" in content
        assert "MODEL_PROXY_API_KEY=kaggle-benchmarks:cool-token\n" in content
        assert "MODEL_PROXY_EXPIRY_TIME=2026-04-17T12:00:00Z\n" in content
        assert "LLM_DEFAULT=google/gemini-3-flash-preview\n" in content
        assert "LLM_DEFAULT_EVAL=google/gemini-3-flash-preview\n" in content
        assert "LLMS_AVAILABLE=google/gemini-3-flash-preview,google/gemini-3.1-flash-lite-preview\n" in content
        out = capsys.readouterr().out
        assert "MODEL_PROXY_API_KEY=****************oken" in out
        assert "LLM_DEFAULT=google/gemini-3-flash-preview" in out
        assert "have been written to" in out

    def test_writes_example_file(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        content = (tmp_path / "example_task.py").read_text()
        assert "import kaggle_benchmarks as kbench" in content
        assert "kaggle_benchmarks_reference.md" in content
        out = capsys.readouterr().out
        assert "Example benchmark task file has been written to" in out

    def test_writes_reference_file(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        ref_file = tmp_path / "kaggle_benchmarks_reference.md"
        assert ref_file.exists()
        content = ref_file.read_text()
        assert "kaggle-benchmarks Task Syntax Reference" in content
        out = capsys.readouterr().out
        assert "Syntax reference has been written to" in out
        assert "@kaggle_benchmarks_reference.md" in out

    def test_skips_reference_file_if_exists(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        ref_file = tmp_path / "kaggle_benchmarks_reference.md"
        ref_file.write_text("existing content\n")
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=str(example_file))
        assert ref_file.read_text() == "existing content\n"
        out = capsys.readouterr().out
        assert "Reference file already exists" in out

    def test_skips_example_file_if_exists(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        example_file = tmp_path / "example_task.py"
        example_file.write_text("existing content\n")
        env_file = str(tmp_path / ".env")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=str(example_file))
        assert example_file.read_text() == "existing content\n"
        out = capsys.readouterr().out
        assert "already exists" in out

    def test_custom_example_file(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "my_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=env_file, example_file=example_file)
        content = (tmp_path / "my_task.py").read_text()
        assert "import kaggle_benchmarks as kbench" in content

    def test_aborted_on_no_confirm(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = str(tmp_path / ".env")
        example_file = str(tmp_path / "example_task.py")
        with patch("builtins.input", return_value="no"):
            api.benchmarks_init_cli(no_confirm=False, env_file=env_file, example_file=example_file)
        assert not (tmp_path / ".env").exists()

    def test_appends_to_existing_file(self, api, capsys, tmp_path):
        api._mock_client.models.model_proxy_api_client.create_default_model_proxy_token.return_value = (
            _make_token_response()
        )
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=hello\n")
        example_file = str(tmp_path / "example_task.py")
        api.benchmarks_init_cli(no_confirm=True, env_file=str(env_file), example_file=example_file)
        content = env_file.read_text()
        assert content.startswith("EXISTING_VAR=hello\n")
        assert "LLM_DEFAULT=google/gemini-3-flash-preview\n" in content
