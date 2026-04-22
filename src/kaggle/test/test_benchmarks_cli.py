"""Tests for ``kaggle benchmarks tasks`` CLI commands.

Organized by user journey:
  TestPush          – ``kaggle benchmarks tasks push <task> -f <file>``
  TestRun           – ``kaggle benchmarks tasks run <task> [-m ...] [--wait]``
  TestCliArgParsing – argparse wiring verification

Uses pure pytest (capsys, tmp_path) instead of unittest.TestCase so that
stdout capture and temp-file creation are handled by built-in fixtures
rather than manual patching.
"""

import argparse
from unittest.mock import patch, MagicMock

import pytest
from requests.exceptions import HTTPError

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_proxy_api_service import ApiCreateDefaultModelProxyTokenResponse
from kagglesdk.benchmarks.types.benchmark_enums import (
    BenchmarkTaskVersionCreationState,
    BenchmarkTaskRunState,
)

# Short aliases for the verbose enum members used throughout the tests.
QUEUED = BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_QUEUED
RUNNING = BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_RUNNING
COMPLETED = BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED

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


def _make_task(slug="my-task", state=COMPLETED):
    t = MagicMock()
    t.slug.task_slug = slug
    t.creation_state = state
    t.create_time = "2026-04-06"
    return t


def _make_run_result(scheduled=True, skipped_reason=None):
    r = MagicMock()
    r.run_scheduled = scheduled
    r.benchmark_task_version_id = 1
    r.benchmark_model_version_id = 10
    r.run_skipped_reason = skipped_reason
    return r


def _make_run(model="gemini-pro", state=RUN_COMPLETED):
    r = MagicMock()
    r.model_version_slug = model
    r.state = state
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


# ============================================================
# Push Journey
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
        ids=["missing_file", "wrong_extension", "no_decorators", "wrong_name", "syntax_error"],
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
        "content, task_name",
        [
            ('@task(name="my-task")\ndef evaluate(): pass\n', "my-task"),
            ("@task\ndef my_task(llm): pass\n", "My Task"),
            ("@task\nasync def my_task(llm): pass\n", "My Task"),
        ],
        ids=["explicit_name", "title_cased", "async_function"],
    )
    def test_push_creates_task(self, api, tmp_path, capsys, content, task_name):
        """Push converts .py → ipynb via jupytext and creates the task."""
        filepath = _write_task_file(tmp_path, content)
        _setup_create_response(api, task_name)

        jt = _push(api, task_name, filepath)

        # Verify jupytext conversion happened
        jt.reads.assert_called_once()
        jt.writes.assert_called_once()
        request = api._mock_benchmarks.create_benchmark_task.call_args[0][0]
        assert request.text == '{"cells": []}'

        assert f"Task '{task_name}' pushed." in capsys.readouterr().out

    @pytest.mark.parametrize("status_code", [403, 404], ids=["forbidden", "not_found"])
    def test_push_creates_new_task_without_prompting(self, api, tmp_path, capsys, status_code):
        """A 403/404 means a new task — push proceeds without confirmation."""
        filepath = _write_task_file(tmp_path)
        api._mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=status_code))
        _setup_create_response(api)
        _push(api, "my-task", filepath)
        assert "Task 'my-task' pushed." in capsys.readouterr().out

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


# ============================================================
# Run Journey
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
        for m in models:
            assert f"{m}: Scheduled" in output

    def test_run_reports_skipped_with_reason(self, api, capsys):
        _setup_completed_task(api)
        _setup_batch_schedule(api, [_make_run_result(scheduled=False, skipped_reason="Already running")])
        api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        output = capsys.readouterr().out
        assert "gemini-pro: Skipped" in output
        assert "Already running" in output

    # -- Interactive model selection --

    def test_run_prompts_model_selection(self, api):
        """No model specified → user picks from a numbered list."""
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
        """No models available on server → ValueError."""
        _setup_completed_task(api)
        _setup_available_models(api, [])
        with pytest.raises(ValueError, match="No benchmark models available"):
            api.benchmarks_tasks_run_cli("my-task")

    def test_run_rejects_invalid_model_selection(self, api):
        """Bad input during interactive model selection → ValueError."""
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


# ============================================================
# CLI Arg Parsing
# ============================================================


class TestCliArgParsing:
    """Tests that argparse wiring for ``kaggle benchmarks tasks`` is correct.

    Verifies argument names, aliases (b/t), required flags, and nargs
    constraints are properly configured in cli.py.
    """

    def setup_method(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
        )
        subparsers = self.parser.add_subparsers(title="commands", dest="command")
        subparsers.required = True
        from kaggle.cli import parse_benchmarks

        parse_benchmarks(subparsers)

    def _parse(self, arg_string):
        return self.parser.parse_args(arg_string.split())

    @pytest.mark.parametrize(
        "cmd, expected",
        [
            ("benchmarks tasks push my-task -f ./task.py", {"task": "my-task", "file": "./task.py"}),
            ("b t push my-task -f ./task.py", {"task": "my-task", "file": "./task.py"}),
            ("benchmarks tasks run my-task", {"task": "my-task", "model": None, "wait": None}),
            ("benchmarks tasks run my-task -m gemini-3 --wait", {"model": ["gemini-3"], "wait": 0}),
            (
                "benchmarks tasks run my-task -m gemini-3 --wait 60",
                {"model": ["gemini-3"], "wait": 60},
            ),
            (
                "benchmarks tasks run my-task -m gemini-3 gpt-5 claude-4",
                {"model": ["gemini-3", "gpt-5", "claude-4"]},
            ),
            ("b t run my-task -m gemini-3", {"task": "my-task", "model": ["gemini-3"]}),
        ],
    )
    def test_parse_success(self, cmd, expected):
        args = self._parse(cmd)
        for key, val in expected.items():
            assert getattr(args, key) == val

    @pytest.mark.parametrize(
        "cmd",
        [
            "benchmarks tasks push my-task",
            "benchmarks tasks run my-task -m",
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
