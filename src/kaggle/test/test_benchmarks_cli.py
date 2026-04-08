import os
from unittest.mock import patch as _patch

# Must be set before importing kaggle, which calls api.authenticate() at
# module level.  Fake legacy credentials keep authenticate() off the network;
# removing KAGGLE_API_TOKEN prevents _introspect_token() from being called.
# We also patch get_access_token_from_env so the ~/.kaggle/access_token file
# doesn't trigger token introspection.
os.environ.pop("KAGGLE_API_TOKEN", None)
os.environ["KAGGLE_USERNAME"] = "testuser"
os.environ["KAGGLE_KEY"] = "testkey"

with _patch("kagglesdk.get_access_token_from_env", return_value=(None, None)):
    import kaggle  # noqa: F401 — triggers authenticate()

import unittest
from unittest.mock import patch, MagicMock
import argparse
import io
import tempfile
import pytest
from requests.exceptions import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.benchmarks.types.benchmark_enums import BenchmarkTaskVersionCreationState, BenchmarkTaskRunState


class TestBenchmarksCli(unittest.TestCase):
    """Tests for `kaggle benchmarks tasks <command>` CLI methods.

    Each test exercises one API method (e.g. benchmarks_tasks_push_cli) with
    mocked SDK calls, verifying the printed output and request arguments match
    the expected user experience.
    """

    TASK_FILE_CONTENT = '@task(name="my-task")\ndef evaluate(): pass\n'

    def setUp(self):
        self.api = KaggleApi()
        # Mock authenticate to avoid real network/creds check during unit tests
        self.api.authenticate = MagicMock()

        # Mock build_kaggle_client to avoid real network calls
        self.mock_client = MagicMock()
        self.api.build_kaggle_client = MagicMock()
        self.api.build_kaggle_client.return_value.__enter__.return_value = self.mock_client
        self.mock_benchmarks = self.mock_client.benchmarks.benchmark_tasks_api_client

    # -- Helpers --

    def _make_task_file(self, content=None):
        """Create a temp .py file with task content. Caller must call .close()."""
        f = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=True)
        f.write(content or self.TASK_FILE_CONTENT)
        f.flush()
        return f

    def _mock_jupytext(self):
        """Return a mock jupytext module and a context manager that patches it in."""
        jt = MagicMock()
        jt.reads.return_value = "mock_notebook"
        jt.writes.return_value = '{"cells": []}'
        return jt, patch.dict("sys.modules", {"jupytext": jt})

    def _setup_create_response(self, task_slug="my-task"):
        mock_resp = MagicMock()
        mock_resp.slug.task_slug = task_slug
        mock_resp.url = f"https://kaggle.com/benchmarks/{task_slug}"
        self.mock_benchmarks.create_benchmark_task.return_value = mock_resp
        return mock_resp

    def _make_mock_task(self, slug="my-task", state="COMPLETED", create_time="2026-04-06"):
        t = MagicMock()
        t.slug.task_slug = slug
        t.creation_state = state
        t.create_time = create_time
        return t

    def _setup_list_response(self, tasks):
        resp = MagicMock()
        resp.tasks = tasks
        self.mock_benchmarks.list_benchmark_tasks.return_value = resp

    def _make_mock_run(
        self,
        model="gemini-pro",
        state=BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_COMPLETED,
        run_id=1,
        start_time=None,
        end_time=None,
        error_message=None,
    ):
        r = MagicMock()
        r.model_slug = model
        r.state = state
        r.id = run_id
        r.start_time = start_time
        r.end_time = end_time
        r.error_message = error_message
        return r

    def _setup_runs_response(self, runs):
        resp = MagicMock()
        resp.runs = runs
        self.mock_benchmarks.list_benchmark_task_runs.return_value = resp

    def _make_run_result(self, scheduled=True, skipped_reason=None):
        r = MagicMock()
        r.run_scheduled = scheduled
        r.benchmark_task_version_id = 1
        r.benchmark_model_version_id = 10
        r.run_skipped_reason = skipped_reason
        return r

    # ---- kaggle benchmarks tasks push <task> -f <file> ----

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_push_success(self, mock_stdout):
        """Happy path: push a valid .py file creates the task on the server."""
        with self._make_task_file() as f:
            self._setup_create_response()
            jt, jt_ctx = self._mock_jupytext()
            with jt_ctx:
                self.api.benchmarks_tasks_push_cli("my-task", f.name)
            self.assertIn("Task 'my-task' pushed.", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_push_failure_pending(self, mock_stdout):
        """Push rejected when the task version is still being created (QUEUED)."""
        with self._make_task_file() as f:
            task = self._make_mock_task(
                state=BenchmarkTaskVersionCreationState.BENCHMARK_TASK_VERSION_CREATION_STATE_QUEUED
            )
            self.mock_benchmarks.get_benchmark_task.return_value = task
            _, jt_ctx = self._mock_jupytext()
            with jt_ctx, self.assertRaises(ValueError) as cm:
                self.api.benchmarks_tasks_push_cli("my-task", f.name)
            self.assertIn("is currently being created", str(cm.exception))

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_push_success_404(self, mock_stdout):
        """A 404 on get means new task — should still create successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        self.mock_benchmarks.get_benchmark_task.side_effect = HTTPError(response=mock_response)
        self._setup_create_response()
        with self._make_task_file() as f:
            _, jt_ctx = self._mock_jupytext()
            with jt_ctx:
                self.api.benchmarks_tasks_push_cli("my-task", f.name)
        self.assertIn("Task 'my-task' pushed.", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_push_converts_to_ipynb(self, mock_stdout):
        """Push converts the .py file to ipynb via jupytext before uploading."""
        with self._make_task_file() as f:
            self._setup_create_response()
            jt, jt_ctx = self._mock_jupytext()
            with jt_ctx:
                self.api.benchmarks_tasks_push_cli("my-task", f.name)
            jt.reads.assert_called_once()
            jt.writes.assert_called_once()
            request = self.mock_benchmarks.create_benchmark_task.call_args[0][0]
            self.assertEqual(request.text, '{"cells": []}')

    # ---- kaggle benchmarks tasks list [--regex] [--status] ----

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_all_success(self, mock_stdout):
        """'kaggle b t list' with no filters prints task table."""
        self._setup_list_response([self._make_mock_task()])
        self.api.benchmarks_tasks_list_cli()
        self.assertIn("Task", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_with_regex_filter(self, mock_stdout):
        """'kaggle b t list --regex math.*' passes regex to the server."""
        self._setup_list_response([self._make_mock_task(slug="math-task")])
        self.api.benchmarks_tasks_list_cli("math.*")
        request = self.mock_benchmarks.list_benchmark_tasks.call_args[0][0]
        self.assertEqual(request.regex_filter, "math.*")
        self.assertIn("math-task", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_with_status_filter(self, mock_stdout):
        """'kaggle b t list --status completed' passes status to the server."""
        self._setup_list_response([self._make_mock_task()])
        self.api.benchmarks_tasks_list_cli(status="completed")
        request = self.mock_benchmarks.list_benchmark_tasks.call_args[0][0]
        self.assertEqual(request.status_filter, "completed")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_shows_creation_state(self, mock_stdout):
        """List output includes a Status column showing the creation state."""
        self._setup_list_response([self._make_mock_task()])
        self.api.benchmarks_tasks_list_cli()
        output = mock_stdout.getvalue()
        self.assertIn("Status", output)
        self.assertIn("COMPLETED", output)

    # ---- kaggle benchmarks tasks status <task> [-m ...] ----

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_status_single_task(self, mock_stdout):
        """'kaggle b t status my-task' shows task info and no-runs message."""
        self.mock_benchmarks.get_benchmark_task.return_value = self._make_mock_task()
        self._setup_runs_response([])
        self.api.benchmarks_tasks_status_cli("my-task")
        self.assertIn("Status", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_status_with_model_filter(self, mock_stdout):
        """'kaggle b t status my-task -m gemini-3' filters runs by model."""
        self.mock_benchmarks.get_benchmark_task.return_value = self._make_mock_task()
        self._setup_runs_response([])
        self.api.benchmarks_tasks_status_cli("my-task", model="gemini-3")
        request = self.mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        self.assertEqual(request.model_slugs, ["gemini-3"])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_status_with_multiple_models_filter(self, mock_stdout):
        """'kaggle b t status my-task -m gemini-3 gpt-5' accepts multiple models."""
        self.mock_benchmarks.get_benchmark_task.return_value = self._make_mock_task()
        self._setup_runs_response([])
        self.api.benchmarks_tasks_status_cli("my-task", model=["gemini-3", "gpt-5"])
        request = self.mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        self.assertEqual(request.model_slugs, ["gemini-3", "gpt-5"])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_status_shows_end_time_and_error(self, mock_stdout):
        """Status output shows end time and error message for errored runs."""
        self.mock_benchmarks.get_benchmark_task.return_value = self._make_mock_task()
        run = self._make_mock_run(
            model="gemini-3",
            state=BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_ERRORED,
            run_id=42,
            start_time="2026-04-06T10:00:00Z",
            end_time="2026-04-06T10:05:00Z",
            error_message="Out of memory",
        )
        self._setup_runs_response([run])
        self.api.benchmarks_tasks_status_cli("my-task")
        output = mock_stdout.getvalue()
        self.assertIn("Ended", output)
        self.assertIn("Out of memory", output)

    # ---- kaggle benchmarks tasks run <task> [-m ...] [--wait] ----

    def _setup_batch_schedule(self, results):
        resp = MagicMock()
        resp.results = results
        self.mock_benchmarks.batch_schedule_benchmark_task_runs.return_value = resp

    def _setup_available_models(self, slugs):
        models = []
        for s in slugs:
            m = MagicMock()
            m.slug = s
            m.display_name = s.title()
            models.append(m)
        resp = MagicMock()
        resp.benchmark_models = models
        self.mock_client.benchmarks.benchmarks_api_client.list_benchmark_models.return_value = resp

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_single_model(self, mock_stdout):
        """'kaggle b t run my-task -m gemini-pro' schedules one run."""
        self._setup_batch_schedule([self._make_run_result()])
        self.api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        self.assertIn("Submitted run(s) for task 'my-task'", mock_stdout.getvalue())
        self.assertIn("gemini-pro: Scheduled", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_multiple_models(self, mock_stdout):
        """'kaggle b t run my-task -m gemini-pro gemma-2b' schedules two runs."""
        self._setup_batch_schedule([self._make_run_result(), self._make_run_result()])
        self.api.benchmarks_tasks_run_cli("my-task", ["gemini-pro", "gemma-2b"])
        output = mock_stdout.getvalue()
        self.assertIn("gemini-pro: Scheduled", output)
        self.assertIn("gemma-2b: Scheduled", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_skipped_result(self, mock_stdout):
        """When the server skips a run (e.g. already running), show reason."""
        self._setup_batch_schedule([self._make_run_result(scheduled=False, skipped_reason="Already running")])
        self.api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        output = mock_stdout.getvalue()
        self.assertIn("gemini-pro: Skipped", output)
        self.assertIn("Already running", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_no_model_prompts_selection(self, mock_stdout):
        """When no model is specified, user is prompted to select from available models."""
        self._setup_available_models(["gemini-pro", "gemma-2b"])
        self._setup_batch_schedule([self._make_run_result()])
        with patch("builtins.input", return_value="1"):
            self.api.benchmarks_tasks_run_cli("my-task")
        request = self.mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        self.assertEqual(request.model_slugs, ["gemini-pro"])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_no_model_select_all(self, mock_stdout):
        """When no model is specified and user selects 'all'."""
        self._setup_available_models(["gemini-pro", "gemma-2b"])
        self._setup_batch_schedule([])
        with patch("builtins.input", return_value="all"):
            self.api.benchmarks_tasks_run_cli("my-task")
        request = self.mock_benchmarks.batch_schedule_benchmark_task_runs.call_args[0][0]
        self.assertEqual(request.model_slugs, ["gemini-pro", "gemma-2b"])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_with_wait(self, mock_stdout):
        """Test --wait polls until runs complete."""
        self._setup_batch_schedule([self._make_run_result()])
        running = self._make_mock_run(state=BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_RUNNING)
        done = self._make_mock_run(state=BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_COMPLETED)
        resp1 = MagicMock(runs=[running])
        resp2 = MagicMock(runs=[done])
        self.mock_benchmarks.list_benchmark_task_runs.side_effect = [resp1, resp2]
        with patch("time.sleep"):
            self.api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=0)
        output = mock_stdout.getvalue()
        self.assertIn("Waiting for run(s) to complete", output)
        self.assertIn("All runs completed", output)
        self.assertIn("gemini-pro: COMPLETED", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_with_timeout(self, mock_stdout):
        """Test --wait with timeout stops waiting."""
        self._setup_batch_schedule([self._make_run_result()])
        running = self._make_mock_run(state=BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_RUNNING)
        self._setup_runs_response([running])
        with patch("time.sleep"), patch("time.time", side_effect=[1000, 1060]):
            self.api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"], wait=30)
        output = mock_stdout.getvalue()
        self.assertIn("Waiting for run(s) to complete", output)
        self.assertIn("Timed out waiting for runs after 30 seconds", output)

    # ---- kaggle benchmarks tasks download <task> [-m ...] [-o <dir>] ----

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_download_to_specific_output(self, mock_stdout):
        """'kaggle b t download my-task -o my_output_dir' saves to given dir."""
        self._setup_runs_response([self._make_mock_run()])
        self.mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        self.api.download_file = MagicMock()
        self.api.benchmarks_tasks_download_cli("my-task", output="my_output_dir")
        self.assertIn("Downloaded output for", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_download_with_model_filter(self, mock_stdout):
        """'kaggle b t download my-task -m gemini-pro' filters by model."""
        self._setup_runs_response([self._make_mock_run()])
        self.mock_benchmarks.download_benchmark_task_run_output.return_value = MagicMock()
        self.api.download_file = MagicMock()
        self.api.benchmarks_tasks_download_cli("my-task", model="gemini-pro")
        request = self.mock_benchmarks.list_benchmark_task_runs.call_args[0][0]
        self.assertEqual(request.model_slugs, ["gemini-pro"])

    # ---- kaggle benchmarks tasks delete <task> [-y] ----

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_delete_with_no_confirm_flag(self, mock_stdout):
        """'kaggle b t delete my-task -y' prints not-supported message."""
        self.api.benchmarks_tasks_delete_cli("my-task", True)
        self.assertIn("Delete is not supported", mock_stdout.getvalue())

    # ---- push input validation (before any server call) ----

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_push_validation_failure(self, mock_stdout):
        """Push errors when the requested task name doesn't match any @task in the file."""
        with self._make_task_file('@task(name="real-task")\ndef my_task(llm): pass\n') as f:
            with self.assertRaises(ValueError) as cm:
                self.api.benchmarks_tasks_push_cli("wrong-task", f.name)
            self.assertIn("Task 'wrong-task' not found", str(cm.exception))

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_push_validation_no_tasks(self, mock_stdout):
        """Push errors when the file has no @task decorators at all."""
        with self._make_task_file("def regular_function(): pass\n") as f:
            with self.assertRaises(ValueError) as cm:
                self.api.benchmarks_tasks_push_cli("any-task", f.name)
            self.assertIn("No @task decorators found", str(cm.exception))

    # ---- edge-case coverage ----

    def test_push_file_not_found(self):
        """Push errors immediately when the source file doesn't exist."""
        with self.assertRaises(ValueError) as cm:
            self.api.benchmarks_tasks_push_cli("my-task", "/nonexistent/task.py")
        self.assertIn("does not exist", str(cm.exception))

    def test_push_not_py_file(self):
        """Push errors when the file is not a .py file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w") as f:
            f.write("hello")
            f.flush()
            with self.assertRaises(ValueError) as cm:
                self.api.benchmarks_tasks_push_cli("my-task", f.name)
            self.assertIn("must be a .py file", str(cm.exception))

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_empty_result(self, mock_stdout):
        """'kaggle b t list' with no tasks still prints the table header."""
        self._setup_list_response([])
        self.api.benchmarks_tasks_list_cli()
        output = mock_stdout.getvalue()
        self.assertIn("Task", output)  # Header still printed

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_download_skips_non_completed_runs(self, mock_stdout):
        """Runs that are still RUNNING should not be downloaded."""
        running = self._make_mock_run(state=BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_RUNNING)
        self._setup_runs_response([running])
        self.api.download_file = MagicMock()
        self.api.benchmarks_tasks_download_cli("my-task")
        self.api.download_file.assert_not_called()


class TestBenchmarksCliParsing:
    """Tests that argparse wiring for `kaggle benchmarks tasks` is correct.

    These verify that argument names, aliases (b/t), required flags,
    and nargs constraints are properly configured in cli.py.
    """

    def setup_method(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
        )
        subparsers = self.parser.add_subparsers(
            title="commands",
            dest="command",
        )
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
            ("benchmarks tasks list", {"regex": None, "status": None}),
            ("benchmarks tasks list --regex ^math", {"regex": "^math"}),
            ("benchmarks tasks list --status completed", {"status": "completed"}),
            ("benchmarks tasks list --regex ^math --status errored", {"regex": "^math", "status": "errored"}),
            ("benchmarks tasks status my-task", {"task": "my-task", "model": None}),
            ("benchmarks tasks status my-task -m gemini-3 gpt-5", {"task": "my-task", "model": ["gemini-3", "gpt-5"]}),
            ("benchmarks tasks run my-task", {"task": "my-task", "model": None, "wait": None}),
            ("benchmarks tasks run my-task -m gemini-3 --wait", {"model": ["gemini-3"], "wait": 0}),
            ("benchmarks tasks run my-task -m gemini-3 --wait 60", {"model": ["gemini-3"], "wait": 60}),
            ("benchmarks tasks run my-task -m gemini-3 gpt-5 claude-4", {"model": ["gemini-3", "gpt-5", "claude-4"]}),
            ("benchmarks tasks download my-task", {"task": "my-task", "model": None, "output": None}),
            ("benchmarks tasks download my-task -o ./results", {"output": "./results"}),
            ("benchmarks tasks download my-task -m gemini-3 -o ./results", {"model": ["gemini-3"], "output": "./results"}),
            ("benchmarks tasks delete my-task", {"task": "my-task", "no_confirm": False}),
            ("benchmarks tasks delete my-task -y", {"no_confirm": True}),
            ("benchmarks tasks delete my-task --yes", {"no_confirm": True}),
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
            "benchmarks tasks status my-task -m",
            "benchmarks tasks run my-task -m",
            "benchmarks tasks download my-task -m",
        ],
    )
    def test_parse_error(self, cmd):
        with pytest.raises(SystemExit):
            self._parse(cmd)



class TestTaskNameExtraction(unittest.TestCase):
    """Tests for _get_task_names_from_file(), which parses @task decorators
    from Python source code using AST to validate push inputs.
    """

    def setUp(self):
        self.api = KaggleApi()

    def test_extract_simple_task(self):
        """@kbench.task with no name= arg uses the function name as title case."""
        code = """
import kaggle_benchmarks as kbench
@kbench.task
def my_task(llm):
    pass
"""
        task_names = self.api._get_task_names_from_file(code)
        self.assertEqual(task_names, ["My Task"])

    def test_extract_task_with_name(self):
        """@kbench.task(name='custom_name') uses the explicit name."""
        code = """
import kaggle_benchmarks as kbench
@kbench.task(name="custom_name")
def my_task(llm):
    pass
"""
        task_names = self.api._get_task_names_from_file(code)
        self.assertEqual(task_names, ["custom_name"])

    def test_extract_multiple_tasks(self):
        """Multiple @task decorators in one file are all extracted."""
        code = """
@task
def task1(llm): pass

@task(name="task2_custom")
def task2(llm): pass
"""
        task_names = self.api._get_task_names_from_file(code)
        self.assertEqual(set(task_names), {"Task1", "task2_custom"})

    def test_extract_no_tasks(self):
        """File with no @task decorators returns empty list."""
        code = """
def regular_function(): pass
"""
        task_names = self.api._get_task_names_from_file(code)
        self.assertEqual(task_names, [])

    def test_extract_syntax_error(self):
        """Files with syntax errors return empty list instead of crashing."""
        code = """
def broken_function(
"""
        task_names = self.api._get_task_names_from_file(code)
        self.assertEqual(task_names, [])

    def test_extract_async_task(self):
        """Async function definitions with @task are also extracted."""
        code = """
@task
async def my_async_task(llm):
    pass
"""
        task_names = self.api._get_task_names_from_file(code)
        self.assertEqual(task_names, ["My Async Task"])


if __name__ == "__main__":
    unittest.main()
