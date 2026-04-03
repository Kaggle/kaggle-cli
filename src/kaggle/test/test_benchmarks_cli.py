import unittest
from unittest.mock import patch, MagicMock
import argparse
import io
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi


class TestBenchmarksCli(unittest.TestCase):

    def setUp(self):
        self.api = KaggleApi()
        # Mock authenticate to avoid real network/creds check during unit tests
        self.api.authenticate = MagicMock()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_push_success(self, mock_stdout):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=True) as f:
            f.write("def evaluate(): pass")
            f.flush()
            # This should call the future implementation
            self.api.benchmarks_tasks_push_cli("my-task", f.name)
            # Future expectation:
            self.assertIn("Task pushed successfully", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_list_all_success(self, mock_stdout):
        self.api.benchmarks_tasks_list_cli()
        # We expect a table header or title
        self.assertIn("Task Name", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_list_with_regex_filter(self, mock_stdout):
        self.api.benchmarks_tasks_list_cli("math.*")
        # Should filter or show filtered view
        self.assertIn("math.*", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_status_single_task(self, mock_stdout):
        self.api.benchmarks_tasks_status_cli("my-task")
        self.assertIn("Status", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_run_single_model(self, mock_stdout):
        self.api.benchmarks_tasks_run_cli("my-task", ["gemini-pro"])
        # Should print tracking URL or run started
        self.assertIn("Tracking URL", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_run_multiple_models(self, mock_stdout):
        self.api.benchmarks_tasks_run_cli("my-task", ["gemini-pro", "gemma-2b"])
        self.assertIn("gemini-pro", mock_stdout.getvalue())
        self.assertIn("gemma-2b", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_download_to_specific_output(self, mock_stdout):
        self.api.benchmarks_tasks_download_cli("my-task", output="my_output_dir")
        self.assertIn("Downloaded to my_output_dir", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_with_no_confirm_flag(self, mock_stdout):
        self.api.benchmarks_tasks_delete_cli("my-task", True)
        self.assertIn("Deleted my-task", mock_stdout.getvalue())


class TestBenchmarksCliParsing(unittest.TestCase):
    """Test that argparse correctly routes benchmark CLI commands."""

    def setUp(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
        )
        subparsers = self.parser.add_subparsers(
            title="commands", dest="command",
        )
        subparsers.required = True

        from kaggle.cli import parse_benchmarks
        parse_benchmarks(subparsers)

    def _parse(self, arg_string):
        return self.parser.parse_args(arg_string.split())

    # ---- push ----

    def test_push_args(self):
        args = self._parse("benchmarks tasks push my-task -f ./task.py")
        self.assertEqual(args.task, "my-task")
        self.assertEqual(args.file, "./task.py")

    def test_push_alias(self):
        args = self._parse("b t push my-task -f ./task.py")
        self.assertEqual(args.task, "my-task")
        self.assertEqual(args.file, "./task.py")

    def test_push_requires_file(self):
        with self.assertRaises(SystemExit):
            self._parse("benchmarks tasks push my-task")

    # ---- list ----

    def test_list_no_args(self):
        args = self._parse("benchmarks tasks list")
        self.assertIsNone(args.regex)

    def test_list_with_regex(self):
        args = self._parse("benchmarks tasks list --regex ^math")
        self.assertEqual(args.regex, "^math")

    # ---- status ----

    def test_status_task_only(self):
        args = self._parse("benchmarks tasks status my-task")
        self.assertEqual(args.task, "my-task")
        self.assertIsNone(args.model)

    def test_status_with_models(self):
        args = self._parse("benchmarks tasks status my-task -m gemini-3 gpt-5")
        self.assertEqual(args.task, "my-task")
        self.assertEqual(args.model, ["gemini-3", "gpt-5"])

    def test_status_model_requires_value(self):
        """nargs='+' should reject -m with zero model names."""
        with self.assertRaises(SystemExit):
            self._parse("benchmarks tasks status my-task -m")

    # ---- run ----

    def test_run_task_only(self):
        args = self._parse("benchmarks tasks run my-task")
        self.assertEqual(args.task, "my-task")
        self.assertIsNone(args.model)
        self.assertFalse(args.wait)

    def test_run_with_models_and_wait(self):
        args = self._parse("benchmarks tasks run my-task -m gemini-3 --wait")
        self.assertEqual(args.model, ["gemini-3"])
        self.assertTrue(args.wait)

    def test_run_multiple_models(self):
        args = self._parse("benchmarks tasks run my-task -m gemini-3 gpt-5 claude-4")
        self.assertEqual(args.model, ["gemini-3", "gpt-5", "claude-4"])

    def test_run_model_requires_value(self):
        """nargs='+' should reject -m with zero model names."""
        with self.assertRaises(SystemExit):
            self._parse("benchmarks tasks run my-task -m")

    # ---- download ----

    def test_download_task_only(self):
        args = self._parse("benchmarks tasks download my-task")
        self.assertEqual(args.task, "my-task")
        self.assertIsNone(args.model)
        self.assertIsNone(args.output)

    def test_download_with_output(self):
        args = self._parse("benchmarks tasks download my-task -o ./results")
        self.assertEqual(args.output, "./results")

    def test_download_with_model_and_output(self):
        args = self._parse("benchmarks tasks download my-task -m gemini-3 -o ./results")
        self.assertEqual(args.model, ["gemini-3"])
        self.assertEqual(args.output, "./results")

    def test_download_model_requires_value(self):
        """nargs='+' should reject -m with zero model names."""
        with self.assertRaises(SystemExit):
            self._parse("benchmarks tasks download my-task -m")

    # ---- delete ----

    def test_delete_task_only(self):
        args = self._parse("benchmarks tasks delete my-task")
        self.assertEqual(args.task, "my-task")
        self.assertFalse(args.no_confirm)

    def test_delete_with_yes_flag(self):
        args = self._parse("benchmarks tasks delete my-task -y")
        self.assertTrue(args.no_confirm)

    def test_delete_with_long_yes_flag(self):
        args = self._parse("benchmarks tasks delete my-task --yes")
        self.assertTrue(args.no_confirm)

    # ---- aliases ----

    def test_alias_b_t(self):
        args = self._parse("b t run my-task -m gemini-3")
        self.assertEqual(args.task, "my-task")
        self.assertEqual(args.model, ["gemini-3"])


if __name__ == "__main__":
    unittest.main()
