import os
import json
import unittest
import tempfile
from unittest.mock import MagicMock, patch

from kaggle.api.kaggle_api_extended import KaggleApi


class TestBenchmarks(unittest.TestCase):
    def setUp(self):
        self.api = KaggleApi()
        self.api.get_config_value = MagicMock(return_value="testuser")

    @patch("jupytext.read")
    @patch("jupytext.write")
    def test_benchmarks_pull(self, mock_write, mock_read):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.api.kernels_pull = MagicMock(return_value=tmpdir)
            # Create a dummy ipynb file as if kernels_pull pulled it
            dummy_ipynb = os.path.join(tmpdir, "notebook.ipynb")
            with open(dummy_ipynb, "w") as f:
                f.write("{}")

            result_dir = self.api.benchmarks_pull(
                "testuser/my-benchmark", path=tmpdir, quiet=True
            )
            self.assertEqual(result_dir, tmpdir)

            # 1. notebooks should be renamed to benchmark.ipynb
            benchmark_ipynb = os.path.join(tmpdir, "benchmark.ipynb")
            self.assertTrue(os.path.exists(benchmark_ipynb))
            self.assertFalse(os.path.exists(dummy_ipynb))

            # 2. jupytext read and write should be called
            mock_read.assert_called_once_with(benchmark_ipynb)
            mock_write.assert_called_once()
            self.assertEqual(
                mock_write.call_args[0][1], os.path.join(tmpdir, "benchmark.py")
            )
            self.assertEqual(mock_write.call_args[1]["fmt"], "py:percent")
            self.api.kernels_pull.assert_called_once_with(
                "testuser/my-benchmark", path=tmpdir, metadata=True, quiet=True
            )

    @patch("jupytext.read")
    @patch("jupytext.write")
    def test_benchmarks_publish_and_run(self, mock_write, mock_read):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy python file
            py_file = os.path.join(tmpdir, "benchmark.py")
            with open(py_file, "w") as f:
                f.write("print('hello')")

            self.api.kernels_push = MagicMock(return_value="push_success")

            result = self.api.benchmarks_publish_and_run(
                kernel="testuser/new-benchmark",
                path=tmpdir,
                file_name="benchmark.py",
                quiet=True,
            )

            self.assertEqual(result, "push_success")

            # verify jupytext was used to read .py and write .ipynb
            mock_read.assert_called_once_with(py_file, fmt="py:percent")
            mock_write.assert_called_once()
            self.assertEqual(
                mock_write.call_args[0][1], os.path.join(tmpdir, "benchmark.ipynb")
            )

            # verify metadata was created correctly
            metadata_file = os.path.join(tmpdir, "kernel-metadata.json")
            self.assertTrue(os.path.exists(metadata_file))
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["id"], "testuser/new-benchmark")
            self.assertEqual(metadata["code_file"], "benchmark.ipynb")
            self.assertIn("personal-benchmark", metadata["keywords"])

            self.api.kernels_push.assert_called_once_with(tmpdir)

    @patch("jupytext.read")
    @patch("jupytext.write")
    def test_benchmarks_publish_and_run_existing_metadata(self, mock_write, mock_read):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy python file
            py_file = os.path.join(tmpdir, "benchmark.py")
            with open(py_file, "w") as f:
                f.write("print('hello')")

            # Create existing metadata
            metadata_file = os.path.join(tmpdir, "kernel-metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(
                    {
                        "id": "otheruser/existing-benchmark",
                        "code_file": "old.ipynb",
                        "keywords": ["tag1"],
                    },
                    f,
                )

            self.api.kernels_push = MagicMock(return_value="push_success")

            # Act
            self.api.benchmarks_publish_and_run(path=tmpdir, quiet=True)

            # Assert
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            # The keyword and code_file should be forcibly updated
            self.assertIn("personal-benchmark", metadata["keywords"])
            self.assertIn("tag1", metadata["keywords"])
            self.assertEqual(metadata["code_file"], "benchmark.ipynb")
            # id remains the same
            self.assertEqual(metadata["id"], "otheruser/existing-benchmark")

    @patch("jupytext.read")
    @patch("jupytext.write")
    def test_benchmarks_publish_and_run_explicit_kernel(self, mock_write, mock_read):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "benchmark.py")
            with open(py_file, "w") as f:
                f.write("print('hello')")

            # Create existing metadata
            metadata_file = os.path.join(tmpdir, "kernel-metadata.json")
            with open(metadata_file, "w") as f:
                json.dump({"id": "old/existing", "id_no": 12345}, f)

            self.api.kernels_push = MagicMock(return_value="push_success")

            # Act with explicit kernel override
            self.api.benchmarks_publish_and_run(
                kernel="newuser/new-benchmark", path=tmpdir, quiet=True
            )

            # Assert
            self.api.kernels_push.assert_called_once_with(tmpdir)
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["id"], "newuser/new-benchmark")
            self.assertEqual(metadata["title"], "New Benchmark")
            self.assertNotIn("id_no", metadata)
            self.assertIn("personal-benchmark", metadata["keywords"])

    @patch("time.sleep")
    def test_benchmarks_get_results(self, mock_sleep):
        # mock status to return 'running' once, then 'complete'
        class MockStatus:
            def __init__(self, status):
                self.status = status
                self.failure_message = ""

        self.api.kernels_status = MagicMock(
            side_effect=[MockStatus("running"), MockStatus("complete")]
        )
        self.api.kernels_output = MagicMock(return_value="output_data")

        result = self.api.benchmarks_get_results(
            "testuser/my-bench", path="some_path", poll_interval=10
        )

        self.assertEqual(result, "output_data")
        self.assertEqual(self.api.kernels_status.call_count, 2)
        mock_sleep.assert_called_once_with(10)
        self.api.kernels_output.assert_called_once_with(
            kernel="testuser/my-bench",
            path=os.path.join("some_path", "output"),
            file_pattern=None,
            force=True,
            quiet=False,
        )

    def test_benchmarks_get_results_error(self):
        class MockStatusError:
            def __init__(self):
                self.status = "error"
                self.failure_message = "syntax error"

        self.api.kernels_status = MagicMock(return_value=MockStatusError())
        self.api.kernels_output = MagicMock()

        with self.assertRaisesRegex(ValueError, "error state"):
            self.api.benchmarks_get_results("testuser/my-bench")

    @patch("time.sleep")
    def test_benchmarks_get_results_no_kernel(self, mock_sleep):
        class MockStatus:
            def __init__(self, status):
                self.status = status
                self.failure_message = ""

        self.api.kernels_status = MagicMock(return_value=MockStatus("complete"))
        self.api.kernels_output = MagicMock(return_value="output_data")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing metadata
            metadata_file = os.path.join(tmpdir, "kernel-metadata.json")
            with open(metadata_file, "w") as f:
                json.dump({"id": "implicit/my-bench"}, f)

            result = self.api.benchmarks_get_results(kernel=None, path=tmpdir)

            self.assertEqual(result, "output_data")
            self.api.kernels_output.assert_called_once_with(
                kernel="implicit/my-bench",
                path=os.path.join(tmpdir, "output"),
                file_pattern=None,
                force=True,
                quiet=False,
            )


if __name__ == "__main__":
    unittest.main()
