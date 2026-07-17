import argparse
from unittest.mock import MagicMock, patch
import pytest
import os
from contextlib import redirect_stdout
import io

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.benchmarks.types.benchmarks_api_service import ApiBenchmarkLeaderboard


@pytest.fixture
def api():
    a = KaggleApi()
    a.authenticate = MagicMock()  # type: ignore[method-assign]
    mock_client = MagicMock()
    a.build_kaggle_client = MagicMock()  # type: ignore[method-assign]
    a.build_kaggle_client.return_value.__enter__.return_value = mock_client
    a._mock_client = mock_client  # type: ignore[attr-defined]
    a._mock_benchmarks = mock_client.benchmarks.benchmarks_api_client  # type: ignore[attr-defined]
    return a


def _make_leaderboard_response(rows_data):
    response = MagicMock(spec=ApiBenchmarkLeaderboard)
    rows = []
    for r_data in rows_data:
        row = MagicMock()
        row.model_version_name = r_data["model_name"]
        row.model_version_slug = r_data["model_slug"]

        task_results = []
        for t_data in r_data["results"]:
            tr = MagicMock()
            tr.benchmark_task_name = t_data["task_name"]
            tr.benchmark_task_slug = t_data["task_slug"]
            tr.task_version = 1

            res = MagicMock()
            if "score" in t_data:
                res.numeric_result.value = t_data["score"]
                res.boolean_result = None
            elif "pass" in t_data:
                res.numeric_result = None
                res.boolean_result = t_data["pass"]
            else:
                res.numeric_result = None
                res.boolean_result = None
            tr.result = res
            task_results.append(tr)

        row.task_results = task_results
        rows.append(row)
    response.rows = rows
    return response


class TestBenchmarksLeaderboard:

    def test_benchmark_leaderboard_view_success(self, api):
        # Arrange
        mock_response = _make_leaderboard_response([])
        api._mock_benchmarks.get_benchmark_leaderboard.return_value = mock_response

        # Act
        response = api.benchmark_leaderboard_view("owner/benchmark-slug", version=2)

        # Assert
        api._mock_benchmarks.get_benchmark_leaderboard.assert_called_once()
        request = api._mock_benchmarks.get_benchmark_leaderboard.call_args[0][0]
        assert request.owner_slug == "owner"
        assert request.benchmark_slug == "benchmark-slug"
        assert request.version_number == 2
        assert response == mock_response

    def test_benchmark_leaderboard_cli_show_table(self, api, capsys):
        # Arrange
        rows_data = [
            {
                "model_name": "Model A",
                "model_slug": "model-a",
                "results": [
                    {"task_name": "Task 1", "task_slug": "task-1", "score": 0.85},
                    {"task_name": "Task 2", "task_slug": "task-2", "score": 0.92},
                ],
            },
            {
                "model_name": "Model B",
                "model_slug": "model-b",
                "results": [
                    {"task_name": "Task 1", "task_slug": "task-1", "score": 0.70},
                ],
            },
        ]
        mock_response = _make_leaderboard_response(rows_data)
        api._mock_benchmarks.get_benchmark_leaderboard.return_value = mock_response

        # Act
        api.benchmark_leaderboard_cli("owner/benchmark-slug", view=True)

        # Assert
        captured = capsys.readouterr()
        # Verify headers
        assert "Model" in captured.out
        assert "Task 1" in captured.out
        assert "Task 2" in captured.out
        # Verify values
        assert "Model A" in captured.out
        assert "0.85" in captured.out
        assert "0.92" in captured.out
        assert "Model B" in captured.out
        assert "0.7" in captured.out

        lines = captured.out.split("\n")
        model_b_line = [l for l in lines if "Model B" in l][0]
        assert "N/A" in model_b_line

    def test_benchmark_leaderboard_cli_show_csv(self, api, capsys):
        # Arrange
        rows_data = [
            {
                "model_name": "Model A",
                "model_slug": "model-a",
                "results": [
                    {"task_name": "Task 1", "task_slug": "task-1", "score": 0.85},
                ],
            }
        ]
        mock_response = _make_leaderboard_response(rows_data)
        api._mock_benchmarks.get_benchmark_leaderboard.return_value = mock_response

        # Act
        api.benchmark_leaderboard_cli("owner/benchmark-slug", view=True, csv_display=True)

        # Assert
        captured = capsys.readouterr()
        lines = [l.strip() for l in captured.out.strip().split("\n")]
        assert lines[0] == "Model,Task 1"
        assert lines[1] == "Model A,0.85"

    def test_benchmark_leaderboard_cli_download(self, api, tmp_path):
        # Arrange
        rows_data = [
            {
                "model_name": "Model A",
                "model_slug": "model-a",
                "results": [
                    {"task_name": "Task 1", "task_slug": "task-1", "score": 0.85},
                ],
            }
        ]
        mock_response = _make_leaderboard_response(rows_data)
        api._mock_benchmarks.get_benchmark_leaderboard.return_value = mock_response

        # Act
        api.benchmark_leaderboard_cli("owner/benchmark-slug", download=True, path=str(tmp_path))

        # Assert
        expected_file = tmp_path / "benchmark-slug_leaderboard.csv"
        assert expected_file.exists()
        content = expected_file.read_text()
        lines = [l.strip() for l in content.strip().split("\n")]
        assert lines[0] == "Model,Task 1"
        assert lines[1] == "Model A,0.85"

    def test_benchmark_leaderboard_cli_no_show_or_download_raises_value_error(self, api):
        with pytest.raises(ValueError, match="Either --show or --download must be specified"):
            api.benchmark_leaderboard_cli("owner/benchmark-slug")

    def test_benchmark_leaderboard_cli_no_results_quiet(self, api, capsys):
        # Arrange
        mock_response = _make_leaderboard_response([])
        api._mock_benchmarks.get_benchmark_leaderboard.return_value = mock_response

        # Act
        api.benchmark_leaderboard_cli("owner/benchmark-slug", view=True, quiet=True)

        # Assert
        captured = capsys.readouterr()
        assert "No results found" not in captured.out

    def test_benchmark_leaderboard_cli_download_quiet(self, api, tmp_path, capsys):
        # Arrange
        rows_data = [
            {
                "model_name": "Model A",
                "model_slug": "model-a",
                "results": [
                    {"task_name": "Task 1", "task_slug": "task-1", "score": 0.85},
                ],
            }
        ]
        mock_response = _make_leaderboard_response(rows_data)
        api._mock_benchmarks.get_benchmark_leaderboard.return_value = mock_response

        # Act
        api.benchmark_leaderboard_cli("owner/benchmark-slug", download=True, path=str(tmp_path), quiet=True)

        # Assert
        captured = capsys.readouterr()
        assert "Leaderboard downloaded to" not in captured.out

        expected_file = tmp_path / "benchmark-slug_leaderboard.csv"
        assert expected_file.exists()


class TestCliArgParsing:

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
            (
                "benchmarks leaderboard owner/benchmark-slug -s",
                {
                    "command": "leaderboard",
                    "benchmark": "owner/benchmark-slug",
                    "view": True,
                    "download": False,
                    "version": None,
                    "path": None,
                },
            ),
            (
                "benchmarks leaderboard owner/benchmark-slug --download --path /tmp/down",
                {
                    "command": "leaderboard",
                    "benchmark": "owner/benchmark-slug",
                    "view": False,
                    "download": True,
                    "version": None,
                    "path": "/tmp/down",
                },
            ),
            (
                "benchmarks leaderboard owner/benchmark-slug --version 3 -s -d",
                {
                    "command": "leaderboard",
                    "benchmark": "owner/benchmark-slug",
                    "view": True,
                    "download": True,
                    "version": 3,
                },
            ),
            (
                "benchmarks leaderboard owner/benchmark-slug -s -q",
                {
                    "command": "leaderboard",
                    "benchmark": "owner/benchmark-slug",
                    "view": True,
                    "download": False,
                    "version": None,
                    "path": None,
                    "quiet": True,
                },
            ),
            (
                "benchmarks leaderboard owner/benchmark-slug -s --quiet",
                {
                    "command": "leaderboard",
                    "benchmark": "owner/benchmark-slug",
                    "view": True,
                    "download": False,
                    "version": None,
                    "path": None,
                    "quiet": True,
                },
            ),
        ],
    )
    def test_cli_parsing(self, cmd, expected):
        args = self._parse(cmd)
        for k, v in expected.items():
            assert getattr(args, k) == v
