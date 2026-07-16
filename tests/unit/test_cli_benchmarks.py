# coding=utf-8
import pytest

# --- Benchmarks Top-level Commands ---


def test_benchmarks_auth_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "auth"])
    assert func.__name__ == "benchmarks_auth_cli"
    assert kwargs.get("no_confirm") is False
    assert kwargs.get("env_file") == ".env"


def test_benchmarks_auth_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "auth", "-y", "--env-file", ".env.prod"])
    assert func.__name__ == "benchmarks_auth_cli"
    assert kwargs["no_confirm"] is True
    assert kwargs["env_file"] == ".env.prod"


def test_benchmarks_init_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "init"])
    assert func.__name__ == "benchmarks_init_cli"
    assert kwargs.get("no_confirm") is False
    assert kwargs.get("env_file") == ".env"
    assert kwargs.get("example_file") == "example_task.py"


def test_benchmarks_init_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        ["benchmarks", "init", "-y", "--env-file", ".env.prod", "--example-file", "my_example.py"]
    )
    assert func.__name__ == "benchmarks_init_cli"
    assert kwargs["no_confirm"] is True
    assert kwargs["env_file"] == ".env.prod"
    assert kwargs["example_file"] == "my_example.py"


def test_benchmarks_leaderboard_parser_missing_benchmark_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "leaderboard"])


def test_benchmarks_leaderboard_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "benchmarks",
            "leaderboard",
            "my-benchmark",
            "--version",
            "2",
            "--show",
            "--download",
            "-p",
            "/tmp/lead",
            "--quiet",
            "--format",
            "json",
        ]
    )
    assert func.__name__ == "benchmark_leaderboard_cli"
    assert kwargs["benchmark"] == "my-benchmark"
    assert kwargs["version"] == 2
    assert kwargs["view"] is True
    assert kwargs["download"] is True
    assert kwargs["path"] == "/tmp/lead"
    assert kwargs["quiet"] is True
    assert kwargs["output_format"] == "json"
    assert kwargs.get("csv_display") is False


def test_benchmarks_topics_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "topics"])
    assert func.__name__ == "benchmark_list_topics_cli"
    assert kwargs.get("entity_ref") is None
    assert kwargs.get("sort_by") is None
    assert kwargs.get("search") is None


def test_benchmarks_topics_list_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "benchmarks",
            "topics",
            "list",
            "my-benchmark",
            "--sort-by",
            "new",
            "-s",
            "search term",
        ]
    )
    assert func.__name__ == "benchmark_list_topics_cli"
    assert kwargs["entity_ref"] == "my-benchmark"
    assert kwargs["sort_by"] == "new"
    assert kwargs["search"] == "search term"


def test_benchmarks_topics_show_missing_topic_ref_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "topics", "show"])


def test_benchmarks_topics_show_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "topics", "show", "topic-ref-url"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "topic-ref-url"
    assert kwargs.get("topic_id_arg") is None


def test_benchmarks_topics_show_two_args_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "topics", "show", "my-benchmark", "12345"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "my-benchmark"
    assert kwargs["topic_id_arg"] == 12345


# --- Benchmarks Tasks Commands ---


def test_benchmarks_tasks_push_parser_missing_args_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "push"])
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "push", "my-task"])  # missing -f


def test_benchmarks_tasks_push_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "benchmarks",
            "tasks",
            "push",
            "my-task",
            "-f",
            "file.py",
            "--wait",
            "100",
            "--poll-interval",
            "30",
            "-v",
            "-d",
            "dataset1",
            "-d",
            "dataset2",
        ]
    )
    assert func.__name__ == "benchmarks_tasks_push_cli"
    assert kwargs["task"] == "my-task"
    assert kwargs["file"] == "file.py"
    assert kwargs["wait"] == 100
    assert kwargs["poll_interval"] == 30
    assert kwargs["verbose"] is True
    assert kwargs["kaggle_datasets"] == ["dataset1", "dataset2"]


def test_benchmarks_tasks_push_parser_wait_no_value_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "tasks", "push", "my-task", "-f", "file.py", "--wait"])
    assert kwargs["wait"] == 0  # const=0


def test_benchmarks_tasks_run_parser_missing_task_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "run"])


def test_benchmarks_tasks_run_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "benchmarks",
            "tasks",
            "run",
            "my-task",
            "-m",
            "model1",
            "-m",
            "model2",
            "--wait",
            "100",
            "--poll-interval",
            "30",
            "-v",
        ]
    )
    assert func.__name__ == "benchmarks_tasks_run_cli"
    assert kwargs["task"] == "my-task"
    assert kwargs["model"] == ["model1", "model2"]
    assert kwargs["wait"] == 100
    assert kwargs["poll_interval"] == 30
    assert kwargs["verbose"] is True


def test_benchmarks_tasks_list_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "tasks", "list"])
    assert func.__name__ == "benchmarks_tasks_list_cli"
    assert kwargs.get("name_regex") is None
    assert kwargs.get("status") is None
    assert kwargs.get("page_size") is None
    assert kwargs.get("show_all") is False


def test_benchmarks_tasks_list_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "benchmarks",
            "tasks",
            "list",
            "--name-regex",
            "test.*",
            "--status",
            "running",
            "--page-size",
            "10",
            "--all",
        ]
    )
    assert func.__name__ == "benchmarks_tasks_list_cli"
    assert kwargs["name_regex"] == "test.*"
    assert kwargs["status"] == "running"
    assert kwargs["page_size"] == 10
    assert kwargs["show_all"] is True


def test_benchmarks_tasks_status_parser_missing_task_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "status"])


def test_benchmarks_tasks_status_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "tasks", "status", "my-task", "-m", "model1"])
    assert func.__name__ == "benchmarks_tasks_status_cli"
    assert kwargs["task"] == "my-task"
    assert kwargs["model"] == ["model1"]


def test_benchmarks_tasks_download_parser_missing_task_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "download"])


def test_benchmarks_tasks_download_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "benchmarks",
            "tasks",
            "download",
            "my-task",
            "-m",
            "model1",
            "-o",
            "/tmp/out",
            "--include-source",
            "--force",
        ]
    )
    assert func.__name__ == "benchmarks_tasks_download_cli"
    assert kwargs["task"] == "my-task"
    assert kwargs["model"] == ["model1"]
    assert kwargs["output"] == "/tmp/out"
    assert kwargs["include_source"] is True
    assert kwargs["force"] is True


def test_benchmarks_tasks_log_parser_missing_task_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "log"])


def test_benchmarks_tasks_log_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "tasks", "log", "my-task", "-m", "model1"])
    assert func.__name__ == "benchmarks_tasks_log_cli"
    assert kwargs["task"] == "my-task"
    assert kwargs["model"] == ["model1"]


def test_benchmarks_tasks_models_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "tasks", "models"])
    assert func.__name__ == "benchmarks_tasks_models_cli"
    assert kwargs == {}


def test_benchmarks_tasks_delete_parser_missing_task_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "delete"])


def test_benchmarks_tasks_delete_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "tasks", "delete", "my-task", "-y"])
    assert func.__name__ == "benchmarks_tasks_delete_cli"
    assert kwargs["task"] == "my-task"
    assert kwargs["no_confirm"] is True


def test_benchmarks_tasks_publish_parser_missing_task_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["benchmarks", "tasks", "publish"])


def test_benchmarks_tasks_publish_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["benchmarks", "tasks", "publish", "my-task", "--no-publish-backing-notebook"])
    assert func.__name__ == "benchmarks_tasks_publish_cli"
    assert kwargs["task"] == "my-task"
    assert kwargs["publish_backing_notebook"] is False
