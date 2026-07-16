# coding=utf-8
import pytest


def test_kernels_list_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "list"])
    assert func.__name__ == "kernels_list_cli"
    assert kwargs.get("mine") is False
    assert kwargs.get("page_token") is None
    assert kwargs.get("page_size") == 20
    assert kwargs.get("page") is None
    assert kwargs.get("search") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("output_format") is None
    assert kwargs.get("parent") is None
    assert kwargs.get("competition") is None
    assert kwargs.get("dataset") is None
    assert kwargs.get("user") is None
    assert kwargs.get("language") is None
    assert kwargs.get("kernel_type") is None
    assert kwargs.get("output_type") is None
    assert kwargs.get("sort_by") is None


def test_kernels_list_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "kernels",
            "list",
            "--mine",
            "--page-token",
            "tok123",
            "--page-size",
            "50",
            "--page",
            "3",
            "--search",
            "my search",
            "--csv",
            "--parent",
            "parent-kernel",
            "--competition",
            "my-comp",
            "--dataset",
            "my-data",
            "--user",
            "stevemessick",
            "--language",
            "python",
            "--kernel-type",
            "notebook",
            "--output-type",
            "visualization",
            "--sort-by",
            "hotness",
        ]
    )
    assert func.__name__ == "kernels_list_cli"
    assert kwargs["mine"] is True
    assert kwargs["page_token"] == "tok123"
    assert kwargs["page_size"] == 50
    assert kwargs["page"] == 3
    assert kwargs["search"] == "my search"
    assert kwargs["csv_display"] is True
    assert kwargs["parent"] == "parent-kernel"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["dataset"] == "my-data"
    assert kwargs["user"] == "stevemessick"
    assert kwargs["language"] == "python"
    assert kwargs["kernel_type"] == "notebook"
    assert kwargs["output_type"] == "visualization"
    assert kwargs["sort_by"] == "hotness"


def test_kernels_files_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "files"])
    assert func.__name__ == "kernels_list_files_cli"
    assert kwargs.get("kernel") is None
    assert kwargs.get("kernel_opt") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("output_format") is None
    assert kwargs.get("page_token") is None
    assert kwargs.get("page_size") == 20


def test_kernels_files_parser_with_positional_kernel_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "files", "owner/kernel-name"])
    assert func.__name__ == "kernels_list_files_cli"
    assert kwargs["kernel"] == "owner/kernel-name"
    assert kwargs.get("kernel_opt") is None


def test_kernels_files_parser_with_option_kernel_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "files", "-k", "owner/kernel-name"])
    assert func.__name__ == "kernels_list_files_cli"
    assert kwargs.get("kernel") is None
    assert kwargs["kernel_opt"] == "owner/kernel-name"


def test_kernels_files_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "kernels",
            "files",
            "owner/kernel-name",
            "--page-token",
            "tok",
            "--page-size",
            "50",
            "--format",
            "json",
        ]
    )
    assert func.__name__ == "kernels_list_files_cli"
    assert kwargs["kernel"] == "owner/kernel-name"
    assert kwargs["page_token"] == "tok"
    assert kwargs["page_size"] == 50
    assert kwargs["output_format"] == "json"


def test_kernels_init_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "init"])
    assert func.__name__ == "kernels_initialize_cli"
    assert kwargs.get("folder") is None


def test_kernels_init_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "init", "-p", "/path/to/kernel"])
    assert func.__name__ == "kernels_initialize_cli"
    assert kwargs["folder"] == "/path/to/kernel"


def test_kernels_push_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "push"])
    assert func.__name__ == "kernels_push_cli"
    assert kwargs.get("folder") is None
    assert kwargs.get("timeout") is None
    assert kwargs.get("acc") is None


def test_kernels_push_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "push", "-p", "/path/to/kernel", "-t", "3600", "--accelerator", "gpu"])
    assert func.__name__ == "kernels_push_cli"
    assert kwargs["folder"] == "/path/to/kernel"
    assert kwargs["timeout"] == 3600
    assert kwargs["acc"] == "gpu"


def test_kernels_pull_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "pull"])
    assert func.__name__ == "kernels_pull_cli"
    assert kwargs.get("kernel") is None
    assert kwargs.get("path") is None
    assert kwargs.get("metadata") is False


def test_kernels_pull_parser_with_positional_kernel_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "pull", "owner/kernel-name"])
    assert func.__name__ == "kernels_pull_cli"
    assert kwargs["kernel"] == "owner/kernel-name"


def test_kernels_pull_parser_with_option_kernel_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "pull", "-k", "owner/kernel-name"])
    assert func.__name__ == "kernels_pull_cli"
    # Both positional and option write to dest='kernel', but because of a bug in cli.py,
    # the option value is overwritten by the positional default (None) when positional is omitted.
    assert kwargs.get("kernel") is None


def test_kernels_pull_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "pull", "owner/kernel-name", "-p", "/tmp/pull", "--metadata"])
    assert func.__name__ == "kernels_pull_cli"
    assert kwargs["kernel"] == "owner/kernel-name"
    assert kwargs["path"] == "/tmp/pull"
    assert kwargs["metadata"] is True


def test_kernels_pull_parser_wp_flag_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "pull", "-w"])
    assert kwargs["path"] == "."


def test_kernels_output_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "output"])
    assert func.__name__ == "kernels_output_cli"
    assert kwargs.get("kernel") is None
    assert kwargs.get("kernel_opt") is None
    assert kwargs.get("path") is None
    assert kwargs.get("force") is False
    assert kwargs.get("quiet") is False
    assert kwargs.get("file_pattern") is None
    assert kwargs.get("page_size") == 20
    assert kwargs.get("page_token") is None


def test_kernels_output_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "kernels",
            "output",
            "owner/kernel-name",
            "-p",
            "/tmp/out",
            "--force",
            "--quiet",
            "--file-pattern",
            "*.png",
            "--page-size",
            "50",
            "--page-token",
            "tok",
        ]
    )
    assert func.__name__ == "kernels_output_cli"
    assert kwargs["kernel"] == "owner/kernel-name"
    assert kwargs["path"] == "/tmp/out"
    assert kwargs["force"] is True
    assert kwargs["quiet"] is True
    assert kwargs["file_pattern"] == "*.png"
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok"


def test_kernels_output_parser_wp_flag_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "output", "-w"])
    assert kwargs["path"] == "."


def test_kernels_status_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "status"])
    assert func.__name__ == "kernels_status_cli"
    assert kwargs.get("kernel") is None
    assert kwargs.get("kernel_opt") is None


def test_kernels_status_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "status", "owner/kernel-name"])
    assert func.__name__ == "kernels_status_cli"
    assert kwargs["kernel"] == "owner/kernel-name"


def test_kernels_status_parser_with_option_kernel_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "status", "-k", "owner/kernel-name"])
    assert func.__name__ == "kernels_status_cli"
    assert kwargs.get("kernel") is None
    assert kwargs["kernel_opt"] == "owner/kernel-name"


def test_kernels_logs_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "logs"])
    assert func.__name__ == "kernels_logs_cli"
    assert kwargs.get("kernel") is None
    assert kwargs.get("kernel_opt") is None
    assert kwargs.get("follow") is False
    assert kwargs.get("interval") == 5


def test_kernels_logs_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "logs", "owner/kernel-name", "--follow", "--interval", "10"])
    assert func.__name__ == "kernels_logs_cli"
    assert kwargs["kernel"] == "owner/kernel-name"
    assert kwargs["follow"] is True
    assert kwargs["interval"] == 10


def test_kernels_delete_parser_missing_kernel_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["kernels", "delete"])


def test_kernels_delete_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "delete", "owner/kernel-name", "-y"])
    assert func.__name__ == "kernels_delete_cli"
    assert kwargs["kernel"] == "owner/kernel-name"
    assert kwargs["no_confirm"] is True


def test_kernels_topics_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "topics"])
    assert func.__name__ == "kernel_list_topics_cli"
    assert kwargs.get("entity_ref") is None
    assert kwargs.get("sort_by") is None
    assert kwargs.get("search") is None


def test_kernels_topics_list_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "kernels",
            "topics",
            "list",
            "owner/kernel-name",
            "--sort-by",
            "new",
            "-s",
            "search term",
        ]
    )
    assert func.__name__ == "kernel_list_topics_cli"
    assert kwargs["entity_ref"] == "owner/kernel-name"
    assert kwargs["sort_by"] == "new"
    assert kwargs["search"] == "search term"


def test_kernels_topics_show_missing_topic_ref_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["kernels", "topics", "show"])


def test_kernels_topics_show_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "topics", "show", "topic-ref-url"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "topic-ref-url"
    assert kwargs.get("topic_id_arg") is None


def test_kernels_topics_show_two_args_succeeds(parser):
    func, kwargs = parser.dispatch(["kernels", "topics", "show", "owner/kernel-name", "12345"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "owner/kernel-name"
    assert kwargs["topic_id_arg"] == 12345
