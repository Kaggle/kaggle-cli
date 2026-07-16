# coding=utf-8
import pytest


def test_datasets_list_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "list"])
    assert func.__name__ == "dataset_list_cli"
    assert kwargs.get("sort_by") is None
    assert kwargs.get("size") is None
    assert kwargs.get("file_type") is None
    assert kwargs.get("license_name") is None
    assert kwargs.get("tag_ids") is None
    assert kwargs.get("search") is None
    assert kwargs.get("mine") is False
    assert kwargs.get("user") is None
    assert kwargs.get("page_size") is None
    assert kwargs.get("page_token") is None
    assert kwargs.get("page") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("output_format") is None
    assert kwargs.get("max_size") is None
    assert kwargs.get("min_size") is None


def test_datasets_list_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "datasets",
            "list",
            "--sort-by",
            "votes",
            "--size",
            "100",
            "--file-type",
            "csv",
            "--license",
            "cc",
            "--tags",
            "tag1,tag2",
            "--search",
            "my search",
            "--mine",
            "--user",
            "stevemessick",
            "--page-size",
            "50",
            "--page-token",
            "tok123",
            "--page",
            "2",
            "--csv",
            "--max-size",
            "1000",
            "--min-size",
            "10",
        ]
    )
    assert func.__name__ == "dataset_list_cli"
    assert kwargs["sort_by"] == "votes"
    assert kwargs["size"] == 100
    assert kwargs["file_type"] == "csv"
    assert kwargs["license_name"] == "cc"
    assert kwargs["tag_ids"] == "tag1,tag2"
    assert kwargs["search"] == "my search"
    assert kwargs["mine"] is True
    assert kwargs["user"] == "stevemessick"
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok123"
    assert kwargs["page"] == 2
    assert kwargs["csv_display"] is True
    assert kwargs["max_size"] == 1000
    assert kwargs["min_size"] == 10


def test_datasets_files_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "files"])
    assert func.__name__ == "dataset_list_files_cli"
    assert kwargs.get("dataset") is None
    assert kwargs.get("dataset_opt") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("output_format") is None
    assert kwargs.get("page_token") is None
    assert kwargs.get("page_size") == 20


def test_datasets_files_parser_with_positional_dataset_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "files", "owner/dataset-name"])
    assert func.__name__ == "dataset_list_files_cli"
    assert kwargs["dataset"] == "owner/dataset-name"
    assert kwargs.get("dataset_opt") is None


def test_datasets_files_parser_with_option_dataset_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "files", "-d", "owner/dataset-name"])
    assert func.__name__ == "dataset_list_files_cli"
    assert kwargs.get("dataset") is None
    assert kwargs["dataset_opt"] == "owner/dataset-name"


def test_datasets_files_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "datasets",
            "files",
            "owner/dataset-name",
            "--page-token",
            "tok",
            "--page-size",
            "50",
            "--format",
            "json",
        ]
    )
    assert func.__name__ == "dataset_list_files_cli"
    assert kwargs["dataset"] == "owner/dataset-name"
    assert kwargs["page_token"] == "tok"
    assert kwargs["page_size"] == 50
    assert kwargs["output_format"] == "json"


def test_datasets_download_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "download"])
    assert func.__name__ == "dataset_download_cli"
    assert kwargs.get("dataset") is None
    assert kwargs.get("dataset_opt") is None
    assert kwargs.get("file_name") is None
    assert kwargs.get("path") is None
    assert kwargs.get("unzip") is False
    assert kwargs.get("force") is False
    assert kwargs.get("quiet") is False


def test_datasets_download_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "datasets",
            "download",
            "owner/dataset-name",
            "-f",
            "file.csv",
            "-p",
            "/tmp/download",
            "--unzip",
            "--force",
            "--quiet",
        ]
    )
    assert func.__name__ == "dataset_download_cli"
    assert kwargs["dataset"] == "owner/dataset-name"
    assert kwargs["file_name"] == "file.csv"
    assert kwargs["path"] == "/tmp/download"
    assert kwargs["unzip"] is True
    assert kwargs["force"] is True
    assert kwargs["quiet"] is True


def test_datasets_download_parser_wp_flag_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "download", "-w"])
    assert kwargs["path"] == "."


def test_datasets_create_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "create"])
    assert func.__name__ == "dataset_create_new_cli"
    assert kwargs.get("folder") is None
    assert kwargs.get("public") is False
    assert kwargs.get("quiet") is False
    assert kwargs.get("convert_to_csv") is True
    assert kwargs.get("dir_mode") == "skip"


def test_datasets_create_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "datasets",
            "create",
            "-p",
            "/path/to/data",
            "--public",
            "--quiet",
            "-t",
            "-r",
            "zip",
        ]
    )
    assert func.__name__ == "dataset_create_new_cli"
    assert kwargs["folder"] == "/path/to/data"
    assert kwargs["public"] is True
    assert kwargs["quiet"] is True
    assert kwargs["convert_to_csv"] is False
    assert kwargs["dir_mode"] == "zip"


def test_datasets_version_parser_missing_message_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["datasets", "version", "-p", "/path/to/data"])


def test_datasets_version_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "datasets",
            "version",
            "-m",
            "version message",
            "-p",
            "/path/to/data",
            "--quiet",
            "-t",
            "-r",
            "tar",
            "-d",
        ]
    )
    assert func.__name__ == "dataset_create_version_cli"
    assert kwargs["version_notes"] == "version message"
    assert kwargs["folder"] == "/path/to/data"
    assert kwargs["quiet"] is True
    assert kwargs["convert_to_csv"] is False
    assert kwargs["dir_mode"] == "tar"
    assert kwargs["delete_old_versions"] is True


def test_datasets_init_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "init"])
    assert func.__name__ == "dataset_initialize_cli"
    assert kwargs.get("folder") is None


def test_datasets_init_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "init", "-p", "/path/to/data"])
    assert func.__name__ == "dataset_initialize_cli"
    assert kwargs["folder"] == "/path/to/data"


def test_datasets_metadata_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "metadata"])
    assert func.__name__ == "dataset_metadata_cli"
    assert kwargs.get("dataset") is None
    assert kwargs.get("dataset_opt") is None
    assert kwargs.get("update") is False
    assert kwargs.get("path") is None


def test_datasets_metadata_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "datasets",
            "metadata",
            "owner/dataset-name",
            "--update",
            "-p",
            "/path/to/metadata",
        ]
    )
    assert func.__name__ == "dataset_metadata_cli"
    assert kwargs["dataset"] == "owner/dataset-name"
    assert kwargs["update"] is True
    assert kwargs["path"] == "/path/to/metadata"


def test_datasets_status_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "status"])
    assert func.__name__ == "dataset_status_cli"
    assert kwargs.get("dataset") is None
    assert kwargs.get("dataset_opt") is None
    assert kwargs.get("format") is None


def test_datasets_status_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "status", "owner/dataset-name", "--format", "json"])
    assert func.__name__ == "dataset_status_cli"
    assert kwargs["dataset"] == "owner/dataset-name"
    assert kwargs["format"] == "json"


def test_datasets_delete_parser_missing_dataset_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["datasets", "delete"])


def test_datasets_delete_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "delete", "owner/dataset-name", "-y"])
    assert func.__name__ == "dataset_delete_cli"
    assert kwargs["dataset"] == "owner/dataset-name"
    assert kwargs["no_confirm"] is True


def test_datasets_topics_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "topics"])
    assert func.__name__ == "dataset_list_topics_cli"
    assert kwargs.get("entity_ref") is None
    assert kwargs.get("sort_by") is None
    assert kwargs.get("search") is None


def test_datasets_topics_list_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "datasets",
            "topics",
            "list",
            "owner/dataset-name",
            "--sort-by",
            "new",
            "-s",
            "search term",
        ]
    )
    assert func.__name__ == "dataset_list_topics_cli"
    assert kwargs["entity_ref"] == "owner/dataset-name"
    assert kwargs["sort_by"] == "new"
    assert kwargs["search"] == "search term"


def test_datasets_topics_show_missing_topic_ref_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["datasets", "topics", "show"])


def test_datasets_topics_show_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "topics", "show", "topic-ref-url"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "topic-ref-url"
    assert kwargs.get("topic_id_arg") is None


def test_datasets_topics_show_two_args_succeeds(parser):
    func, kwargs = parser.dispatch(["datasets", "topics", "show", "owner/dataset-name", "12345"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "owner/dataset-name"
    assert kwargs["topic_id_arg"] == 12345
