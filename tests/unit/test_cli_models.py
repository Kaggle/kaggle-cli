# coding=utf-8
import pytest

# --- Models Top-level Commands ---


def test_models_get_parser_missing_model_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "get"])


def test_models_get_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "get", "owner/model-name", "-p", "/tmp/model"])
    assert func.__name__ == "model_get_cli"
    assert kwargs["model"] == "owner/model-name"
    assert kwargs["folder"] == "/tmp/model"


def test_models_list_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "list"])
    assert func.__name__ == "model_list_cli"
    assert kwargs.get("sort_by") is None
    assert kwargs.get("search") is None
    assert kwargs.get("owner") is None
    assert kwargs.get("page_size") == 20
    assert kwargs.get("page_token") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("output_format") is None


def test_models_list_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "list",
            "--sort-by",
            "votes",
            "-s",
            "search term",
            "--owner",
            "stevemessick",
            "--page-size",
            "50",
            "--page-token",
            "tok",
            "--csv",
        ]
    )
    assert func.__name__ == "model_list_cli"
    assert kwargs["sort_by"] == "votes"
    assert kwargs["search"] == "search term"
    assert kwargs["owner"] == "stevemessick"
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok"
    assert kwargs["csv_display"] is True


def test_models_init_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "init"])
    assert func.__name__ == "model_initialize_cli"
    assert kwargs.get("folder") is None


def test_models_init_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "init", "-p", "/path/to/model"])
    assert func.__name__ == "model_initialize_cli"
    assert kwargs["folder"] == "/path/to/model"


def test_models_create_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "create"])
    assert func.__name__ == "model_create_new_cli"
    assert kwargs.get("folder") is None


def test_models_create_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "create", "-p", "/path/to/model"])
    assert func.__name__ == "model_create_new_cli"
    assert kwargs["folder"] == "/path/to/model"


def test_models_delete_parser_missing_model_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "delete"])


def test_models_delete_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "delete", "owner/model-name", "-y"])
    assert func.__name__ == "model_delete_cli"
    assert kwargs["model"] == "owner/model-name"
    assert kwargs["no_confirm"] is True


def test_models_update_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "update"])
    assert func.__name__ == "model_update_cli"
    assert kwargs.get("folder") is None


def test_models_update_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "update", "-p", "/path/to/model"])
    assert func.__name__ == "model_update_cli"
    assert kwargs["folder"] == "/path/to/model"


def test_models_topics_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "topics"])
    assert func.__name__ == "model_list_topics_cli"
    assert kwargs.get("entity_ref") is None
    assert kwargs.get("sort_by") is None
    assert kwargs.get("search") is None


def test_models_topics_list_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "topics",
            "list",
            "owner/model-name",
            "--sort-by",
            "new",
            "-s",
            "search term",
        ]
    )
    assert func.__name__ == "model_list_topics_cli"
    assert kwargs["entity_ref"] == "owner/model-name"
    assert kwargs["sort_by"] == "new"
    assert kwargs["search"] == "search term"


def test_models_topics_show_missing_topic_ref_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "topics", "show"])


def test_models_topics_show_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "topics", "show", "topic-ref-url"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "topic-ref-url"
    assert kwargs.get("topic_id_arg") is None


def test_models_topics_show_two_args_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "topics", "show", "owner/model-name", "12345"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "owner/model-name"
    assert kwargs["topic_id_arg"] == 12345


# --- Model Instances Commands ---


def test_model_instances_get_parser_missing_instance_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "get"])


def test_model_instances_get_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        ["models", "instances", "get", "owner/model/framework/variation", "-p", "/tmp/instance"]
    )
    assert func.__name__ == "model_instance_get_cli"
    assert kwargs["model_instance"] == "owner/model/framework/variation"
    assert kwargs["folder"] == "/tmp/instance"


def test_model_instances_init_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "instances", "init"])
    assert func.__name__ == "model_instance_initialize_cli"
    assert kwargs.get("folder") is None


def test_model_instances_init_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "instances", "init", "-p", "/path/to/instance"])
    assert func.__name__ == "model_instance_initialize_cli"
    assert kwargs["folder"] == "/path/to/instance"


def test_model_instances_create_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "instances", "create"])
    assert func.__name__ == "model_instance_create_cli"
    assert kwargs.get("folder") is None
    assert kwargs.get("quiet") is False
    assert kwargs.get("dir_mode") == "skip"


def test_model_instances_create_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "create",
            "-p",
            "/path/to/instance",
            "--quiet",
            "-r",
            "zip",
        ]
    )
    assert func.__name__ == "model_instance_create_cli"
    assert kwargs["folder"] == "/path/to/instance"
    assert kwargs["quiet"] is True
    assert kwargs["dir_mode"] == "zip"
    assert kwargs.get("ignore_patterns") is None


def test_model_instances_create_parser_with_ignore_patterns_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "create",
            "--ignore-patterns",
            "*.tmp",
            "--ignore-patterns",
            "temp/",
        ]
    )
    assert func.__name__ == "model_instance_create_cli"
    assert kwargs["ignore_patterns"] == ["*.tmp", "temp/"]


def test_model_instances_files_parser_missing_instance_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "files"])


def test_model_instances_files_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "files",
            "owner/model/framework/variation",
            "--page-size",
            "50",
            "--page-token",
            "tok",
            "--format",
            "json",
        ]
    )
    assert func.__name__ == "model_instance_files_cli"
    assert kwargs["model_instance"] == "owner/model/framework/variation"
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok"
    assert kwargs["output_format"] == "json"


def test_model_instances_list_parser_missing_instance_fails(parser):
    # Wait, 'list' requires model_instance?
    # In cli.py:
    # parser_model_instances_list_optional.add_argument("model_instance", help=Help.param_model_instance)
    # Yes, it is positional and not nargs='?', so it is REQUIRED.
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "list"])


def test_model_instances_list_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "list",
            "owner/model/framework/variation",
            "--page-size",
            "50",
            "--page-token",
            "tok",
            "--format",
            "json",
        ]
    )
    assert func.__name__ == "model_instances_list_cli"
    assert kwargs["model_instance"] == "owner/model/framework/variation"
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok"
    assert kwargs["output_format"] == "json"


def test_model_instances_delete_parser_missing_instance_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "delete"])


def test_model_instances_delete_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "instances", "delete", "owner/model/framework/variation", "-y"])
    assert func.__name__ == "model_instance_delete_cli"
    assert kwargs["model_instance"] == "owner/model/framework/variation"
    assert kwargs["no_confirm"] is True


def test_model_instances_update_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "instances", "update"])
    assert func.__name__ == "model_instance_update_cli"
    assert kwargs.get("folder") is None


def test_model_instances_update_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["models", "instances", "update", "-p", "/path/to/instance"])
    assert func.__name__ == "model_instance_update_cli"
    assert kwargs["folder"] == "/path/to/instance"


# --- Model Instance Versions Commands ---


def test_model_instance_versions_list_parser_missing_instance_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "versions", "list"])


def test_model_instance_versions_list_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "versions",
            "list",
            "owner/model/framework/variation",
            "--page-size",
            "50",
            "--page-token",
            "tok",
            "--format",
            "json",
        ]
    )
    assert func.__name__ == "model_instance_versions_list_cli"
    assert kwargs["model_instance"] == "owner/model/framework/variation"
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok"
    assert kwargs["output_format"] == "json"


def test_model_instance_versions_create_parser_missing_instance_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "versions", "create"])


def test_model_instance_versions_create_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "versions",
            "create",
            "owner/model/framework/variation",
            "-p",
            "/path/to/version",
            "-n",
            "version notes",
            "--quiet",
            "-r",
            "tar",
        ]
    )
    assert func.__name__ == "model_instance_version_create_cli"
    assert kwargs["model_instance"] == "owner/model/framework/variation"
    assert kwargs["folder"] == "/path/to/version"
    assert kwargs["version_notes"] == "version notes"
    assert kwargs["quiet"] is True
    assert kwargs["dir_mode"] == "tar"
    assert kwargs.get("ignore_patterns") is None


def test_model_instance_versions_create_parser_with_ignore_patterns_succeeds(
    parser,
):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "versions",
            "create",
            "owner/model/framework/variation",
            "--ignore-patterns",
            "*.tmp",
            "--ignore-patterns",
            "temp/",
        ]
    )
    assert func.__name__ == "model_instance_version_create_cli"
    assert kwargs["model_instance"] == "owner/model/framework/variation"
    assert kwargs["ignore_patterns"] == ["*.tmp", "temp/"]


def test_model_instance_versions_download_parser_missing_version_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "versions", "download"])


def test_model_instance_versions_download_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "versions",
            "download",
            "owner/model/framework/variation/1",
            "-p",
            "/tmp/download",
            "--untar",
            "--force",
            "--quiet",
        ]
    )
    assert func.__name__ == "model_instance_version_download_cli"
    assert kwargs["model_instance_version"] == "owner/model/framework/variation/1"
    assert kwargs["path"] == "/tmp/download"
    assert kwargs["untar"] is True
    assert kwargs["force"] is True
    assert kwargs["quiet"] is True


def test_model_instance_versions_files_parser_missing_version_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "versions", "files"])


def test_model_instance_versions_files_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "versions",
            "files",
            "owner/model/framework/variation/1",
            "--page-size",
            "50",
            "--page-token",
            "tok",
            "--format",
            "json",
        ]
    )
    assert func.__name__ == "model_instance_version_files_cli"
    assert kwargs["model_instance_version"] == "owner/model/framework/variation/1"
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok"
    assert kwargs["output_format"] == "json"


def test_model_instance_versions_delete_parser_missing_version_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["models", "instances", "versions", "delete"])


def test_model_instance_versions_delete_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "models",
            "instances",
            "versions",
            "delete",
            "owner/model/framework/variation/1",
            "-y",
        ]
    )
    assert func.__name__ == "model_instance_version_delete_cli"
    assert kwargs["model_instance_version"] == "owner/model/framework/variation/1"
    assert kwargs["no_confirm"] is True
