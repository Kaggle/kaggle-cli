# coding=utf-8
import pytest


def test_competitions_list_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "list"])
    assert func.__name__ == "competitions_list_cli"
    assert kwargs.get("group") is None
    assert kwargs.get("category") is None
    assert kwargs.get("sort_by") is None
    assert kwargs.get("page") == -1
    assert kwargs.get("search") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("page_size") is None
    assert kwargs.get("page_token") is None
    assert kwargs.get("output_format") is None


def test_competitions_list_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "list",
            "--group",
            "entered",
            "--category",
            "featured",
            "--sort-by",
            "prize",
            "--page",
            "3",
            "--search",
            "test",
            "--csv",
            "--page-size",
            "50",
        ]
    )
    assert func.__name__ == "competitions_list_cli"
    assert kwargs["group"] == "entered"
    assert kwargs["category"] == "featured"
    assert kwargs["sort_by"] == "prize"
    assert kwargs["page"] == 3
    assert kwargs["search"] == "test"
    assert kwargs["csv_display"] is True
    assert kwargs["page_size"] == 50
    assert kwargs.get("page_token") is None
    assert kwargs.get("output_format") is None


def test_competitions_list_parser_with_page_token_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "list", "--page-token", "token123", "--format", "json"])
    assert func.__name__ == "competitions_list_cli"
    assert kwargs["page_token"] == "token123"
    assert kwargs["output_format"] == "json"
    assert kwargs["csv_display"] is False
    assert kwargs["page"] == -1


def test_competitions_files_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "files"])
    assert func.__name__ == "competition_list_files_cli"
    assert kwargs.get("competition") is None
    assert kwargs.get("competition_opt") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("output_format") is None
    assert kwargs.get("quiet") is False
    assert kwargs.get("page_token") is None
    assert kwargs.get("page_size") == 20


def test_competitions_files_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "files",
            "my-competition",
            "--quiet",
            "--page-size",
            "50",
            "--page-token",
            "tok",
        ]
    )
    assert func.__name__ == "competition_list_files_cli"
    assert kwargs["competition"] == "my-competition"
    assert kwargs["quiet"] is True
    assert kwargs["page_size"] == 50
    assert kwargs["page_token"] == "tok"


def test_competitions_download_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "download"])
    assert func.__name__ == "competition_download_cli"
    assert kwargs.get("competition") is None
    assert kwargs.get("file_name") is None
    assert kwargs.get("path") is None
    assert kwargs.get("force") is False
    assert kwargs.get("quiet") is False


def test_competitions_download_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "download",
            "my-competition",
            "-f",
            "file.csv",
            "-p",
            "/tmp/download",
            "--force",
            "--quiet",
        ]
    )
    assert func.__name__ == "competition_download_cli"
    assert kwargs["competition"] == "my-competition"
    assert kwargs["file_name"] == "file.csv"
    assert kwargs["path"] == "/tmp/download"
    assert kwargs["force"] is True
    assert kwargs["quiet"] is True


def test_competitions_download_parser_wp_flag_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "download", "-w"])
    assert kwargs["path"] == "."


def test_competitions_submit_parser_missing_message_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "submit", "my-comp", "-f", "sub.csv"])


def test_competitions_submit_parser_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "submit",
            "my-comp",
            "-f",
            "sub.csv",
            "-m",
            "my message",
            "--sandbox",
        ]
    )
    assert func.__name__ == "competition_submit_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["file_name"] == "sub.csv"
    assert kwargs["message"] == "my message"
    assert kwargs["sandbox"] is True
    assert kwargs.get("kernel") is None
    assert kwargs.get("version") is None


def test_competitions_submissions_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "submissions"])
    assert func.__name__ == "competition_submissions_cli"
    assert kwargs.get("competition") is None
    assert kwargs.get("csv_display") is False
    assert kwargs.get("page_size") is None


def test_competitions_submissions_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "submissions", "my-comp", "--page-size", "10", "--csv"])
    assert func.__name__ == "competition_submissions_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["page_size"] == 10
    assert kwargs["csv_display"] is True


def test_competitions_leaderboard_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "leaderboard"])
    assert func.__name__ == "competition_leaderboard_cli"
    assert kwargs.get("competition") is None
    assert kwargs.get("view") is False
    assert kwargs.get("download") is False
    assert kwargs.get("path") is None


def test_competitions_leaderboard_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        ["competitions", "leaderboard", "my-comp", "--show", "--download", "-p", "/tmp/lead"]
    )
    assert func.__name__ == "competition_leaderboard_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["view"] is True
    assert kwargs["download"] is True
    assert kwargs["path"] == "/tmp/lead"


def test_competitions_team_submissions_parser_missing_team_id_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "team-submissions"])


def test_competitions_team_submissions_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "team-submissions", "12345"])
    assert func.__name__ == "competition_team_submissions_cli"
    assert kwargs["team_id"] == 12345


def test_competitions_submission_limits_positional_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "submission-limits", "my-comp", "--json"])
    assert func.__name__ == "competition_get_submission_limits_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["json_output"] is True


def test_competitions_submission_limits_dash_c_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "submission-limits", "-c", "my-comp"])
    assert func.__name__ == "competition_get_submission_limits_cli"
    assert kwargs["competition"] is None
    assert kwargs["competition_opt"] == "my-comp"
    assert kwargs["json_output"] is False


def test_competitions_episodes_parser_missing_sub_id_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "episodes"])


def test_competitions_episodes_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "episodes", "123", "--csv"])
    assert func.__name__ == "competition_list_episodes_cli"
    assert kwargs["submission_id"] == 123
    assert kwargs["csv_display"] is True


def test_competitions_replay_parser_missing_episode_id_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "replay"])


def test_competitions_replay_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "replay", "456", "-p", "/tmp/replay"])
    assert func.__name__ == "competition_episode_replay_cli"
    assert kwargs["episode_id"] == 456
    assert kwargs["path"] == "/tmp/replay"


def test_competitions_logs_parser_missing_args_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "logs"])
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "logs", "456"])


def test_competitions_logs_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "logs", "456", "1", "-p", "/tmp/logs"])
    assert func.__name__ == "competition_episode_agent_logs_cli"
    assert kwargs["episode_id"] == 456
    assert kwargs["agent_index"] == 1
    assert kwargs["path"] == "/tmp/logs"


def test_competitions_pages_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "pages", "-c", "my-comp"])
    assert func.__name__ == "competition_list_pages_cli"
    assert kwargs.get("competition") is None
    assert kwargs["competition_opt"] == "my-comp"
    assert kwargs["content"] is False
    assert kwargs["page_name"] is None


def test_competitions_pages_list_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "pages",
            "list",
            "-c",
            "my-comp",
            "--content",
            "--page-name",
            "rules",
        ]
    )
    assert func.__name__ == "competition_list_pages_cli"
    assert kwargs.get("competition") is None
    assert kwargs["competition_opt"] == "my-comp"
    assert kwargs["content"] is True
    assert kwargs["page_name"] == "rules"


def test_competitions_pages_create_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "pages",
            "create",
            "-c",
            "my-comp",
            "--page-name",
            "rules",
            "-f",
            "rules.html",
            "--mime-type",
            "text/html",
            "--post-title",
            "Rules",
            "--publish",
        ]
    )
    assert func.__name__ == "competition_create_page_cli"
    assert kwargs.get("competition") is None
    assert kwargs["competition_opt"] == "my-comp"
    assert kwargs["page_name"] == "rules"
    assert kwargs["file_path"] == "rules.html"
    assert kwargs["mime_type"] == "text/html"
    assert kwargs["post_title"] == "Rules"
    assert kwargs["publish"] is True


def test_competitions_pages_update_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "pages",
            "update",
            "-c",
            "my-comp",
            "--page-name",
            "rules",
            "-f",
            "new_rules.html",
            "--new-name",
            "new-rules",
            "--publish",
        ]
    )
    assert func.__name__ == "competition_update_page_cli"
    assert kwargs.get("competition") is None
    assert kwargs["competition_opt"] == "my-comp"
    assert kwargs["page_name"] == "rules"
    assert kwargs["file_path"] == "new_rules.html"
    assert kwargs["new_name"] == "new-rules"
    assert kwargs["publish"] is True
    assert kwargs.get("unpublish") is False


def test_competitions_pages_delete_succeeds(parser):
    func, kwargs = parser.dispatch(
        ["competitions", "pages", "delete", "-c", "my-comp", "--page-name", "rules", "--yes", "--quiet"]
    )
    assert func.__name__ == "competition_delete_page_cli"
    assert kwargs.get("competition") is None
    assert kwargs["competition_opt"] == "my-comp"
    assert kwargs["page_name"] == "rules"
    assert kwargs["no_confirm"] is True
    assert kwargs["quiet"] is True


def test_competitions_hosts_dash_c_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "hosts", "-c", "my-comp"])
    assert func.__name__ == "competition_list_hosts_cli"
    assert kwargs.get("competition") is None
    assert kwargs["competition_opt"] == "my-comp"


def test_competitions_hosts_positional_succeeds(parser):
    # `hosts` is a flat parser (used to be a subcommand group, but argparse
    # can't disambiguate a parent positional from a subcommand token).
    func, kwargs = parser.dispatch(["competitions", "hosts", "my-comp"])
    assert func.__name__ == "competition_list_hosts_cli"
    assert kwargs["competition"] == "my-comp"


def test_competitions_data_update_missing_args_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "data", "update", "my-comp"])


def test_competitions_data_update_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "data",
            "update",
            "my-comp",
            "-p",
            "/path/to/data",
            "-m",
            "update msg",
            "--rerun",
            "--include-hidden",
        ]
    )
    assert func.__name__ == "competition_data_update_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["path"] == "/path/to/data"
    assert kwargs["version_notes"] == "update msg"
    assert kwargs["rerun"] is True
    assert kwargs["include_hidden"] is True
    assert kwargs.get("ignore_patterns") is None


def test_competitions_data_update_with_ignore_patterns_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "data",
            "update",
            "my-comp",
            "-p",
            "/path/to/data",
            "-m",
            "update msg",
            "--ignore-patterns",
            "*.tmp",
            "--ignore-patterns",
            "temp/",
        ]
    )
    assert func.__name__ == "competition_data_update_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["path"] == "/path/to/data"
    assert kwargs["version_notes"] == "update msg"
    assert kwargs["ignore_patterns"] == ["*.tmp", "temp/"]


def test_competitions_settings_get_succeeds(parser):

    func, kwargs = parser.dispatch(["competitions", "settings", "get", "my-comp", "--json"])
    assert func.__name__ == "competition_get_settings_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["json_output"] is True


def test_competitions_settings_update_missing_file_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "settings", "update", "my-comp"])


def test_competitions_settings_update_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "settings", "update", "my-comp", "-f", "settings.json", "--json"])
    assert func.__name__ == "competition_update_settings_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["file_path"] == "settings.json"
    assert kwargs["json_output"] is True


def test_competitions_solution_create_missing_path_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "solution", "create", "my-comp"])


def test_competitions_solution_create_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "solution", "create", "my-comp", "-p", "/tmp/sol.csv", "-q"])
    assert func.__name__ == "competition_create_solution_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["path"] == "/tmp/sol.csv"
    assert kwargs["quiet"] is True


def test_competitions_solution_create_with_dash_c_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "solution", "create", "-c", "my-comp", "-p", "/tmp/sol.csv"])
    assert func.__name__ == "competition_create_solution_cli"
    assert kwargs["competition"] is None
    assert kwargs["competition_opt"] == "my-comp"
    assert kwargs["path"] == "/tmp/sol.csv"


def test_competitions_solution_status_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "solution", "status", "my-comp", "--json"])
    assert func.__name__ == "competition_get_solution_status_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["json_output"] is True


def test_competitions_solution_status_default_no_json_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "solution", "status", "my-comp"])
    assert func.__name__ == "competition_get_solution_status_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["json_output"] is False


def test_competitions_launch_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "launch", "my-comp"])
    assert func.__name__ == "competition_launch_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs.get("at") is None


def test_competitions_launch_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "launch", "my-comp", "--at", "2027-01-01T00:00:00Z"])
    assert func.__name__ == "competition_launch_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["at"] == "2027-01-01T00:00:00Z"


def test_competitions_init_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "init"])
    assert func.__name__ == "competition_initialize_cli"
    assert kwargs.get("folder") is None


def test_competitions_init_parser_with_folder_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "init", "my-folder"])
    assert func.__name__ == "competition_initialize_cli"
    assert kwargs["folder"] == "my-folder"


def test_competitions_create_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "create"])
    assert func.__name__ == "competition_create_new_cli"
    assert kwargs.get("folder") is None


def test_competitions_create_parser_with_path_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "create", "-p", "my-path"])
    assert func.__name__ == "competition_create_new_cli"
    assert kwargs["folder"] == "my-path"


def test_competitions_topics_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "topics"])
    assert func.__name__ == "competition_list_topics_cli"
    assert kwargs.get("competition") is None
    assert kwargs.get("sort_by") is None


def test_competitions_topics_list_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "topics", "list", "my-comp", "--sort-by", "new"])
    assert func.__name__ == "competition_list_topics_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["sort_by"] == "new"


def test_competitions_topics_show_missing_topic_ref_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["competitions", "topics", "show"])


def test_competitions_topics_show_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "topics", "show", "topic-ref-url"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "topic-ref-url"
    assert kwargs.get("topic_id_arg") is None


def test_competitions_topics_show_two_args_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "topics", "show", "my-comp", "12345"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "my-comp"
    assert kwargs["topic_id_arg"] == 12345


def test_competitions_topic_messages_parser_default_succeeds(parser):
    func, kwargs = parser.dispatch(["competitions", "topic-messages"])
    assert func.__name__ == "competition_list_topic_messages_cli"
    assert kwargs.get("competition") is None
    assert kwargs.get("topic_id") is None
    assert kwargs.get("sort_by") is None
    assert kwargs.get("page_size") is None


def test_competitions_topic_messages_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "competitions",
            "topic-messages",
            "my-comp",
            "123",
            "--sort-by",
            "top",
            "--page-size",
            "10",
            "--csv",
        ]
    )
    assert func.__name__ == "competition_list_topic_messages_cli"
    assert kwargs["competition"] == "my-comp"
    assert kwargs["topic_id"] == 123
    assert kwargs["sort_by"] == "top"
    assert kwargs["page_size"] == 10
    assert kwargs["csv_display"] is True
