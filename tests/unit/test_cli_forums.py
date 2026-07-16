# coding=utf-8
import pytest


def test_forums_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["forums"])
    assert func.__name__ == "forums_list_cli"
    assert kwargs.get("csv_display") is None
    assert kwargs.get("output_format") is None
    assert kwargs.get("quiet") is None


def test_forums_list_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(["forums", "list", "--csv", "--quiet"])
    assert func.__name__ == "forums_list_cli"
    assert kwargs["csv_display"] is True
    assert kwargs["quiet"] is True


def test_forums_topics_default_list_succeeds(parser):
    func, kwargs = parser.dispatch(["forums", "topics"])
    assert func.__name__ == "forums_list_topics_cli"
    assert kwargs.get("forum") is None
    assert kwargs.get("sort_by") is None
    assert kwargs.get("search") is None
    assert kwargs.get("category") is None
    assert kwargs.get("group") is None


def test_forums_topics_list_parser_with_flags_succeeds(parser):
    func, kwargs = parser.dispatch(
        [
            "forums",
            "topics",
            "list",
            "general",
            "--sort-by",
            "hot",
            "-s",
            "query",
            "--category",
            "q&a",
            "--group",
            "all",
        ]
    )
    assert func.__name__ == "forums_list_topics_cli"
    assert kwargs["forum"] == "general"
    assert kwargs["sort_by"] == "hot"
    assert kwargs["search"] == "query"
    assert kwargs["category"] == "q&a"
    assert kwargs["group"] == "all"


def test_forums_topics_show_parser_missing_topic_ref_fails(parser):
    with pytest.raises(SystemExit):
        parser.dispatch(["forums", "topics", "show"])


def test_forums_topics_show_parser_succeeds(parser):
    func, kwargs = parser.dispatch(["forums", "topics", "show", "topic-ref-url"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "topic-ref-url"
    assert kwargs.get("topic_id_arg") is None


def test_forums_topics_show_parser_two_args_succeeds(parser):
    func, kwargs = parser.dispatch(["forums", "topics", "show", "general", "12345"])
    assert func.__name__ == "forums_topic_show_cli"
    assert kwargs["topic_ref"] == "general"
    assert kwargs["topic_id_arg"] == 12345
