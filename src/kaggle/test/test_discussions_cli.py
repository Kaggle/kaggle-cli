"""Tests for discussion topics CLI parsing across all entity types.

Covers the argparse wiring for:
  - ``kaggle <entity> topics [options]``
  - ``kaggle <entity> topics show <topic-ref> [options]``

where <entity> is one of: competitions, datasets, models, benchmarks, forums.

These tests verify that:
  1. Arg parsing dispatches to the correct API method.
  2. The ``topics show`` subcommand does not crash from leaked parent-parser
     kwargs (the original bug).
  3. Each entity's ``topics`` parser wires the expected default func.
"""

import argparse
from unittest.mock import MagicMock

import pytest

from kaggle.api.kaggle_api_extended import KaggleApi

# ---- Fixtures ----


@pytest.fixture
def api():
    """A KaggleApi with mocked auth and client — no network calls."""
    a = KaggleApi()
    a.authenticate = MagicMock()
    mock_client = MagicMock()
    a.build_kaggle_client = MagicMock()
    a.build_kaggle_client.return_value.__enter__.return_value = mock_client
    a._mock_client = mock_client
    return a


@pytest.fixture
def parser(monkeypatch, api):
    """Build the full argument parser tree for testing argparse dispatch."""
    import kaggle

    monkeypatch.setattr(kaggle, "api", api)

    from kaggle.cli import (
        Help,
        parse_benchmarks,
        parse_competitions,
        parse_datasets,
        parse_forums,
        parse_models,
    )
    import kaggle.cli

    monkeypatch.setattr(kaggle.cli, "api", api)

    root = argparse.ArgumentParser()
    root.add_argument("-W", "--no-warn", dest="disable_version_warning", action="store_true")
    subparsers = root.add_subparsers(title="commands", dest="command")
    subparsers.required = True
    subparsers.choices = Help.kaggle_choices

    parse_competitions(subparsers)
    parse_datasets(subparsers)
    parse_models(subparsers)
    parse_benchmarks(subparsers)
    parse_forums(subparsers)
    return root


def _dispatch(parser, argv):
    """Parse *argv* and return (func, command_args) like main() does."""
    args = parser.parse_args(argv)
    command_args = dict(vars(args))
    del command_args["func"]
    del command_args["command"]
    del command_args["disable_version_warning"]
    return args.func, command_args


# ============================================================
# Arg parsing: topics (default func) — correct dispatch
# ============================================================


class TestTopicsListParsing:
    """Verify that ``<entity> topics`` (no subcommand) dispatches to the
    correct list-topics CLI method.

    Note: argparse subparser choices restrict the first positional arg
    to 'show', so the entity ref must be omitted or passed via the
    parent parser's positional (for competitions) to test dispatch.
    """

    def test_competitions_topics_defaults_to_list(self, parser):
        func, kwargs = _dispatch(parser, ["competitions", "topics"])
        assert func.__name__ == "competition_list_topics_cli"

    def test_datasets_topics_defaults_to_list(self, parser):
        func, kwargs = _dispatch(parser, ["datasets", "topics"])
        assert func.__name__ == "dataset_list_topics_cli"

    def test_models_topics_defaults_to_list(self, parser):
        func, kwargs = _dispatch(parser, ["models", "topics"])
        assert func.__name__ == "model_list_topics_cli"

    def test_benchmarks_topics_defaults_to_list(self, parser):
        func, kwargs = _dispatch(parser, ["benchmarks", "topics"])
        assert func.__name__ == "benchmark_list_topics_cli"

    def test_forums_topics_defaults_to_list(self, parser):
        func, kwargs = _dispatch(parser, ["forums", "topics"])
        assert func.__name__ == "forums_list_topics_cli"

    @pytest.mark.parametrize(
        "entity, extra_args",
        [
            ("datasets", ["list", "--sort-by", "hot", "-s", "keyword"]),
            ("models", ["--page-size", "50"]),
            ("benchmarks", ["--page-token", "abc"]),
            ("competitions", ["list", "--page", "5", "--sort-by", "recent"]),
        ],
        ids=["datasets_sort_search", "models_page_size", "benchmarks_page_token", "competitions_page_sort"],
    )
    def test_topics_list_with_options(self, parser, entity, extra_args):
        """Optional flags are parsed without error for entity topics."""
        func, kwargs = _dispatch(parser, [entity, "topics"] + extra_args)
        # Should dispatch to the list func, not show.
        assert "show" not in func.__name__

    def test_competitions_topics_list_does_not_accept_search(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["competitions", "topics", "list", "--search", "query"])


# ============================================================
# Arg parsing: topics show — correct func + kwargs
# ============================================================


class TestTopicsShowParsing:
    """Verify that ``<entity> topics show <ref>`` parses correctly
    and dispatches to forums_topic_show_cli.
    """

    @pytest.mark.parametrize(
        "entity",
        ["competitions", "datasets", "models", "benchmarks", "forums"],
    )
    def test_topics_show_dispatches_to_forums_topic_show_cli(self, parser, entity):
        """All entities dispatch 'topics show' to forums_topic_show_cli."""
        func, kwargs = _dispatch(parser, [entity, "topics", "show", "12345"])
        assert func.__name__ == "forums_topic_show_cli"
        assert kwargs["topic_ref"] == "12345"

    @pytest.mark.parametrize(
        "entity",
        ["competitions", "datasets", "models", "benchmarks", "forums"],
    )
    def test_topics_show_two_arg_form(self, parser, entity):
        """Two-arg form: <entity> topics show <ref> <topic-id>."""
        func, kwargs = _dispatch(parser, [entity, "topics", "show", "some-ref", "99"])
        assert func.__name__ == "forums_topic_show_cli"
        assert kwargs["topic_ref"] == "some-ref"
        assert kwargs["topic_id_arg"] == 99

    @pytest.mark.parametrize(
        "entity",
        ["competitions", "datasets", "models", "benchmarks", "forums"],
    )
    def test_topics_show_with_options(self, parser, entity):
        """Optional flags (--page-size, --csv, --quiet) parse correctly."""
        func, kwargs = _dispatch(
            parser,
            [entity, "topics", "show", "123", "--page-size", "10", "-v", "-q"],
        )
        assert func.__name__ == "forums_topic_show_cli"
        assert kwargs["page_size"] == 10
        assert kwargs["csv_display"] is True
        assert kwargs["quiet"] is True


# ============================================================
# Regression: topics show must not crash from parent kwargs
# ============================================================


class TestTopicsShowNoCrash:
    """Regression tests for the TypeError bugs.

    The original bug: argparse leaks parent-parser kwargs into the
    ``show`` subcommand handler, causing:
      TypeError: got an unexpected keyword argument 'sort_by'
      TypeError: got an unexpected keyword argument 'competition'
    """

    @pytest.mark.parametrize(
        "entity",
        ["competitions", "datasets", "models", "benchmarks", "forums"],
    )
    def test_topics_show_callable_with_leaked_kwargs(self, parser, api, entity):
        """Calling func(**command_args) must not raise TypeError.

        This simulates what main() does: it calls args.func(**command_args)
        with all the argparse namespace vars — including those from the
        parent 'topics' parser.
        """
        func, kwargs = _dispatch(parser, [entity, "topics", "show", "12345"])

        # Mock the underlying method so we don't hit the network.
        mock_topic = MagicMock()
        mock_topic.content = None  # Avoid bleach.clean on MagicMock
        api.forums_topic_show = MagicMock(return_value=(mock_topic, [], ""))

        # Bind the unbound function to the api instance, just like main() does.
        bound = func.__get__(api, type(api))

        # Should not raise TypeError from leaked parent-parser kwargs.
        try:
            bound(**kwargs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"TypeError from leaked kwargs: {e}")
            if "multiple values for argument" in str(e):
                pytest.fail(f"TypeError from duplicate kwargs: {e}")
            raise


# ============================================================
# API method: entity-specific list topics CLI methods
# ============================================================


class TestEntityListTopicsCli:
    """Verify the entity-specific list topics CLI methods."""

    def test_dataset_list_topics_cli_requires_ref(self, api):
        with pytest.raises(ValueError, match="No dataset specified"):
            api.dataset_list_topics_cli()

    def test_model_list_topics_cli_requires_ref(self, api):
        with pytest.raises(ValueError, match="No model specified"):
            api.model_list_topics_cli()

    def test_benchmark_list_topics_cli_requires_ref(self, api):
        with pytest.raises(ValueError, match="No benchmark specified"):
            api.benchmark_list_topics_cli()

    @pytest.mark.parametrize(
        "method_name",
        ["dataset_list_topics_cli", "model_list_topics_cli", "benchmark_list_topics_cli"],
    )
    def test_list_topics_prints_no_topics_found(self, api, capsys, method_name):
        """When no topics are returned, prints 'No topics found'."""
        mock_response = MagicMock()
        mock_response.topics = []
        mock_response.next_page_token = ""
        api.forums_list_topics = MagicMock(return_value=mock_response)

        method = getattr(api, method_name)
        method(entity_ref="test-ref")

        assert "No topics found" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "method_name",
        ["dataset_list_topics_cli", "model_list_topics_cli", "benchmark_list_topics_cli"],
    )
    def test_list_topics_shows_table(self, api, capsys, method_name):
        """When topics are returned, prints a table."""
        mock_topic = MagicMock()
        mock_response = MagicMock()
        mock_response.topics = [mock_topic]
        mock_response.next_page_token = ""
        api.forums_list_topics = MagicMock(return_value=mock_response)
        api.print_table = MagicMock()

        method = getattr(api, method_name)
        method(entity_ref="test-ref")

        api.print_table.assert_called_once()

    @pytest.mark.parametrize(
        "method_name",
        ["dataset_list_topics_cli", "model_list_topics_cli", "benchmark_list_topics_cli"],
    )
    def test_list_topics_passes_all_params(self, api, method_name):
        """All params are forwarded to forums_list_topics."""
        mock_response = MagicMock()
        mock_response.topics = []
        mock_response.next_page_token = ""
        api.forums_list_topics = MagicMock(return_value=mock_response)

        method = getattr(api, method_name)
        method(
            entity_ref="test-ref",
            sort_by="hot",
            page_size=50,
            page_token="tok",
            search="query",
        )

        api.forums_list_topics.assert_called_once_with(
            forum_slug="test-ref",
            sort_by="hot",
            page_size=50,
            page_token="tok",
            search="query",
        )


# ============================================================
# API method: forums_topic_show_cli accepts **kwargs
# ============================================================


class TestForumsTopicShowCliKwargs:
    """Verify forums_topic_show_cli absorbs extra kwargs."""

    def test_accepts_extra_kwargs_without_error(self, api):
        """Extra kwargs (from parent parser) should not cause TypeError."""
        mock_topic = MagicMock()
        mock_topic.content = None  # Avoid bleach.clean on MagicMock
        api.forums_topic_show = MagicMock(return_value=(mock_topic, [], ""))

        # These are the kind of extra kwargs that leak from parent parsers.
        api.forums_topic_show_cli(
            topic_ref="12345",
            sort_by="hot",
            search="test",
            entity_ref="some-entity",
            forum="getting-started",
            category="all",
            group="all",
            competition="titanic",
            competition_opt=None,
        )
        # If we get here without TypeError, the test passes.
        api.forums_topic_show.assert_called_once()

    def test_topic_id_parsing_bare_id(self, api):
        """Bare topic ID is parsed correctly."""
        mock_topic = MagicMock()
        mock_topic.content = None
        api.forums_topic_show = MagicMock(return_value=(mock_topic, [], ""))

        api.forums_topic_show_cli(topic_ref="12345")

        api.forums_topic_show.assert_called_once_with(12345, page_size=None, page_token=None)

    def test_topic_id_parsing_slash_form(self, api):
        """Slash form 'forum-slug/12345' extracts the topic ID."""
        mock_topic = MagicMock()
        mock_topic.content = None
        api.forums_topic_show = MagicMock(return_value=(mock_topic, [], ""))

        api.forums_topic_show_cli(topic_ref="getting-started/12345")

        api.forums_topic_show.assert_called_once_with(12345, page_size=None, page_token=None)

    def test_topic_id_multi_slash_parses_correctly(self, api):
        """Multi-slash ref like 'google/gemma-4/1' extracts the topic ID."""
        mock_topic = MagicMock()
        mock_topic.content = None
        api.forums_topic_show = MagicMock(return_value=(mock_topic, [], ""))

        api.forums_topic_show_cli(topic_ref="google/gemma-4/1")

        api.forums_topic_show.assert_called_once_with(1, page_size=None, page_token=None)

    def test_topic_id_parsing_two_arg_form(self, api):
        """Two-arg form uses topic_id_arg."""
        mock_topic = MagicMock()
        mock_topic.content = None
        api.forums_topic_show = MagicMock(return_value=(mock_topic, [], ""))

        api.forums_topic_show_cli(topic_ref="getting-started", topic_id_arg=99)

        api.forums_topic_show.assert_called_once_with(99, page_size=None, page_token=None)

    def test_no_topic_raises_value_error(self, api):
        """Missing topic_ref raises ValueError."""
        with pytest.raises(ValueError, match="No topic specified"):
            api.forums_topic_show_cli(topic_ref=None)

    def test_invalid_slash_ref_raises_value_error(self, api):
        """Non-numeric part after '/' raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="must be a numeric topic ID"):
            api.forums_topic_show_cli(topic_ref="google/gemma-4")

    def test_invalid_bare_ref_raises_value_error(self, api):
        """Non-numeric bare ref raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Expected a numeric topic ID"):
            api.forums_topic_show_cli(topic_ref="not-a-number")


# ============================================================
# Arg parsing: topics list — correct func + kwargs
# ============================================================


class TestTopicsListSubcommand:
    """Verify that ``<entity> topics list [ref]`` parses correctly
    and dispatches to the correct list-topics CLI method.
    """

    @pytest.mark.parametrize(
        "entity, expected_func",
        [
            ("competitions", "competition_list_topics_cli"),
            ("datasets", "dataset_list_topics_cli"),
            ("models", "model_list_topics_cli"),
            ("benchmarks", "benchmark_list_topics_cli"),
            ("forums", "forums_list_topics_cli"),
        ],
    )
    def test_topics_list_subcommand_dispatches_correctly(self, parser, entity, expected_func):
        """All entities dispatch 'topics list' to their respective list func."""
        func, kwargs = _dispatch(parser, [entity, "topics", "list"])
        assert func.__name__ == expected_func

    @pytest.mark.parametrize(
        "entity, ref_key",
        [
            ("competitions", "competition"),
            ("datasets", "entity_ref"),
            ("models", "entity_ref"),
            ("benchmarks", "entity_ref"),
            ("forums", "forum"),
        ],
    )
    def test_topics_list_subcommand_with_ref(self, parser, entity, ref_key):
        """Position ref is parsed correctly inside 'list' subcommand."""
        func, kwargs = _dispatch(parser, [entity, "topics", "list", "my-special-ref"])
        assert kwargs[ref_key] == "my-special-ref"

    def test_competitions_topics_list_with_c_option(self, parser):
        """Competitions topics list supports -c option."""
        func, kwargs = _dispatch(parser, ["competitions", "topics", "list", "-c", "titanic"])
        assert func.__name__ == "competition_list_topics_cli"
        assert kwargs["competition_opt"] == "titanic"
