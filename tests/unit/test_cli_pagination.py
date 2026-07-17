# coding=utf-8
import argparse
from unittest.mock import MagicMock, patch
import pytest
from kaggle.api.kaggle_api_extended import KaggleApi


@pytest.fixture
def api():
    a = KaggleApi()
    a.authenticate = MagicMock()  # type: ignore[method-assign]
    # Mock build_kaggle_client to avoid any real client instantiation
    a.build_kaggle_client = MagicMock()  # type: ignore[method-assign]
    return a


@pytest.fixture
def parser(monkeypatch, api):
    import kaggle

    monkeypatch.setattr(kaggle, "api", api)
    from kaggle.cli import (
        parse_competitions,
        parse_datasets,
        parse_kernels,
        Help,
    )
    import kaggle.cli

    monkeypatch.setattr(kaggle.cli, "api", api)

    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(title="commands", dest="command")
    subparsers.required = True
    subparsers.choices = Help.kaggle_choices  # type: ignore[assignment]

    parse_competitions(subparsers)
    parse_datasets(subparsers)
    parse_kernels(subparsers)
    return root


def _dispatch(parser, argv):
    args = parser.parse_args(argv)
    command_args = dict(vars(args))
    del command_args["func"]
    del command_args["command"]
    return args.func, command_args


def test_competitions_list_pagination(parser):
    # Should accept --page-token and --page-size
    func, kwargs = _dispatch(parser, ["competitions", "list", "--page-token", "token123", "--page-size", "50"])
    assert func.__name__ == "competitions_list_cli"
    assert kwargs["page_token"] == "token123"
    assert kwargs["page_size"] == 50

    # Should also accept --page (backward compatibility)
    func, kwargs = _dispatch(parser, ["competitions", "list", "--page", "5"])
    assert func.__name__ == "competitions_list_cli"
    assert kwargs["page"] == 5


def test_datasets_list_pagination(parser):
    # Should accept --page-token and --page-size
    func, kwargs = _dispatch(parser, ["datasets", "list", "--page-token", "token123", "--page-size", "50"])
    assert func.__name__ == "dataset_list_cli"
    assert kwargs["page_token"] == "token123"
    assert kwargs["page_size"] == 50

    # Should also accept --page (backward compatibility)
    func, kwargs = _dispatch(parser, ["datasets", "list", "--page", "5"])
    assert func.__name__ == "dataset_list_cli"
    assert kwargs["page"] == 5


def test_kernels_list_pagination(parser):
    # Should accept --page-token and --page-size
    func, kwargs = _dispatch(parser, ["kernels", "list", "--page-token", "token123", "--page-size", "50"])
    assert func.__name__ == "kernels_list_cli"
    assert kwargs["page_token"] == "token123"
    assert kwargs["page_size"] == 50

    # Should also accept --page (backward compatibility)
    func, kwargs = _dispatch(parser, ["kernels", "list", "--page", "5"])
    assert func.__name__ == "kernels_list_cli"
    assert kwargs["page"] == 5


# ---- Validation Tests ----


@patch.object(KaggleApi, "competitions_list")
def test_competitions_list_mutual_exclusion(mock_list, api):
    with pytest.raises(ValueError, match="Cannot specify both page and page_token"):
        api.competitions_list_cli(page=5, page_token="token")


@patch.object(KaggleApi, "dataset_list_with_response")
def test_datasets_list_mutual_exclusion(mock_list, api):
    with pytest.raises(ValueError, match="Cannot specify both page and page_token"):
        api.dataset_list_cli(page=5, page_token="token")


@patch.object(KaggleApi, "kernels_list_with_response")
def test_kernels_list_mutual_exclusion(mock_list, api):
    with pytest.raises(ValueError, match="Cannot specify both page and page_token"):
        api.kernels_list_cli(page=5, page_token="token")


@patch.object(KaggleApi, "competition_list_topics")
def test_competition_topics_mutual_exclusion(mock_list, api):
    # Mock get_config_value to avoid config lookup issues
    api.get_config_value = MagicMock(return_value="some-competition")
    with pytest.raises(ValueError, match="Cannot specify both page and page_token"):
        api.competition_list_topics_cli(page=5, page_token="token")
