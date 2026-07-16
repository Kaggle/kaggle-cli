"""Shared test configuration for kaggle CLI tests.

Must be set before importing kaggle, which calls api.authenticate() at
module level.  Fake legacy credentials keep authenticate() off the network;
removing KAGGLE_API_TOKEN prevents _introspect_token() from being called.
We also patch get_access_token_from_env so the ~/.kaggle/access_token file
doesn't trigger token introspection.
"""

import os
from unittest.mock import patch

os.environ.pop("KAGGLE_API_TOKEN", None)
os.environ["KAGGLE_USERNAME"] = "testuser"
os.environ["KAGGLE_KEY"] = "testkey"

with patch("kagglesdk.get_access_token_from_env", return_value=(None, None)):
    import kaggle  # noqa: F401 — triggers authenticate()


import argparse
from unittest.mock import MagicMock
import pytest
from kaggle.api.kaggle_api_extended import KaggleApi


@pytest.fixture
def api():
    a = KaggleApi()
    a.authenticate = MagicMock()
    a.build_kaggle_client = MagicMock()
    return a


class KaggleParser:
    def __init__(self, parser):
        self._parser = parser

    def dispatch(self, argv):
        args = self._parser.parse_args(argv)
        command_args = dict(vars(args))
        del command_args["func"]
        del command_args["command"]
        return args.func, command_args


@pytest.fixture
def parser(monkeypatch, api):
    import kaggle

    monkeypatch.setattr(kaggle, "api", api)

    from kaggle.cli import (
        parse_quota,
        parse_config,
        parse_auth,
        parse_files,
        parse_competitions,
        parse_datasets,
        parse_kernels,
        parse_models,
        parse_forums,
        parse_benchmarks,
        Help,
    )
    import kaggle.cli

    monkeypatch.setattr(kaggle.cli, "api", api)

    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(title="commands", dest="command")
    subparsers.required = True
    subparsers.choices = Help.kaggle_choices

    parse_quota(subparsers)
    parse_config(subparsers)
    parse_auth(subparsers)
    parse_files(subparsers)
    parse_competitions(subparsers)
    parse_datasets(subparsers)
    parse_kernels(subparsers)
    parse_models(subparsers)
    parse_forums(subparsers)
    parse_benchmarks(subparsers)
    return KaggleParser(root)
