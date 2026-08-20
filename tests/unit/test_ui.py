# coding=utf-8
import io
import unittest
from unittest.mock import patch

import pytest

from kaggle import ui
from kaggle.api.kaggle_api_extended import KaggleApi, OutputFormat

rich_console = pytest.importorskip("rich.console")


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    return api


class Item:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _render(items, fields, labels=None, width=200):
    """Renders a rich table to a string using a fixed-width console."""
    console = rich_console.Console(file=io.StringIO(), width=width, force_terminal=False)
    with patch.object(rich_console, "Console", return_value=console):
        ui.print_rich_table(items, fields, labels)
    return console.file.getvalue()


class TestPrintRichTable(unittest.TestCase):
    def test_renders_values_and_labels(self):
        output = _render([Item(ref="titanic", team_count=15000)], ["ref", "team_count"], ["ref", "teamCount"])

        self.assertIn("teamCount", output)
        self.assertIn("titanic", output)
        self.assertIn("15000", output)

    def test_renders_markup_in_values_literally(self):
        """Values from the API must never be parsed as rich console markup."""
        output = _render([Item(ref="a[bold]b[/bold]c")], ["ref"])

        self.assertIn("a[bold]b[/bold]c", output)

    def test_folds_long_values_instead_of_truncating(self):
        """Narrow terminals must wrap values so that no data is lost."""
        value = "x" * 60
        output = _render([Item(ref=value)], ["ref"], width=30)

        self.assertNotIn("…", output)
        self.assertEqual(value, "".join(c for c in output if c == "x"))

    def test_int_and_size_columns_are_right_justified(self):
        output = _render([Item(name="a", votes=1, size="2KB")], ["name", "votes", "size"], width=40)
        # The header row is the first line containing the column labels.
        header = next(line for line in output.splitlines() if "votes" in line)
        cells = [cell for cell in header.split("│") if cell.strip()]

        self.assertTrue(cells[0].startswith(" name"), header)
        self.assertTrue(cells[1].endswith("votes "), header)
        self.assertTrue(cells[2].endswith("size "), header)

    def test_uses_custom_value_getter(self):
        console = rich_console.Console(file=io.StringIO(), width=80, force_terminal=False)
        with patch.object(rich_console, "Console", return_value=console):
            ui.print_rich_table([{"ref": "abc"}], ["ref"], value_getter=lambda i, f: i[f])

        self.assertIn("abc", console.file.getvalue())


class TestRichOutputFormat(unittest.TestCase):
    def setUp(self):
        self.api = _make_api()

    def test_rich_is_a_supported_output_format(self):
        self.assertEqual(OutputFormat.RICH, self.api._get_output_format(False, "rich"))

    def test_rich_format_supports_projection(self):
        self.assertEqual(OutputFormat.RICH, self.api._get_output_format(False, "rich(ref)"))

    def test_csv_flag_takes_precedence_over_rich(self):
        self.assertEqual(OutputFormat.CSV, self.api._get_output_format(True, None))

    def test_default_is_table_when_no_format_requested(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(OutputFormat.TABLE, self.api._get_output_format(False, None))

    def test_env_var_sets_the_default_format(self):
        with patch.dict("os.environ", {"KAGGLE_OUTPUT_FORMAT": "rich"}):
            self.assertEqual(OutputFormat.RICH, self.api._get_output_format(False, None))

    def test_env_var_is_ignored_when_format_is_explicit(self):
        with patch.dict("os.environ", {"KAGGLE_OUTPUT_FORMAT": "rich"}):
            self.assertEqual(OutputFormat.JSON, self.api._get_output_format(False, "json"))

    def test_unknown_env_var_value_falls_back_to_table(self):
        with patch.dict("os.environ", {"KAGGLE_OUTPUT_FORMAT": "nonsense"}):
            self.assertEqual(OutputFormat.TABLE, self.api._get_output_format(False, None))

    def test_print_results_routes_rich_format_to_print_rich(self):
        items = [Item(ref="titanic")]
        with patch.object(KaggleApi, "print_rich") as mock_print_rich:
            self.api.print_results(items, ["ref"], output_format="rich")

        mock_print_rich.assert_called_once_with(items, ["ref"], ["ref"])

    def test_print_results_still_defaults_to_print_table(self):
        items = [Item(ref="titanic")]
        with patch.dict("os.environ", {}, clear=True), patch.object(KaggleApi, "print_table") as mock_print_table:
            self.api.print_results(items, ["ref"])

        mock_print_table.assert_called_once_with(items, ["ref"], ["ref"])


class TestPrintRich(unittest.TestCase):
    def setUp(self):
        self.api = _make_api()

    def test_maps_camel_case_fields_to_snake_case_attributes(self):
        items = [Item(team_count=42)]
        with patch.object(ui, "print_rich_table") as mock_table:
            self.api.print_rich(items, ["teamCount"], ["teamCount"])

        _, _, _ = mock_table.call_args[0]
        value_getter = mock_table.call_args[1]["value_getter"]
        self.assertEqual(42, value_getter(items[0], "teamCount"))

    def test_prints_nothing_for_empty_items(self):
        """Matches print_table, which prints nothing when there are no items."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.api.print_rich([], ["ref"])

        self.assertEqual("", mock_stdout.getvalue())

    def test_falls_back_to_table_when_rich_is_missing(self):
        items = [Item(ref="titanic")]
        with patch.object(ui, "is_available", return_value=False):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                with patch.object(KaggleApi, "print_table") as mock_print_table:
                    self.api.print_rich(items, ["ref"], ["ref"])

        self.assertIn("pip install kaggle[rich]", mock_stderr.getvalue())
        mock_print_table.assert_called_once_with(items, ["ref"], ["ref"])
