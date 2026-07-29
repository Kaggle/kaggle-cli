import pytest
from unittest.mock import patch, MagicMock
from kaggle.ui import print_rich_table, print_info, print_error, print_success, RICH_AVAILABLE

class DummyItem:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_print_rich_table_no_data():
    with patch("kaggle.ui.console") as mock_console:
        assert print_rich_table([], ["id", "name"]) is True
        mock_console.print.assert_called_once()
        assert "No data found" in str(mock_console.print.call_args)

def test_print_rich_table_with_data():
    items = [DummyItem(id=1, name="Test")]
    with patch("kaggle.ui.console") as mock_console:
        assert print_rich_table(items, ["id", "name"], ["ID", "Name"]) is True
        mock_console.print.assert_called_once()
        table_arg = mock_console.print.call_args[0][0]
        # It should be a Table object
        from rich.table import Table
        assert isinstance(table_arg, Table)
        assert len(table_arg.columns) == 2
        assert len(table_arg.rows) == 1

def test_print_info():
    with patch("kaggle.ui.console") as mock_console:
        print_info("Hello")
        mock_console.print.assert_called_once_with("Hello", style="info")

def test_print_error():
    with patch("kaggle.ui.console") as mock_console:
        print_error("Error msg")
        mock_console.print.assert_called_once_with("[error]Error:[/error] Error msg")

def test_print_success():
    with patch("kaggle.ui.console") as mock_console:
        print_success("Done")
        mock_console.print.assert_called_once_with("✅ [success]Done[/success]")
