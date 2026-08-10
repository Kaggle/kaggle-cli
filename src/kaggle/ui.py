"""Optional `rich`-backed table rendering for the ``rich`` output format.

``rich`` is an optional dependency (``pip install kaggle[rich]``). This module is
only imported when the user explicitly opts in via ``--format rich`` or
``KAGGLE_OUTPUT_FORMAT=rich``, so the default output paths are untouched.
"""

from typing import Any, Callable, List, Literal, Optional

INSTALL_HINT = "The 'rich' output format requires the 'rich' package. Install it with: pip install kaggle[rich]"

# Fields that are right-justified even when their values are not numeric, to
# match the alignment of the default table renderer.
_RIGHT_JUSTIFIED_FIELDS = frozenset({"size", "reward"})


def is_available() -> bool:
    """Returns whether the optional `rich` dependency is importable."""
    try:
        import rich  # noqa: F401
    except ImportError:
        return False
    return True


def _justify(items: List[Any], field: str, value_getter: Callable[[Any, str], Any]) -> Literal["left", "right"]:
    """Picks the column alignment for a field, mirroring `KaggleApi.print_table`."""
    if field in _RIGHT_JUSTIFIED_FIELDS:
        return "right"
    if items and isinstance(value_getter(items[0], field), int):
        return "right"
    return "left"


def print_rich_table(
    items: List[Any],
    fields: List[str],
    labels: Optional[List[str]] = None,
    value_getter: Callable[[Any, str], Any] = getattr,
) -> None:
    """Prints a table of items using `rich`.

    Args:
        items: A list of items to print.
        fields: A list of fields to select from the items.
        labels: The labels for the fields (defaults to fields).
        value_getter: Callable resolving an item and a field name to a value.
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    if labels is None:
        labels = fields

    table = Table(box=box.ROUNDED, header_style="bold cyan", border_style="dim")
    for field, label in zip(fields, labels):
        # `overflow="fold"` wraps long values instead of truncating them, so no
        # data is silently dropped on narrow terminals.
        table.add_column(label, justify=_justify(items, field, value_getter), overflow="fold")

    for item in items:
        # Values are wrapped in `Text` so that content coming back from the API
        # is never interpreted as rich console markup.
        table.add_row(*(Text(str(value_getter(item, field))) for field in fields))

    Console().print(table)
