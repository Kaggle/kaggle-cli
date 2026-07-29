import sys
from typing import List, Any, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.theme import Theme
    from rich.text import Text
    
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green"
    })
    console = Console(theme=custom_theme)
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

def print_rich_table(items: List[Any], fields: List[str], labels: Optional[List[str]] = None, string_formatter=None, attr_getter=None):
    if not RICH_AVAILABLE:
        return False
        
    if labels is None:
        labels = fields
    
    if not items:
        console.print(Panel("No data found.", style="warning", expand=False))
        return True

    from rich import box
    table = Table(show_header=True, header_style="bold bright_cyan", border_style="bright_black", box=box.ROUNDED)
    
    for i, label in enumerate(labels):
        field = fields[i].lower()
        justify = "right" if field in ["size", "reward", "id", "downloadcount", "votecount"] else "left"
        table.add_column(label, justify=justify)
        
    for item in items:
        row = []
        for field in fields:
            val = attr_getter(item, field) if attr_getter else getattr(item, field)
            val_str = string_formatter(val) if string_formatter else str(val)
            row.append(val_str)
        table.add_row(*row)
        
    console.print(table)
    return True

def print_info(message: str):
    if RICH_AVAILABLE:
        console.print(message, style="info")
    else:
        print(message)

def print_error(message: str, exc: Optional[Exception] = None):
    if RICH_AVAILABLE:
        if exc:
            console.print(f"[error]Error:[/error] {message}\n[dim]{str(exc)}[/dim]")
        else:
            console.print(f"[error]Error:[/error] {message}")
    else:
        print(f"Error: {message}", file=sys.stderr)
        if exc:
            print(str(exc), file=sys.stderr)

def print_success(message: str):
    if RICH_AVAILABLE:
        console.print(f"✅ [success]{message}[/success]")
    else:
        print(message)
