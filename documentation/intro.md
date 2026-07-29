# Kaggle CLI TUI (Terminal User Interface)

Kaggle CLI now ships with a built-in modern Terminal User Interface (TUI) powered by [`rich`](https://github.com/Textualize/rich). 

## Visual Enhancements

When you run list commands (e.g., `kaggle datasets list`, `kaggle competitions list`, etc.) in an interactive terminal, the CLI will format the output as a gorgeous table with:
- **Rounded Borders:** Clean and modern look.
- **Color Coding:** Headers in bright cyan, borders in bright black.
- **Smart Alignment:** Numeric columns (like sizes, download counts, and vote counts) are automatically right-aligned for easier scanning.
- **Graceful Text Wrapping:** Content that overflows is neatly wrapped.

## Automatic Fallback (No-breaking changes)

The new UI is completely **safe for automation scripts**. The CLI automatically detects if it is being piped into another command (e.g., `kaggle datasets list > data.txt` or `kaggle datasets list | grep ...`). 

If standard output is not a TTY terminal, or if the `rich` library is not available, the CLI will seamlessly and silently fall back to the raw, uncolored string formatting. This guarantees 100% backward compatibility for all your existing scripts.

## Supported Commands
The modern UI affects the output of all list-oriented commands, including:
- `kaggle competitions list`
- `kaggle datasets list`
- `kaggle kernels list`
- `kaggle models list`
- `kaggle models instances`
- `kaggle config view`
