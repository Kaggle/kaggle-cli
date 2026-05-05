# Kaggle Hackathons CLI Reference

This reference covers `kaggle hackathons …` (alias `kaggle h …`), which
provides CLI access to the hackathon-specific endpoints exposed by Kaggle's
MCP server.

## Prerequisites

- Python 3.11+
- `kaggle` CLI installed (`pip install kaggle` or `pip install -e .` from source)
- Valid Kaggle credentials: `KAGGLE_API_TOKEN` env var, `~/.kaggle/access_token`, or OAuth via `kaggle auth login`

## Command Hierarchy

```
kaggle hackathons (alias: kaggle h)
├── get <competition>                                  — Show overview pages for a hackathon
└── writeups
    ├── list <competition>                             — List submitted write-ups
    ├── download <competition> [-p <path>]             — Download all write-ups as CSV
    └── resolve-links <writeup_id>                     — Resolve datasets/notebooks/links inside a write-up
```

Each command maps 1:1 to an MCP tool exposed by Kaggle's `McpClient`.

| CLI                                                    | MCP tool                       |
| ------------------------------------------------------ | ------------------------------ |
| `kaggle h get <competition>`                           | `get_hackathon_overview`       |
| `kaggle h writeups list <competition>`                 | `list_hackathon_write_ups`     |
| `kaggle h writeups download <competition> [-p <path>]` | `download_hackathon_write_ups` |
| `kaggle h writeups resolve-links <writeup_id>`         | `get_resolved_writeup_links`   |

## Examples

### Show the overview of a hackathon

```bash
kaggle hackathons get my-hackathon
```

By default the overview pages are printed as plain text (HTML stripped).
Use `-v / --csv` to emit a CSV table of `name`-keyed pages instead.

### List all write-up submissions

```bash
kaggle h writeups list my-hackathon
```

Lists each submitted write-up with its `id`, owning team, title, URL,
competition id, and `template` flag. Add `-v` for CSV output, `-q` to
suppress the trailing total/next-page-token lines.

### Download a CSV of all write-ups

```bash
# Default destination: ./<competition>-writeups.csv
kaggle h writeups download my-hackathon

# Custom destination file:
kaggle h writeups download my-hackathon -p out/writeups.csv

# Custom destination directory (filename appended automatically):
kaggle h writeups download my-hackathon -p exports/
```

This command calls the host-only `ExportHackathonWriteUpsCsv` RPC and
writes the CSV body to disk. It only works after the hackathon has
closed, and only for the competition's host(s) and judges.

### Resolve the links inside a write-up

```bash
kaggle h writeups resolve-links 12345
```

Returns metadata (download URLs, file summaries, thumbnails) for every
link embedded inside the write-up — datasets, notebooks, YouTube videos,
external pages, etc. Use this to follow a write-up's references without
manually scraping the rendered page.

## Output formats

All `hackathons` commands default to a human-friendly table for list output
and plain text for the overview. Pass `-v / --csv` to switch list outputs
to CSV. The `download` subcommand always writes CSV (that's the whole
point) and prints a one-line confirmation unless `-q / --quiet` is set.
