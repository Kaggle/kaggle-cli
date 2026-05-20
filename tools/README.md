# tools/

Developer tooling that lives outside the shipped `kaggle` package.

## `check_mcp_cli_parity.py`

Parity gate that ensures every MCP tool advertised by
`Kaggle.Sdk/mcp/McpClient.cs` (in the private `Kaggle/kaggleazure` repo) has
either a matching `kaggle` CLI command or an explicit "skip" reason. Runs in
CI on every PR; fails loudly when a new MCP tool is added without a CLI
counterpart.

### How it works

1. Loads `McpClient.cs` from a local path (`--mcp-client`) or raw URL
   (`--mcp-client-url`). The local file wins if it exists.
2. Parses every `[McpServerTool(Name = "<name>")]` annotation.
3. Walks the `kaggle` argparse tree (built from `src/kaggle/cli.py`) and
   collects the full set of registered subcommand paths.
4. Loads `tools/mcp_cli_mapping.yaml` and verifies that every MCP tool maps
   either to a real CLI command path or a `skip: <reason>` string.
5. Prints a markdown coverage table and exits non-zero if anything is
   missing or broken.

### Local usage

If you have the `kaggleazure` repo checked out as a sibling directory, the
defaults Just Work:

```bash
python3 tools/check_mcp_cli_parity.py
```

Otherwise, point at a local copy or fetch via URL (the latter requires
`GITHUB_TOKEN` because `Kaggle/kaggleazure` is private):

```bash
# Explicit local path
python3 tools/check_mcp_cli_parity.py \
    --mcp-client /path/to/Kaggle.Sdk/mcp/McpClient.cs

# Fetch over HTTPS (needs a token with `repo` scope)
GITHUB_TOKEN=ghp_xxx python3 tools/check_mcp_cli_parity.py \
    --mcp-client-url https://raw.githubusercontent.com/Kaggle/kaggleazure/ci/Kaggle.Sdk/mcp/McpClient.cs
```

The script has zero third-party dependencies — only the Python standard
library plus whatever `kaggle.cli` already imports.

### Adding a new MCP tool

When `McpClient.cs` gains a new `[McpServerTool(Name = "...")]`, the gate
will fail with a "MISSING" entry until you add a row to
`tools/mcp_cli_mapping.yaml`:

```yaml
# A real CLI command — must match a registered argparse subcommand chain.
my_new_tool: my-group my-subcommand

# Or, if there is intentionally no CLI surface, a skip with a reason.
my_new_tool: "skip: server-side handshake, exposed via `kaggle foo bar`"
```

Reasons matter: empty `skip:` entries are rejected. Use them to capture
why the gap exists (work-in-progress, browser-only flow, internal RPC, etc.)
so future readers don't have to guess.

After editing the mapping, re-run the script locally to confirm it exits 0.

### CI integration

`.github/workflows/mcp-cli-parity.yaml` runs the script on every PR using
`--mcp-client-url` and `secrets.KAGGLEAZURE_READ_TOKEN` so it doesn't need
the private repo checked out.
