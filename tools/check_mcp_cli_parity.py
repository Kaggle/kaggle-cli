#!/usr/bin/env python3
"""MCP ↔ CLI parity gate.

Compares the set of MCP tools advertised by ``Kaggle.Sdk/mcp/McpClient.cs``
against the set of ``kaggle`` CLI commands wired in ``src/kaggle/cli.py``.

Each MCP tool must have an entry in ``tools/mcp_cli_mapping.yaml``. The
entry is either a CLI command path (e.g. ``hackathons writeups download``)
that points to a real subparser, or a ``skip: <reason>`` string for tools
that intentionally have no CLI surface (auth flows, browser-only RPCs, or
tools we have not yet wrapped).

Exits non-zero when:
  * an MCP tool has no mapping entry (NEW unmapped tool — fail loud), OR
  * a mapped tool points to a CLI path that does not exist.

Always prints a markdown coverage table to stdout.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAPPING = REPO_ROOT / "tools" / "mcp_cli_mapping.yaml"
DEFAULT_LOCAL_MCP = REPO_ROOT.parent / "kaggleazure" / "Kaggle.Sdk" / "mcp" / "McpClient.cs"
DEFAULT_REMOTE_MCP = (
    "https://raw.githubusercontent.com/Kaggle/kaggleazure/ci/Kaggle.Sdk/mcp/McpClient.cs"
)

MCP_TOOL_PATTERN = re.compile(r'McpServerTool\(Name\s*=\s*"([^"]+)"')


# ---------------------------------------------------------------------------
# MCP tool extraction
# ---------------------------------------------------------------------------


def fetch_mcp_client_source(local_path: str | None, url: str | None) -> str:
    """Return the McpClient.cs source. Local file wins if present."""
    if local_path:
        p = Path(local_path)
        if p.is_file():
            return p.read_text(encoding="utf-8")
    if url:
        headers = {"User-Agent": "kaggle-cli-parity-check"}
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("KAGGLEAZURE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted URL)
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            hint = ""
            if exc.code in (401, 403, 404) and not token:
                hint = (
                    "\n  Hint: Kaggle/kaggleazure is a private repo. Set GITHUB_TOKEN "
                    "(with `repo` scope) so the script can fetch the raw file."
                )
            raise SystemExit(f"Failed to fetch McpClient.cs from {url}: HTTP {exc.code}{hint}")
        except urllib.error.URLError as exc:
            raise SystemExit(f"Failed to fetch McpClient.cs from {url}: {exc}")
    raise SystemExit(
        "No McpClient.cs source available. Pass --mcp-client <path> or --mcp-client-url <url>."
    )


def extract_mcp_tools(source: str) -> List[str]:
    """Return the sorted, de-duplicated list of MCP tool names."""
    return sorted(set(MCP_TOOL_PATTERN.findall(source)))


# ---------------------------------------------------------------------------
# CLI command extraction
# ---------------------------------------------------------------------------


def collect_cli_commands() -> Set[str]:
    """Build the kaggle argparse tree and return all command paths.

    A command path is the space-joined chain of subparser names from the
    top-level command down to a leaf (e.g. ``"hackathons writeups list"``).
    Aliases are emitted alongside the canonical name so that mappings may
    reference either form.
    """
    # Make ``kaggle`` importable even if this script is run from any cwd.
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # ``kaggle`` calls ``api.authenticate()`` at import time. Pin fake creds
    # and stub out the access-token lookup so the import never hits the network.
    os.environ.pop("KAGGLE_API_TOKEN", None)
    os.environ.setdefault("KAGGLE_USERNAME", "parity-check")
    os.environ.setdefault("KAGGLE_KEY", "parity-check")
    from unittest.mock import patch

    with patch("kagglesdk.get_access_token_from_env", return_value=(None, None)):
        import kaggle.cli as kaggle_cli  # noqa: WPS433 — runtime import is intentional

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    for name in (
        "parse_competitions",
        "parse_datasets",
        "parse_kernels",
        "parse_models",
        "parse_files",
        "parse_forums",
        "parse_hackathons",
        "parse_benchmarks",
        "parse_config",
        "parse_auth",
    ):
        getattr(kaggle_cli, name)(subparsers)

    paths: Set[str] = set()
    _collect_subparser_paths(parser, [], paths)
    return paths


def _collect_subparser_paths(parser, prefix: List[str], paths: Set[str]) -> None:
    """Recurse into argparse subparser actions to collect leaf command paths."""
    for action in parser._actions:  # noqa: SLF001 — argparse exposes no public API
        if not isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            continue
        # ``action._name_parser_map`` always holds the {name: parser} dict
        # even when ``cli.py`` overwrites ``action.choices`` with a list of
        # display strings.
        name_parser_map = action._name_parser_map  # noqa: SLF001
        for name, sub_parser in name_parser_map.items():
            child_prefix = prefix + [name]
            path = " ".join(child_prefix)
            paths.add(path)
            _collect_subparser_paths(sub_parser, child_prefix, paths)


# ---------------------------------------------------------------------------
# Mapping file (mini YAML parser — keeps the script dependency-free)
# ---------------------------------------------------------------------------


def load_mapping(path: Path) -> Dict[str, str]:
    """Parse a tiny ``key: value`` YAML file. Comments + blank lines OK."""
    if not path.is_file():
        raise SystemExit(f"Mapping file not found: {path}")
    mapping: Dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
            raise SystemExit(f"{path}:{line_no}: expected 'key: value', got: {raw!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise SystemExit(f"{path}:{line_no}: empty key")
        if key in mapping:
            raise SystemExit(f"{path}:{line_no}: duplicate key {key!r}")
        mapping[key] = value
    return mapping


# ---------------------------------------------------------------------------
# Validation + reporting
# ---------------------------------------------------------------------------


def validate(
    mcp_tools: Iterable[str], mapping: Dict[str, str], cli_paths: Set[str]
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Return (missing, mapped_ok, mapped_bad, skipped, stale_mapping_keys)."""
    missing: List[str] = []
    mapped_ok: List[str] = []
    mapped_bad: List[str] = []
    skipped: List[str] = []
    mcp_set = set(mcp_tools)

    for tool in mcp_tools:
        entry = mapping.get(tool)
        if entry is None:
            missing.append(tool)
            continue
        if entry.startswith("skip:"):
            reason = entry[len("skip:") :].strip()
            if not reason:
                missing.append(tool)
            else:
                skipped.append(tool)
            continue
        if entry in cli_paths:
            mapped_ok.append(tool)
        else:
            mapped_bad.append(tool)

    stale = sorted(set(mapping) - mcp_set)
    return missing, mapped_ok, mapped_bad, skipped, stale


def render_report(
    mcp_tools: List[str],
    mapping: Dict[str, str],
    cli_paths: Set[str],
    missing: List[str],
    mapped_ok: List[str],
    mapped_bad: List[str],
    skipped: List[str],
    stale: List[str],
) -> str:
    total = len(mcp_tools)
    lines: List[str] = []
    lines.append("# MCP ↔ CLI Parity Report\n")
    lines.append(f"- Total MCP tools: **{total}**")
    lines.append(f"- Mapped to CLI command: **{len(mapped_ok)}**")
    lines.append(f"- Skipped (with reason): **{len(skipped)}**")
    lines.append(f"- Missing mapping entry: **{len(missing)}**")
    lines.append(f"- Mapping points at non-existent CLI path: **{len(mapped_bad)}**")
    if stale:
        lines.append(f"- Stale mapping keys (no matching MCP tool): **{len(stale)}**")
    lines.append("")
    lines.append("| MCP tool | Status | CLI path / reason |")
    lines.append("| --- | --- | --- |")
    for tool in mcp_tools:
        entry = mapping.get(tool)
        if entry is None:
            status = "MISSING"
            detail = "_add an entry to tools/mcp_cli_mapping.yaml_"
        elif entry.startswith("skip:"):
            status = "skip"
            detail = entry[len("skip:") :].strip()
        elif entry in cli_paths:
            status = "ok"
            detail = f"`kaggle {entry}`"
        else:
            status = "BROKEN"
            detail = f"`{entry}` (not a registered CLI path)"
        lines.append(f"| `{tool}` | {status} | {detail} |")
    if stale:
        lines.append("")
        lines.append("## Stale mapping entries")
        lines.append("")
        lines.append("These mapping keys do not correspond to any MCP tool — "
                     "remove or update them:")
        for key in stale:
            lines.append(f"- `{key}` → `{mapping[key]}`")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcp-client",
        default=str(DEFAULT_LOCAL_MCP),
        help=f"Local path to McpClient.cs (default: {DEFAULT_LOCAL_MCP})",
    )
    parser.add_argument(
        "--mcp-client-url",
        default=DEFAULT_REMOTE_MCP,
        help=f"Raw URL of McpClient.cs to fetch when local copy is absent "
        f"(default: {DEFAULT_REMOTE_MCP})",
    )
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING),
        help=f"Path to mcp_cli_mapping.yaml (default: {DEFAULT_MAPPING})",
    )
    args = parser.parse_args(argv)

    source = fetch_mcp_client_source(args.mcp_client, args.mcp_client_url)
    mcp_tools = extract_mcp_tools(source)
    mapping = load_mapping(Path(args.mapping))
    cli_paths = collect_cli_commands()

    missing, mapped_ok, mapped_bad, skipped, stale = validate(mcp_tools, mapping, cli_paths)
    print(render_report(mcp_tools, mapping, cli_paths, missing, mapped_ok, mapped_bad, skipped, stale))

    failed = bool(missing) or bool(mapped_bad)
    if failed:
        if missing:
            print(f"\nERROR: {len(missing)} MCP tool(s) without a mapping entry:", file=sys.stderr)
            for tool in missing:
                print(f"  - {tool}", file=sys.stderr)
        if mapped_bad:
            print(
                f"\nERROR: {len(mapped_bad)} mapping(s) point to non-existent CLI paths:",
                file=sys.stderr,
            )
            for tool in mapped_bad:
                print(f"  - {tool}: {mapping[tool]!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
