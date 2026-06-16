# Configuration CLI Reference

Use `kaggle config` to view, set, and unset local Kaggle CLI configuration. The
CLI reads config from a JSON file plus `KAGGLE_` environment variables.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Write access to the Kaggle config directory when setting or unsetting values.

## Command Hierarchy

```text
kaggle config
├── view
├── set
└── unset
```

## Configuration File Location

- If `KAGGLE_CONFIG_DIR` is set, the CLI uses that directory.
- Otherwise it normally uses `~/.kaggle`.
- On Linux, when `~/.kaggle` does not exist, it follows the XDG config path:
  `${XDG_CONFIG_HOME:-~/.config}/kaggle`.
- The config file name is `kaggle.json`.
- Newly created config files are chmodded to `0600`.
- Environment variables beginning with `KAGGLE_` are merged into config values
  after file values are read.

## `kaggle config view`

Prints current config values.

**Usage:**

```bash
kaggle config view
```

**Options:**

- None.

**Examples:**

```bash
kaggle config view
```

**Purpose:** Inspect configured username, auth method, path, proxy, and default
competition.

## `kaggle config set`

Sets a configuration value.

**Usage:**

```bash
kaggle config set -n <NAME> -v <VALUE>
```

**Options:**

- `-n, --name <NAME>`: Config key.
- `-v, --value <VALUE>`: Config value.

**Common names:**

- `competition`: Default competition slug.
- `path`: Default download folder.
- `proxy`: Proxy for HTTP requests.

**Examples:**

```bash
kaggle config set -n competition -v titanic
kaggle config set -n path -v /tmp/kaggle-downloads
kaggle config set -n proxy -v http://proxy.example:8080
```

**Purpose:** Persist default values used by other commands.

## `kaggle config unset`

Removes a configuration value from the config file.

**Usage:**

```bash
kaggle config unset -n <NAME>
```

**Options:**

- `-n, --name <NAME>`: Config key to remove.

**Examples:**

```bash
kaggle config unset -n competition
```

**Purpose:** Clear a persisted default and return to fallback behavior.

## Behavior Details

- `set` updates both the in-memory config and the JSON config file.
- `unset` removes a key only from the config file; environment variables can
  still provide a value on the next load.
- Download commands use configured `path` as the base default when no explicit
  path is provided.
- Competition commands that accept optional competition slugs can use configured
  `competition` as the default.

## Notes

- CLI help text names `competition`, `path`, and `proxy` as valid config names.
  Authentication-related keys may also appear in config output.
- If the config file is world-readable on non-Windows systems, the CLI prints a
  permissions warning.
