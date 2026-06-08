# Authentication CLI Reference

Use `kaggle auth` to run OAuth login, print an OAuth access token, or revoke the
active OAuth refresh token. The CLI also supports non-command authentication via
`KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, and legacy API key credentials.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- A Kaggle account.
- Browser access for the default OAuth login flow, or use `--no-launch-browser`
  to print a URL.

## Command Hierarchy

```text
kaggle auth
├── login
├── print-access-token
└── revoke
```

## Authentication Sources

The CLI authentication flow tries these sources:

1. Access token sources such as `KAGGLE_API_TOKEN` or
   `~/.kaggle/access_token`.
2. Legacy API key config from environment/config values `username` and `key`.
3. OAuth credentials created by `kaggle auth login`.

If none succeed, the CLI prints auth help and exits.

Some commands are allowed logged out, including help/version and selected
public dataset file/download flows.

## `kaggle auth login`

Runs Kaggle OAuth login.

**Usage:**

```bash
kaggle auth login [options]
```

**Options:**

- `--no-launch-browser`: Do not launch a browser; print an auth URL instead.
- `--force`: Re-run login even if already logged in.

**Examples:**

```bash
kaggle auth login
kaggle auth login --no-launch-browser
kaggle auth login --force
```

**Purpose:** Store OAuth credentials for the active Kaggle account.

**Behavior details:**

- If credentials already exist and `--force` is not set, the command prints the
  current account and exits with instructions to use `--force`.
- The OAuth flow requests the default scope `resources.admin:*`.

## `kaggle auth print-access-token`

Prints an access token for the active OAuth account.

**Usage:**

```bash
kaggle auth print-access-token [options]
```

**Options:**

- `--expiration <DURATION>`: Override token duration.

**Examples:**

```bash
kaggle auth print-access-token
```

**Purpose:** Emit a token that can be placed in `KAGGLE_API_TOKEN` or another
supported token source.

**Behavior details:**

- Requires OAuth credentials from `kaggle auth login`.
- If no OAuth credentials exist, the command tells the user to run
  `kaggle auth login`.
- Some CLI help examples use short forms such as `6h`. For automation, omit
  `--expiration` unless the accepted duration format has been verified with
  the installed CLI.

## `kaggle auth revoke`

Revokes the active OAuth refresh token.

**Usage:**

```bash
kaggle auth revoke [options]
```

**Options:**

- `--reason <TEXT>`: Reason sent to the server.

**Examples:**

```bash
kaggle auth revoke
kaggle auth revoke --reason "rotating credentials"
```

**Purpose:** Invalidate the active OAuth credential.

**Behavior details:**

- If no OAuth credentials exist, the command prints `There is no token to
  revoke.` and exits successfully.
- If no reason is provided, the default reason is
  `Manually revoked by user with kaggle-cli`.

## Legacy And Token Auth Notes

- `KAGGLE_API_TOKEN` can authenticate without a legacy username/key pair when
  the token introspection succeeds.
- Legacy config can come from `~/.kaggle/kaggle.json` or environment variables
  such as `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- Prefer `kaggle auth login` for OAuth credentials or `KAGGLE_API_TOKEN` for
  non-interactive authentication.
