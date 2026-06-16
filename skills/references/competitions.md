# Competitions CLI Reference

Use `kaggle competitions` or alias `kaggle c` to discover competitions, inspect
and download competition files, submit predictions or code-kernel output, inspect
submissions and leaderboards, and browse competition discussion topics.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials for most commands. Some file listing/download flows may
  work logged out when the competition permits public access.
- For commands that omit `<COMPETITION>`, set a default with:

```bash
kaggle config set -n competition -v titanic
```

## Command Hierarchy

```text
kaggle competitions (alias: kaggle c)
├── list
├── files
├── download
├── submit
├── submissions
├── leaderboard
├── team-submissions
├── episodes
├── replay
├── logs
├── pages
├── topics
│   ├── list
│   └── show
└── topic-messages
```

## `kaggle competitions list`

Lists available competitions.

**Usage:**

```bash
kaggle competitions list [options]
```

**Options:**

- `--group <GROUP>`: Competition group.
- `--category <CATEGORY>`: Competition category.
- `--sort-by <SORT>`: Sort order.
- `-p, --page <PAGE>`: Page number.
- `--page-size <SIZE>`: Number of items on a page.
- `--page-token <TOKEN>`: Page token.
- `-s, --search <TERM>`: Search text.
- `-v, --csv`: Print CSV instead of table.

**Examples:**

```bash
kaggle competitions list
kaggle competitions list --category gettingStarted --sort-by latestDeadline
kaggle c list -s titanic -v
```

**Purpose:** Find competition slugs to use with other commands.

## `kaggle competitions files`

Lists data files for a competition.

**Usage:**

```bash
kaggle competitions files [COMPETITION] [options]
```

**Arguments:**

- `[COMPETITION]`: Competition URL suffix. Uses configured default when omitted.

**Options:**

- `--page-token <TOKEN>`: Page token.
- `--page-size <SIZE>`: Page size, default 20.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions files titanic
kaggle c files titanic --page-size 3 -v -q
```

**Purpose:** Inspect available files before downloading.

## `kaggle competitions download`

Downloads one or all competition data files.

**Usage:**

```bash
kaggle competitions download [COMPETITION] [options]
```

**Options:**

- `-f, --file <NAME>`: Download one file. Downloads all files when omitted.
- `-p, --path <PATH>`: Download directory.
- `-w, --wp`: Download to current working path.
- `-o, --force`: Force download even if local file looks current.
- `-q, --quiet`: Suppress progress output.

**Examples:**

```bash
kaggle competitions download titanic
kaggle competitions download titanic -f train.csv -p data
kaggle c download -w -o -q
```

**Purpose:** Retrieve competition data for local training or analysis.

## `kaggle competitions submit`

Creates a competition submission from a local file or a kernel output file.

**Usage:**

```bash
kaggle competitions submit [COMPETITION] -m <MESSAGE> [options]
```

**Options:**

- `-f, --file <FILE>`: Local submission file, or output file name for code competitions.
- `-k, --kernel <KERNEL>`: Kernel name for code competition submissions.
- `-m, --message <MESSAGE>`: Required submission description.
- `-v, --version <VERSION>`: Kernel version for code competitions.
- `--sandbox`: Mark as sandbox submission for competition hosts/admins.
- `-q, --quiet`: Suppress progress output.

**Examples:**

```bash
kaggle competitions submit titanic -f submission.csv -m "baseline"
kaggle c submit lux-ai -k user/agent -f submission.tar.gz -m "agent run" -v 3
```

**Purpose:** Submit predictions or code-kernel output to a competition.

**Notes:** `--sandbox` is intended for competition hosts/admins. Code competition
submission uses `-k`, `-f`, and optional `-v`.

## `kaggle competitions submissions`

Shows your submissions for a competition.

**Usage:**

```bash
kaggle competitions submissions [COMPETITION] [options]
```

**Options:**

- `--page-size <SIZE>`: Page size.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions submissions titanic -v
```

**Purpose:** Review submission status and scores.

## `kaggle competitions leaderboard`

Views or downloads competition leaderboard data.

**Usage:**

```bash
kaggle competitions leaderboard [COMPETITION] [options]
```

**Options:**

- `-s, --show`: Show top leaderboard rows.
- `-d, --download`: Download the full leaderboard.
- `-p, --path <PATH>`: Download directory.
- `--page-size <SIZE>`: Page size.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions leaderboard titanic --show
kaggle c leaderboard titanic --download -p leaderboards
```

**Purpose:** Inspect ranking data from the competition leaderboard.

## `kaggle competitions team-submissions`

Lists a team's public submissions.

**Usage:**

```bash
kaggle competitions team-submissions <TEAM_ID> [options]
```

**Arguments:**

- `<TEAM_ID>`: Team ID. Find team IDs from the competition leaderboard display.

**Options:**

- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions team-submissions 12345
```

**Purpose:** For simulation competitions, lists every active public submission;
for regular competitions, lists the team's public leaderboard submission.

## Simulation Competition Commands

### `kaggle competitions episodes`

Lists episodes for a submission in a simulation competition.

**Usage:**

```bash
kaggle competitions episodes <SUBMISSION_ID> [options]
```

**Arguments:**

- `<SUBMISSION_ID>`: Submission ID. Find it with
  `kaggle competitions submissions <competition>`.

**Options:**

- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions episodes 12345678
```

**Purpose:** Inspect simulation match episodes associated with a submission.

### `kaggle competitions replay`

Downloads the replay for a simulation episode.

**Usage:**

```bash
kaggle competitions replay <EPISODE_ID> [options]
```

**Arguments:**

- `<EPISODE_ID>`: Episode ID from `kaggle competitions episodes <submission_id>`.

**Options:**

- `-p, --path <PATH>`: Destination folder.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions replay 987654 -p replays
```

**Purpose:** Download the replay artifact for local review.

### `kaggle competitions logs`

Downloads logs for a specific agent in a simulation episode.

**Usage:**

```bash
kaggle competitions logs <EPISODE_ID> <AGENT_INDEX> [options]
```

**Arguments:**

- `<EPISODE_ID>`: Episode ID from `kaggle competitions episodes <submission_id>`.
- `<AGENT_INDEX>`: Zero-based position of the agent in the episode.

**Options:**

- `-p, --path <PATH>`: Destination folder.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions logs 987654 0 -p logs
```

**Purpose:** Download per-agent episode logs for debugging.

## `kaggle competitions pages`

Lists pages for a competition.

**Usage:**

```bash
kaggle competitions pages [COMPETITION] [options]
```

**Options:**

- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.
- `--content`: Show full page content.
- `--page-name <NAME>`: Filter to a specific page, such as `description`,
  `rules`, or `evaluation`.

**Examples:**

```bash
kaggle competitions pages titanic
kaggle competitions pages titanic --page-name rules --content
```

**Purpose:** Inspect competition page metadata or retrieve page content.

## Competition Discussion Commands

### `kaggle competitions topics list`

Lists discussion topics for a competition.

**Usage:**

```bash
kaggle competitions topics list [COMPETITION] [options]
```

**Arguments:**

- `[COMPETITION]`: Competition slug. If omitted, the default competition from
  config may be used.

**Options:**

- `-s, --sort-by <SORT>`: One of `hot`, `top`, `new`, `recent`, `active`, `relevance`.
- `-p, --page <PAGE>`: Page number.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions topics list titanic
kaggle competitions topics list titanic --sort-by recent -p 2
```

**Purpose:** Browse competition discussions before opening a specific topic.

`kaggle competitions topics` without `list` works as a shortcut for
listing topics.

### `kaggle competitions topics show`

Shows a topic and comments in tree form.

**Usage:**

```bash
kaggle competitions topics show <TOPIC_REF> [TOPIC_ID] [options]
```

**Arguments:**

- `<TOPIC_REF>`: Topic reference. It may be `<forum-or-entity>/<topic-id>` or
  an entity reference when using the two-argument form.
- `[TOPIC_ID]`: Optional topic ID for the two-argument form.

**Options:**

- `--page-size <SIZE>`: Number of comments to return.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions topics show titanic/12345
kaggle competitions topics show titanic 12345
```

**Purpose:** Read a competition discussion topic and its comments.

### `kaggle competitions topic-messages`

Lists messages in a competition discussion topic.

**Usage:**

```bash
kaggle competitions topic-messages [COMPETITION] <TOPIC_ID> [options]
```

**Arguments:**

- `[COMPETITION]`: Competition slug. If omitted, the default competition from
  config may be used.
- `<TOPIC_ID>`: Discussion topic ID.

**Options:**

- `-s, --sort-by <SORT>`: One of `hot`, `new`, `old`, `top`.
- `-n, --page-size <SIZE>`: Max top-level messages to return; `-1` for all.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle competitions topic-messages titanic 12345
kaggle competitions topic-messages titanic 12345 --sort-by old -n 50
```

**Purpose:** Use the deprecated hidden alias to list comments for a competition
topic when older workflows still call `topic-messages`.
