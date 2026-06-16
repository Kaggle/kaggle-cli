# Forums CLI Reference

Use `kaggle forums` or alias `kaggle f` to list Kaggle discussion forums, list
topics in forums, and show a topic with comments in tree form.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials may be needed for user-specific forum groups such as owned,
  bookmarked, upvoted, or drafts.

## Command Hierarchy

```text
kaggle forums (alias: kaggle f)
├── list
└── topics
    ├── list
    └── show
```

`kaggle forums` without a subcommand works as a shortcut for
`kaggle forums list`.

## `kaggle forums list`

Lists discussion forums.

**Usage:**

```bash
kaggle forums list [options]
```

**Options:**

- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle forums
kaggle forums list -v
kaggle f list -q
```

**Purpose:** Find forum slugs such as `getting-started` or `product-feedback`.

## `kaggle forums topics list`

Lists topics in a forum or broader topic category/group.

**Usage:**

```bash
kaggle forums topics list [FORUM] [options]
```

**Arguments:**

- `[FORUM]`: Forum slug. Use `kaggle forums` to list available forums.

**Options:**

- `--group <GROUP>`: Topic group filter. Use live `--help` output when unsure
  which group names the installed CLI accepts.
- `--category <CATEGORY>`: Topic category filter. Use live `--help` output when
  unsure which category names the installed CLI accepts.
- `--sort-by <SORT>`: One of `hot`, `top`, `new`, `recent`, `active`, `relevance`.
- `-s, --search <TERM>`: Search text.
- `--page-size <SIZE>`: Page size.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle forums topics list getting-started
kaggle f topics list --category datasets --sort-by recent
kaggle forums topics list product-feedback -s "api" -v
```

**Purpose:** Browse discussion topics across global forums and resource-linked
categories.

`kaggle forums topics` without `list` works as a shortcut for listing
topics.

## `kaggle forums topics show`

Displays a topic and all comments in tree form.

**Usage:**

```bash
kaggle forums topics show <TOPIC_REF> [TOPIC_ID] [options]
```

**Arguments:**

- `<TOPIC_REF>`: Topic reference in `<forum-name>/<topic-id>` form, or a topic id.
- `[TOPIC_ID]`: Optional topic id when using two-argument form.

**Options:**

- `--page-size <SIZE>`: Page size.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: CSV output where supported.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle forums topics show getting-started/12345
kaggle f topics show getting-started 12345
```

**Purpose:** Read a discussion topic and comments from the CLI.

## Notes

- Resource-specific topics also exist under `competitions`, `datasets`,
  `models`, and `benchmarks`.
- For entity topics, use the entity command so default entity context and
  resource-specific validation are applied.
