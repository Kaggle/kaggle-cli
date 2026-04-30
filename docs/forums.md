# Forums Commands

Commands for browsing and reading Kaggle discussion forums.

## `kaggle forums`

Lists all discussion forums. Also available as `kaggle forums list`.

**Alias:** `fo`

**Usage:**

```bash
kaggle forums [options]
```

**Options:**

*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.

**Example:**

List all forums in CSV format:

```bash
kaggle forums -v
```

**Purpose:**

This command helps you discover all available discussion forums on Kaggle.

## `kaggle forums topics`

Lists discussion topics in a forum.

**Usage:**

```bash
kaggle forums topics <FORUM> [options]
```

**Arguments:**

*   `<FORUM>`: Forum slug (e.g., `getting-started`, `product-feedback`).

**Options:**

*   `--sort-by <SORT_BY>`: Sort order. Valid options: `hot`, `top`, `new`, `recent`, `active`, `relevance`.
*   `--page-size <PAGE_SIZE>`: Number of items per page.
*   `--page-token <PAGE_TOKEN>`: Page token for pagination.
*   `-s, --search <SEARCH_TERM>`: Search query.
*   `--category <CATEGORY>`: Filter by category. Valid options: `all`, `forums`, `competitions`, `datasets`, `competition_write_ups`, `models`, `benchmarks`.
*   `--group <GROUP>`: Filter by group. Valid options: `all`, `owned`, `upvoted`, `bookmarked`, `my_activity`, `drafts`.
*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.

**Example:**

List topics in the "getting-started" forum sorted by most recent, showing 5 per page:

```bash
kaggle forums topics getting-started --sort-by recent --page-size 5
```

**Purpose:**

This command lets you browse discussion topics within a specific forum, with filtering and sorting options.

## `kaggle forums topics show`

Displays a topic with all comments in tree form (indented).

**Usage:**

```bash
kaggle forums topics show <TOPIC_REF> [options]
```

**Arguments:**

*   `<TOPIC_REF>`: A topic reference, which can be:
    *   `<forum-name>/<topic-id>` (e.g., `getting-started/12345`)
    *   `<forum-name> <topic-id>` (two separate arguments)
    *   `<topic-id>` (bare numeric ID)

**Options:**

*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.
*   `--page-size <PAGE_SIZE>`: Number of comments to show per page.
*   `--page-token <PAGE_TOKEN>`: Page token for comment pagination.

**Example:**

Show topic 12345 from the "getting-started" forum:

```bash
kaggle forums topics show getting-started/12345
```

Show the same topic using two separate arguments:

```bash
kaggle forums topics show getting-started 12345
```

Show a topic by bare numeric ID:

```bash
kaggle forums topics show 12345
```

**Purpose:**

This command displays a full discussion topic along with all of its comments rendered in an indented tree structure.
