# Competitions Commands

Commands for interacting with Kaggle competitions.

For tutorials on how to submit to competitions :
* [How to Submit to a Competition](./tutorials.md#tutorial-how-to-submit-to-a-competition)
* [How to Submit to a Code Competition](./tutorials.md#tutorial-how-to-submit-to-a-code-competition)

## `kaggle competitions list`

Lists available competitions.

**Usage:**

```bash
kaggle competitions list [options]
```

**Options:**

*   `--group <GROUP>`: Filter by competition group. Valid options: `general`, `entered`, `inClass`.
*   `--category <CATEGORY>`: Filter by competition category. Valid options: `all`, `featured`, `research`, `recruitment`, `gettingStarted`, `masters`, `playground`.
*   `--sort-by <SORT_BY>`: Sort results. Valid options: `grouped`, `prize`, `earliestDeadline`, `latestDeadline`, `numberOfTeams`, `recentlyCreated` (default: `latestDeadline`).
*   `-p, --page <PAGE>`: Page number for results (default: 1).
*   `-s, --search <SEARCH_TERM>`: Search term.
*   `-v, --csv`: Print results in CSV format.
*   `--format`: Output format (`csv`, `table`, `json`, or a field projection). See [output_format.md](./output_format.md).

**Output columns:**

`ref`, `deadline`, `category`, `reward`, `teamCount`, `userHasEntered`, `userRank`

`userRank` is your public leaderboard position when you have entered the competition. It is `0` when you have not entered, or when no public rank is available yet.

**Example:**

List featured competitions in the general group, sorted by prize:

```bash
kaggle competitions list --group general --category featured --sort-by prize
```

List entered competitions with rank in CSV format:

```bash
kaggle competitions list --group entered -v
```

**Purpose:**

This command helps you discover new competitions or find specific ones based on various criteria. Use `--group entered` to see your rank across competitions you have joined.

## `kaggle competitions files`

Lists files for a specific competition.

**Usage:**

```bash
kaggle competitions files <COMPETITION> [options]
```

**Arguments:**

*   `<COMPETITION>`: Competition URL suffix (e.g., `titanic`).

**Options:**

*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.
*   `--page-token <PAGE_TOKEN>`: Page token for results paging.
*   `--page-size <PAGE_SIZE>`: Number of items to show on a page (default: 20, max: 200).

**Example:**

List the first 3 files for the "titanic" competition in CSV format, quietly:

```bash
kaggle competitions files titanic --page-size=3 -v -q
```

**Purpose:**

Use this command to see the data files available for a competition before downloading them.

## `kaggle competitions download`

Downloads competition files.

**Usage:**

```bash
kaggle competitions download <COMPETITION> [options]
```

**Arguments:**

*   `<COMPETITION>`: Competition URL suffix (e.g., `titanic`).

**Options:**

*   `-f, --file <FILE_NAME>`: Specific file to download (downloads all if not specified).
*   `-p, --path <PATH>`: Folder to download files to (defaults to current directory).
*   `-w, --wp`: Download files to the current working path (equivalent to `-p .`).
*   `-o, --force`: Force download, overwriting existing files.
*   `-q, --quiet`: Suppress verbose output.

**Examples:**

1.  Download all files for the "titanic" competition to the current directory, overwriting existing files, quietly:

    ```bash
    kaggle competitions download titanic -w -o -q
    ```

2.  Download the `test.csv` file from the "titanic" competition to a folder named `tost`:

    ```bash
    kaggle competitions download titanic -f test.csv -p tost
    ```

**Purpose:**

This command allows you to get the necessary data files for a competition onto your local machine.

## `kaggle competitions submit`

Makes a new submission to a competition.

**Usage:**

```bash
kaggle competitions submit <COMPETITION> -f <FILE_NAME> -m <MESSAGE> [options]
```

**Arguments:**

*   `<COMPETITION>`: Competition URL suffix (e.g., `house-prices-advanced-regression-techniques`).
*   `-f, --file <FILE_NAME>`: The submission file.
*   `-m, --message <MESSAGE>`: The submission message.

**Options:**

*   `-k, --kernel <KERNEL>`: Name of the kernel (notebook) to submit (for code competitions).
*   `-v, --version <VERSION>`: Version of the kernel to submit (e.g. `2`).
*   `-q, --quiet`: Suppress verbose output.
*   `--sandbox`: Mark submission as a sandbox submission (competition hosts/admins only).
*   `--wait [SECONDS]`: Wait for the submission to finish scoring, printing the public score when done. Optionally pass a timeout in seconds (`0` or no value = wait up to 12 hours, the maximum notebook runtime). Exits non-zero if scoring fails or the timeout is reached.
*   `--poll-interval <SECONDS>`: Maximum seconds between status polls while waiting (default: `60`, minimum: `5`). Polling starts at 5s and increases automatically.

On a successful submission the command prints the numeric submission ref, e.g. `Submission ref: 12345678`. You can look that submission up later with [`kaggle competitions submission`](#kaggle-competitions-submission).

**Example: Standard (not code) competition:**

Submit `sample_submission.csv` to the "house-prices-advanced-regression-techniques" competition with the message "Test message":

```bash
kaggle competitions submit house-prices-advanced-regression-techniques -f sample_submission.csv -m "Test message"
```

**Example: Code competition:**

Submit the `submission.csv` produced by version `3` of your `<YOUR_USERNAME>/rsna-submission` for the `rsna-2024-lumbar-spine-degenerative-classification` competition:

```bash
kaggle competitions submit rsna-2024-lumbar-spine-degenerative-classification -f submission.csv -k <YOUR_USERNAME>/rsna-submission -v 3 -m "Test message"
```

**Example: Submit and wait for the score (useful in CI):**

Submit and block until scoring finishes (up to a 10-minute timeout), then print the public score:

```bash
kaggle competitions submit house-prices-advanced-regression-techniques -f sample_submission.csv -m "CI run" --wait 600
```

The command exits `0` once the submission is scored and non-zero if scoring fails or the timeout is reached, so it can gate a pipeline.

**Purpose:**

Use this command to upload your predictions or code to a competition for scoring.

## `kaggle competitions submission`

Shows the status and score of a single submission by its numeric ref (as printed by `kaggle competitions submit`).

**Usage:**

```bash
kaggle competitions submission <SUBMISSION_REF>
```

**Arguments:**

*   `<SUBMISSION_REF>`: The numeric submission ref printed by `kaggle competitions submit`.

**Example:**

```bash
kaggle competitions submission 12345678
```

Output:

```
Submission Ref:  12345678
Status:          COMPLETE
Public Score:    0.98765
Private Score:
Description:     Test message
Submission Date: 2026-07-19 12:00:00
```

**Purpose:**

Use this command to check whether a submission has finished scoring and to read its public score — for example, after submitting without `--wait`, or from a script polling for results.

## `kaggle competitions submissions`

Shows your past submissions for a competition.

**Usage:**

```bash
kaggle competitions submissions <COMPETITION> [options]
```

**Arguments:**

*   `<COMPETITION>`: Competition URL suffix (e.g., `house-prices-advanced-regression-techniques`).

**Options:**

*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.

**Example:**

Show submissions for "house-prices-advanced-regression-techniques" in CSV format, quietly:

```bash
kaggle competitions submissions house-prices-advanced-regression-techniques -v -q
```

**Purpose:**

This command allows you to review your previous submission attempts and their scores.

## `kaggle competitions submission-download`

Downloads the submitted file for a single submission by its numeric id.

**Usage:**

```bash
kaggle competitions submission-download <SUBMISSION_ID> [options]
```

**Arguments:**

*   `<SUBMISSION_ID>`: The numeric submission id printed by `kaggle competitions submit`, or listed by `kaggle competitions submissions <COMPETITION>`.

**Options:**

*   `-p, --path <PATH>`: Folder to download the file to. Defaults to the current working directory's Kaggle download location.
*   `-o, --force`: Download even if a local copy already exists (skips the up-to-date check).
*   `-q, --quiet`: Suppress verbose output.

**Example:**

Download the file for submission `12345678` into `./subs`:

```bash
kaggle competitions submission-download 12345678 -p ./subs
```

**Purpose:**

Use this command to retrieve the exact file you (or a teammate) submitted — for example, to inspect an old submission or reproduce a scored result.

## `kaggle competitions leaderboard`

Gets competition leaderboard information.

**Usage:**

```bash
kaggle competitions leaderboard <COMPETITION> [options]
```

**Arguments:**

*   `<COMPETITION>`: Competition URL suffix (e.g., `titanic`).

**Options:**

*   `-s, --show`: Show the top of the leaderboard in the console.
*   `-d, --download`: Download the entire leaderboard to a CSV file.
*   `-p, --path <PATH>`: Folder to download the leaderboard to (if `-d` is used).
*   `-v, --csv`: Print results in CSV format (used with `-s`).
*   `-q, --quiet`: Suppress verbose output.

**Examples:**

1.  Download the "titanic" leaderboard to a folder named `leaders`, quietly:

    ```bash
    kaggle competitions leaderboard titanic -d -p leaders -q
    ```

2.  Download the leaderboard and save it to `leaderboard.txt`:

    ```bash
    kaggle competitions leaderboard titanic > leaderboard.txt
    ```

**Purpose:**

This command lets you view your ranking and the scores of other participants in a competition.

## `kaggle competitions topics list`

Lists discussion topics for a competition.

**Usage:**

```bash
kaggle competitions topics list [COMPETITION] [options]
```

Note: `kaggle competitions topics` (without `list` subcommand) is supported as a shortcut to list topics for the default competition (configured via `kaggle config set competition`).

**Arguments:**

*   `[COMPETITION]`: Competition URL suffix (e.g., `titanic`). Optional if default competition is configured.

**Options:**

*   `-s, --sort-by <SORT_BY>`: Sort order. Valid options: `hot`, `top`, `new`, `recent`, `active`, `relevance`.
*   `--search <SEARCH>`: Search query to filter topics.
*   `--page-size <PAGE_SIZE>`: Number of items to show on a page. Default is 20, max is 200.
*   `--page-token <PAGE_TOKEN>`: Page token for results paging.
*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.

**Example:**

List discussion topics for the "titanic" competition sorted by most recent:

```bash
kaggle competitions topics list titanic -s recent
```

**Purpose:**

This command lets you browse discussion topics for a specific competition.

## `kaggle competitions topics show`

Displays a competition discussion topic with all comments in tree form.

**Usage:**

```bash
kaggle competitions topics show <TOPIC_REF> [options]
```

**Arguments:**

*   `<TOPIC_REF>`: A topic reference, which can be:
    *   `<competition>/<topic-id>` (e.g., `titanic/12345`)
    *   `<competition> <topic-id>` (two separate arguments, where `<topic-id>` is passed as second argument)
    *   `<topic-id>` (bare numeric ID)

**Options:**

*   `--page-size <PAGE_SIZE>`: Number of comments to show per page.
*   `--page-token <PAGE_TOKEN>`: Page token for comment pagination.
*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.

**Example:**

Show topic 12345 from the "titanic" competition:

```bash
kaggle competitions topics show titanic/12345
```

**Purpose:**

This command displays a full discussion topic along with all of its comments rendered in an indented tree structure.

## `kaggle competitions topic-messages`

Lists messages within a competition discussion topic.

> **Deprecated:** This command is deprecated in favor of `kaggle competitions topics show`. It will be removed in a future release.

**Usage:**

```bash
kaggle competitions topic-messages <COMPETITION> <TOPIC_ID> [options]
```

**Arguments:**

*   `<COMPETITION>`: Competition URL suffix (e.g., `titanic`).
*   `<TOPIC_ID>`: The discussion topic id.

**Options:**

*   `-s, --sort-by <SORT_BY>`: Sort order. Valid options: `best`, `new`, `old`.
*   `-n, --page-size <PAGE_SIZE>`: Max top-level messages to return; `-1` for all.
*   `-v, --csv`: Print results in CSV format.
*   `-q, --quiet`: Suppress verbose output.

**Example:**

List all messages for topic 12345 in the "titanic" competition, sorted by newest first:

```bash
kaggle competitions topic-messages titanic 12345 -s new -n -1
```

**Purpose:**

This command displays the messages within a specific competition discussion topic.
