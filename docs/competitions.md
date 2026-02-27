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

**Example:**

List featured competitions in the general group, sorted by prize:

```bash
kaggle competitions list --group general --category featured --sort-by prize
```

**Purpose:**

This command helps you discover new competitions or find specific ones based on various criteria.

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

**Purpose:**

Use this command to upload your predictions or code to a competition for scoring.

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
