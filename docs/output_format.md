# Kaggle CLI Output Formatting Documentation

This documentation describes the output formatting options available in the Kaggle CLI.

## Output Format Options

The Kaggle CLI supports choosing the output format for various commands that list information.

### `--csv` (or `-v`)

Historically, many commands supported a `-v` or `--csv` option to display output as comma-separated values (CSV) instead of a formatted table.

Example:
```sh
kaggle competitions list --csv
```

### `--format`

We have introduced a new `--format` option to provide a unified way to specify the output format.
It accepts the following values:
*   `csv`: Display output as comma-separated values.
*   `table`: Display output as a formatted table (default).
*   `json`: Display output as JSON (to be fully implemented in future updates).

Example:
```sh
kaggle competitions list --format csv
kaggle competitions list --format table
```

### Mutual Exclusion

The `--csv` (or `-v`) option and the `--format` option are **mutually exclusive**. You cannot specify both at the same time.

If you attempt to use both, the CLI will display an error:
```sh
kaggle competitions list --csv --format csv
# Error: argument --format: not allowed with argument -v/--csv
```

## Supported Commands

The following commands support both `--csv` (legacy) and `--format` options:

### Competitions
*   `kaggle competitions list`
*   `kaggle competitions files`
*   `kaggle competitions submissions`
*   `kaggle competitions leaderboard`
*   `kaggle competitions team-submissions`
*   `kaggle competitions episodes`
*   `kaggle competitions pages`
*   `kaggle competitions topic-messages`
*   `kaggle competitions topics list`
*   `kaggle competitions topics show`

### Datasets
*   `kaggle datasets list`
*   `kaggle datasets files`
*   `kaggle datasets topics list`
*   `kaggle datasets topics show`

### Kernels
*   `kaggle kernels list`
*   `kaggle kernels files`
*   `kaggle kernels topics list`
*   `kaggle kernels topics show`

### Models
*   `kaggle models list`
*   `kaggle models topics list`
*   `kaggle models topics show`
*   `kaggle models instances list`
*   `kaggle models instances files`
*   `kaggle models instances versions list`
*   `kaggle models instances versions files`

### Forums
*   `kaggle forums list`
*   `kaggle forums topics list`
*   `kaggle forums topics show`

### Benchmarks
*   `kaggle benchmarks topics list`
*   `kaggle benchmarks topics show`

### Quota
*   `kaggle quota`
