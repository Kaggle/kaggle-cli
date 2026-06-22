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
*   `json`: Display output as JSON.

Example:
```sh
kaggle competitions list --format csv
kaggle competitions list --format table
kaggle competitions list --format json
```

For most commands, the JSON output is a list of objects representing the rows, with keys corresponding to the column headers. For detailed commands like `topics show`, it returns a structured object:
```json
{
  "topic": { ... },
  "comments": [ ... ]
}
```

### Projections (Field Selection)

The `--format` option supports optional `gcloud`-style field selection (projections) by appending a comma-separated list of fields in parentheses to the format name. This allows you to limit the output to only the specified fields and control their order.

Projections are supported for all formats (`csv`, `table`, `json`).

Example:
```sh
# Only show 'ref' and 'reward' columns for competitions in a table
kaggle competitions list --format "table(ref,reward)"

# Export only 'id' and 'publicScore' to JSON for team submissions
kaggle competitions team-submissions --format "json(id,publicScore)" <team_id>

# Export only 'name' and 'size' to CSV for dataset files
kaggle datasets files -d zillow/zecon --format "csv(name,size)"
```

You can specify fields using either their field names (e.g. `totalBytes`) or their display labels (e.g. `size`). If a field is not recognized, the CLI will display an error listing the allowed fields.

#### Special Case: Topics Show

For `topics show` commands, which output both a parent topic and a list of comments, the projection is applied to both types of objects. Fields matching the topic are applied to the topic output, and fields matching comments are applied to the comment output.

Example:
```sh
kaggle forums topics show 123 --format "json(title,content)"
```
In this case, `title` (which is a topic field) will be preserved in the topic output, and `content` (which is a comment field) will be preserved in the comments output.
```json
{
  "topic": {
    "title": "Test Title"
  },
  "comments": [
    {
      "content": "Comment Content"
    }
  ]
}
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
