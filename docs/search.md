# Search Command

Search across all Kaggle content from a single command.

## `kaggle search`

Runs a unified search over Kaggle competitions, datasets, notebooks, models,
users, and discussions and returns one ranked list of results. This is the
cross-content equivalent of the per-command `-s/--search` flags (such as
`kaggle datasets list --search`); use it when you don't yet know which content
type you're looking for, or when you want results from several types at once.

By default, `kaggle search` searches all CLI-supported content types
(competition, dataset, notebook, model, user, discussion, benchmark) — not every
backend document type (it excludes types the CLI cannot render usefully, such as
comments, blogs, and courses). Result ordering uses the backend's canonical
cross-content ranking. Use `--type` to narrow to specific types.

**Usage:**

```bash
kaggle search "<query>" [options]
```

**Options:**

*   `query`: The term(s) to search for (required).
*   `-t, --type <TYPES>`: Restrict results to a comma-separated list of content types. Valid types: `competition`, `dataset`, `notebook`, `model`, `user`, `discussion`, `benchmark`. If omitted, all of these CLI-supported types are searched.
*   `-m, --mine`: Restrict the search to your own content.
*   `--sort-by <SORT_BY>`: Sort order. One of: `relevance` (default), `hotness`, `votes`, `dateCreated`, `dateUpdated`, `totalComments`, `lastViewed`.
*   `--page-size <SIZE>`: Number of results to show on a page (default: 20, max: 100).
*   `--page-token <TOKEN>`: Page token for results paging (printed as `Next Page Token = ...` when more results exist — at the top of a page for the default `table` format, and on stderr for `--csv` and `--format csv`/`json` so that stdout stays parseable; see [Output Format](./output_format.md#pagination-tokens)).
*   `-v, --csv`: Print results in CSV format instead of a table.
*   `--format <FORMAT>`: Print results in the selected format (`csv`, `table`, `json`). Supports field projection, e.g. `--format 'json(type,ref)'`.

**Result columns:** `type`, `ref`, `title`, `owner`, `votes`. The `ref` column
is the identifier you can pass to other commands — `owner/slug` for datasets,
notebooks, and models; the bare slug for competitions and users.

**Examples:**

1.  Search everything for a term:

    ```bash
    kaggle search "protein folding"
    ```

2.  Search only datasets and models:

    ```bash
    kaggle search "diffusion" --type dataset,model
    ```

3.  Find users:

    ```bash
    kaggle search "andrew ng" --type user
    ```

4.  Search your own content, most recently updated first:

    ```bash
    kaggle search "baseline" --mine --sort-by dateUpdated
    ```

5.  Get machine-readable output:

    ```bash
    kaggle search "titanic" --format json
    ```

6.  Fetch the next page:

    ```bash
    kaggle search "llm" --page-size 50 --page-token <TOKEN>
    ```
