# Benchmarks Commands

Commands for interacting with Kaggle Benchmarks. Benchmarks let you define evaluation tasks as Python scripts, run them against one or more LLM models via the Kaggle Model Proxy, and download the results.

The top-level command is `kaggle benchmarks` (alias: `kaggle b`), which has three sub-groups:

*   **`auth`** — Fetch Model Proxy credentials.
*   **`init`** — Fetch credentials and default environment variables for local development.
*   **`tasks`** (alias: `t`) — Manage benchmark tasks (push, run, list, status, download, models, delete).

## `kaggle benchmarks auth`

Fetches a Model Proxy token and persists the credential environment variables to a file.

**Usage:**

```bash
kaggle benchmarks auth [options]
```

**Options:**

*   `-y, --yes`: Automatically confirm without prompting.
*   `--env-file <FILE>`: File to write environment variables to (default: `.env`).

**Example:**

Write Model Proxy credentials to the default `.env` file, confirming automatically:

```bash
kaggle b auth -y
```

**Purpose:**

This command fetches a short-lived Model Proxy API key and URL from Kaggle and appends them to your environment file. The variables written are:

*   `MODEL_PROXY_URL`
*   `MODEL_PROXY_API_KEY`
*   `MODEL_PROXY_EXPIRY_TIME`

## `kaggle benchmarks init`

Fetches Model Proxy credentials **and** additional default environment variables useful for local benchmark development. Also generates a starter example task file and a syntax reference document.

**Usage:**

```bash
kaggle benchmarks init [options]
```

**Options:**

*   `-y, --yes`: Automatically confirm without prompting.
*   `--env-file <FILE>`: File to write environment variables to (default: `.env`).
*   `--example-file <FILE>`: File to write the example benchmark task to (default: `example_task.py`).

**Examples:**

1.  Initialize with defaults (writes `.env`, `example_task.py`, and `kaggle_benchmarks_reference.md`):

    ```bash
    kaggle b init -y
    ```

2.  Initialize with a custom env file and example file:

    ```bash
    kaggle b init -y --env-file my_project/.env --example-file my_project/my_task.py
    ```

**Purpose:**

In addition to the three credential variables written by `auth`, `init` also writes:

*   `LLM_DEFAULT` — Default model slug for tasks.
*   `LLM_DEFAULT_EVAL` — Default model slug for evaluation.
*   `LLMS_AVAILABLE` — Comma-separated list of available model slugs.

`init` also creates two files alongside the example file:

*   **`example_task.py`** (or custom name via `--example-file`) — A starter Python script demonstrating how to define a benchmark task using `@task` decorators and the `kaggle_benchmarks` library.
*   **`kaggle_benchmarks_reference.md`** — A syntax reference document for the `kaggle-benchmarks` task API.

If either file already exists, it is skipped without overwriting.

---

## Tasks Commands

All tasks commands live under `kaggle benchmarks tasks` (alias: `kaggle b t`).

### `kaggle benchmarks tasks push`

Creates or updates a benchmark task from a local Python source file. The file must contain at least one function decorated with `@task`.

**Usage:**

```bash
kaggle benchmarks tasks push <TASK> -f <FILE> [options]
```

**Arguments:**

*   `<TASK>`: Task name. Automatically normalized to a URL-safe slug (e.g., `my_task` or `My Task` becomes `my-task`).

**Options:**

*   `-f, --file <FILE>` *(required)*: Path to the source Python file defining the task.
*   `--wait [TIMEOUT]`: Wait for the task creation to complete. Optionally specify a timeout in seconds (`0` or omit value = wait indefinitely).
*   `--poll-interval <SECONDS>`: Seconds between status polls when using `--wait` (default: `10`).

**Examples:**

1.  Push a task and return immediately:

    ```bash
    kaggle b t push my-task -f benchmark.py
    ```

2.  Push a task and wait for creation to finish:

    ```bash
    kaggle b t push my-task -f benchmark.py --wait
    ```

3.  Push a task and wait with a 60-second timeout, polling every 5 seconds:

    ```bash
    kaggle b t push my-task -f benchmark.py --wait 60 --poll-interval 5
    ```

**Purpose:**

This command reads a `.py` file, converts it to a Jupyter notebook format, and uploads it to Kaggle as a benchmark task. If a task with the same slug already exists, a new version is created. The file is validated to ensure it contains a `@task` decorator matching the given task name.

---

### `kaggle benchmarks tasks run`

Runs a previously pushed task against one or more models.

**Usage:**

```bash
kaggle benchmarks tasks run <TASK> [options]
```

**Arguments:**

*   `<TASK>`: Task name (slug).

**Options:**

*   `-m, --model <MODEL> [MODEL ...]`: One or more model slugs to run against. If omitted, an interactive model picker is displayed.
*   `--wait [TIMEOUT]`: Wait for runs to complete. Optionally specify a timeout in seconds (`0` or omit value = wait indefinitely).
*   `--poll-interval <SECONDS>`: Seconds between status polls when using `--wait` (default: `10`).

**Examples:**

1.  Run a task with interactive model selection:

    ```bash
    kaggle b t run my-task
    ```

2.  Run a task against specific models:

    ```bash
    kaggle b t run my-task -m google/gemini-2.5-pro anthropic/claude-sonnet-4
    ```

3.  Run a task and wait for all runs to finish:

    ```bash
    kaggle b t run my-task -m google/gemini-2.5-pro --wait
    ```

**Purpose:**

This command schedules benchmark runs on the server. The task must be in a `COMPLETED` creation state before it can be run. If no models are specified, the CLI presents a paginated list of available models for interactive selection.

---

### `kaggle benchmarks tasks list`

Lists benchmark tasks owned by the current user.

**Usage:**

```bash
kaggle benchmarks tasks list [options]
```

**Options:**

*   `--name-regex <REGEX>`: Filter task names by regular expression.
*   `--status <STATUS>`: Filter tasks by creation status. Valid values: `queued`, `running`, `completed`, `errored`.

**Examples:**

1.  List all your tasks:

    ```bash
    kaggle b t list
    ```

2.  List only completed tasks whose names contain "gemini":

    ```bash
    kaggle b t list --name-regex gemini --status completed
    ```

**Purpose:**

Displays a table of your benchmark tasks showing the task slug, creation status, and creation timestamp.

---

### `kaggle benchmarks tasks status`

Shows task details and per-model run status.

**Usage:**

```bash
kaggle benchmarks tasks status <TASK> [options]
```

**Arguments:**

*   `<TASK>`: Task name (slug).

**Options:**

*   `-m, --model <MODEL> [MODEL ...]`: Filter the run table to specific model slug(s).

**Examples:**

1.  Show full status for a task:

    ```bash
    kaggle b t status my-task
    ```

2.  Show status for specific models only:

    ```bash
    kaggle b t status my-task -m google/gemini-2.5-pro
    ```

**Purpose:**

Prints the task's metadata (slug, creation status, creation time, URL) followed by a table of all runs. Each run row shows the model name, run state, start time, and end time. Any errored runs display their error messages below the table.

---

### `kaggle benchmarks tasks download`

Downloads output files for completed benchmark runs.

**Usage:**

```bash
kaggle benchmarks tasks download <TASK> [options]
```

**Arguments:**

*   `<TASK>`: Task name (slug).

**Options:**

*   `-m, --model <MODEL> [MODEL ...]`: Download outputs only for specific model slug(s).
*   `-o, --output <DIRECTORY>`: Directory to download output files into (defaults to current working directory).

**Examples:**

1.  Download all completed run outputs for a task:

    ```bash
    kaggle b t download my-task
    ```

2.  Download outputs for a specific model into a custom directory:

    ```bash
    kaggle b t download my-task -m google/gemini-2.5-pro -o ./results
    ```

**Purpose:**

Downloads and extracts the output zip archive for each completed run. Files are organized in a hierarchical layout:

```
<output>/<task>/<model>/<run_id>/
```

Already-downloaded runs (where the output directory exists) are automatically skipped.

---

### `kaggle benchmarks tasks models`

Lists all available benchmark models.

**Usage:**

```bash
kaggle benchmarks tasks models
```

**Example:**

```bash
kaggle b t models
```

**Purpose:**

Prints a table of all models available for benchmark runs, showing each model's slug and display name. This is useful for discovering valid model slugs to pass to `run`, `status`, or `download` commands.

---

### `kaggle benchmarks tasks delete`

Removes a benchmark task.

**Usage:**

```bash
kaggle benchmarks tasks delete <TASK> [options]
```

**Arguments:**

*   `<TASK>`: Task name (slug).

**Options:**

*   `-y, --yes`: Automatically confirm deletion without prompting.

**Example:**

```bash
kaggle b t delete my-task -y
```

**Purpose:**

Deletes a benchmark task and all associated runs. **Note:** This command is not yet supported by the server.
