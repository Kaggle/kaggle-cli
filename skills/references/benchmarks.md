# Kaggle Benchmarks CLI Reference

This reference covers how to use the `kaggle` CLI to manage Kaggle Benchmark tasks — pushing task files, running them against LLM models, checking status, and downloading results.

## Prerequisites

- Python 3.11+
- `kaggle` CLI installed (`pip install kaggle`)
- Valid Kaggle credentials: `KAGGLE_API_TOKEN` env var, `~/.kaggle/access_token` file, or OAuth via `kaggle auth login`

## Command Hierarchy

```
kaggle benchmarks (alias: kaggle b)
├── auth              — Fetch Model Proxy credentials
├── init              — Fetch credentials + setup local dev environment
├── tasks (alias: t)  — Manage benchmark tasks
│   ├── push          — Upload a task from a .py file
│   ├── run           — Run a task against model(s)
│   ├── list          — List your benchmark tasks
│   ├── status        — Show task details and per-model run status
│   ├── download      — Download completed run outputs
│   ├── log / logs    — View execution logs for runs
│   ├── models        — List available benchmark models
│   ├── delete        — Delete a task (not yet supported by server)
│   └── publish       — Publish a task (make it public)
└── topics            — List or show benchmark discussion topics
```

## Setup & Authentication

### `kaggle benchmarks init`

The `init` command fetches Model Proxy credentials, writes default environment variables, generates a starter example task file, and a syntax reference document.

**Usage:**

```bash
kaggle benchmarks init [options]
```

**Examples:**

```bash
# Initialize with defaults (always writes .env, example_task.py, kaggle_benchmarks_reference.md)
kaggle b init -y

# Use custom paths for env file and/or example file:
# kaggle b init -y --env-file my_project/.env --example-file my_project/my_task.py
```

**Options:**
- `-y, --yes`: Skip confirmation prompt
- `--env-file <FILE>`: Path to write env vars (default: `.env`)
- `--example-file <FILE>`: Path to write example task (default: `example_task.py`)

**Purpose:** Prepare a local benchmark project with Model Proxy credentials,
default env vars, and starter benchmark files.

**Environment variables written (appended to the env file):**
- `MODEL_PROXY_URL` — Model Proxy endpoint
- `MODEL_PROXY_API_KEY` — Short-lived API key
- `MODEL_PROXY_EXPIRY_TIME` — Token expiry
- `LLM_DEFAULT` — Default model slug (e.g. `google/gemini-3-flash-preview`)
- `LLM_DEFAULT_EVAL` — Default eval model slug
- `LLMS_AVAILABLE` — Comma-separated list of available model slugs

**⚠ Note:** Environment variables are **appended** to the env file. When loaded via `dotenv`, the last value wins, so re-running `init` or `auth` is safe. The file may accumulate duplicate entries over time; clean up manually if desired.

**Files generated in the same directory as the example file:**
- `example_task.py` — Starter benchmark task using `@task` decorator
- `kaggle_benchmarks_reference.md` — Syntax reference for the `kaggle-benchmarks` Python library

If either file already exists, it is skipped without overwriting.

### `kaggle benchmarks auth`

If you just need the Model Proxy token (without the extra env vars and example files):

**Usage:**

```bash
kaggle benchmarks auth [options]
```

**Options:**

- `-y, --yes`: Skip confirmation prompt.
- `--env-file <FILE>`: Path to append Model Proxy credential variables.

**Examples:**

```bash
# Refresh only the 3 credential variables (MODEL_PROXY_URL, MODEL_PROXY_API_KEY, MODEL_PROXY_EXPIRY_TIME)
kaggle b auth -y

# Or write to a custom env file:
# kaggle b auth -y --env-file custom.env
```

**Purpose:** Refresh only the Model Proxy credential variables without creating
starter task files.

## Core Workflow: Push → Run → Status → Download

### Step 1: Write a Task File

Task files are Python scripts using the `kaggle-benchmarks` library. They must:
- Import `kaggle_benchmarks as kbench`
- Define at least one function decorated with `@kbench.task(...)`
- Call `.run(kbench.llm)` on the task function
- Use `# %%` cell markers to separate notebook cells (percent format)

**⚠ Important:** The `.run()` call is what triggers execution and produces a `.run.json` output file. Without invoking `.run()` (or `.evaluate()`), no run file is produced and nothing is recorded. The push will still succeed (since push validation only checks for `@task` decorators), but the task will silently produce no results when executed on the server.

**Minimal example:**
```python
# %%
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="sample-task")
def sample_task(llm):
    response = llm.prompt("What is 2 + 2?")
    kbench.assertions.assert_in("4", response, expectation="Should contain 4")

sample_task.run(kbench.llm)
```

**Task name defaults:** If you omit the `name=` argument from `@kbench.task()`, the task name defaults to the function name, title-cased with underscores replaced by spaces. For example, `@kbench.task()` on a function named `my_eval` produces the task name `"My Eval"`, which is slugified to `my-eval`.

**Task file format rules:**
- Must be a `.py` file
- Uses "percent format" — `# %%` cell markers separate notebook cells. Each `# %%` starts a new cell. The CLI converts the file to `.ipynb` using `jupytext` with this format.
- IPython magics (`%`, `!`, `%%`) are stripped during AST validation but kept in the final notebook for server execution
- The task name is normalized to a URL-safe slug (e.g. `"My Test Task"` → `my-test-task`)
- The slug used in the CLI must match a `@task` decorator in the file

### `kaggle benchmarks tasks push`

Pushes a benchmark task file to Kaggle.

**Usage:**

```bash
kaggle benchmarks tasks push <TASK> -f <FILE> [options]
```

**Examples:**

```bash
# Push and wait for server-side creation to complete (recommended)
kaggle b t push my-task -f task.py --wait

# Push with timeout (60s) and custom poll interval (5s)
kaggle b t push my-task -f task.py --wait 60 --poll-interval 5

# Push with Kaggle datasets attached
kaggle b t push my-task -f task.py --wait -d kaggle/titanic -d user/my-dataset

# Push without waiting (fire-and-forget; check status with `kaggle b t status`)
# kaggle b t push my-task -f task.py
```

**Arguments:**
- `<TASK>` (positional, required): Task name/slug (e.g. `my-task`)
- `-f, --file <FILE>` (required): Path to the `.py` source file

**Options:**
- `--wait [TIMEOUT]`: Wait for creation to complete. `--wait` alone = wait indefinitely. `--wait 60` = timeout after 60s.
- `--poll-interval <SECONDS>`: Maximum seconds between status polls (default: `60`). Polling starts at 5s and increases by 50% each iteration until reaching this value.
- `-v, --verbose`: Enable verbose polling logs.
- `-d, --kaggle-dataset <DATASET>`: Attach Kaggle dataset to the task's backing notebook (format: `owner/dataset-slug`). Repeat for multiple datasets (e.g. `-d kaggle/titanic -d user/my-dataset`). Mounted at `/kaggle/input/<dataset-slug>/` by default. If a naming conflict occurs, the fully qualified mount path `/kaggle/input/<owner>/<dataset-slug>/` is used instead.

**Purpose:** Create or version a benchmark task from a local `.py` file.

**What happens:**
1. Validates the file is a `.py` file and exists
2. Parses the file AST to verify it contains a `@task` decorator matching the task name
3. If the task name differs from its slug form, prints a warning (e.g. `"My Task"` → `"my-task"`)
4. Converts the `.py` file to `.ipynb` notebook format via `jupytext`
5. Uploads to Kaggle as a benchmark task (creates new or new version if exists)
6. Prints the Task URL and a hint to run

**Error scenarios:**
- File not found: `ValueError: File task.py does not exist`
- Non-`.py` file: `ValueError: File task.txt must be a .py file`
- Missing `@task` decorator: `ValueError: No @task decorators found in file task.py. The file must define at least one task.`
- Task name mismatch: `ValueError: Task 'wrong-name' not found in file task.py. Found tasks: real-task`
- Re-push while previous is still processing (without `--wait`): `ValueError: Task 'my-task' is currently being created (pending). Cannot push now. Use --wait to monitor the existing creation.`
- Re-push with `--wait`: Waits for existing creation to complete, then pushes new version automatically
- Re-push without `-d` when previous version had datasets: Prints yellow warning to stderr: `⚠ Warning: The previous version of 'my-task' had attached Kaggle datasets: ...` and detaches them (re-specify `-d` to keep them).
- Invalid or inaccessible Kaggle dataset: `Failed to push task: Failed to attach the following data sources (not found or inaccessible): <dataset>`


### `kaggle benchmarks tasks run`

Runs a benchmark task against one or more models.

**Usage:**

```bash
kaggle benchmarks tasks run <TASK> [options]
```

**Examples:**

```bash
# Run with interactive model selection (paginated picker)
kaggle b t run my-task

# Run against specific models
kaggle b t run my-task -m gemini-2.5-pro -m claude-sonnet-4

# Run against a model and wait for completion
kaggle b t run my-task -m gemini-2.5-pro --wait

# Run with timeout and custom poll interval
kaggle b t run my-task -m gemini-2.5-pro --wait 30 --poll-interval 5
```

**Arguments:**
- `<TASK>` (positional, required): Task name/slug (e.g. `my-task`)

**Options:**
- `-m, --model <MODEL>` (e.g. `gemini-2.5-pro`): Model slug. Repeat for multiple models (e.g. `-m gemini-2.5-pro -m claude-sonnet-4`). If omitted, shows interactive picker.
- `--wait [TIMEOUT]`: Wait for runs to complete. `0` or omit value = indefinite.
- `--poll-interval <SECONDS>`: Maximum seconds between status polls (default: `60`). Polling starts at 5s and increases by 50% each iteration until reaching this value.
- `-v, --verbose`: Enable verbose polling logs.

**Purpose:** Schedule benchmark runs and optionally wait for completion.

**Interactive model selection:**
- Shows numbered list of available models
- Enter comma-separated numbers (e.g. `1,3,5`) to select specific models
- Enter `all` to select every available model
- Pagination: `n` = next page, `p` = previous page (when > 20 models)

**Error scenarios:**
- Non-existent task: `ValueError: Task 'no-such-task' not found. Check the task name and try again. Use 'kaggle b t list' to see your tasks.`
- Invalid model: `ValueError: Failed to schedule runs. One or more model names may be invalid: ['nonexistent-model']. Use 'kaggle b t run my-task' (without -m) to select from available models.`
- Task not ready: `ValueError: Task 'my-task' is not ready to run (status: QUEUED). Only completed tasks can be run.`
- Timeout: `Timed out waiting for runs after 30 seconds.`

### `kaggle benchmarks tasks status`

Shows task details and per-model run status.

**Usage:**

```bash
kaggle benchmarks tasks status <TASK> [options]
```

**Arguments:**

- `<TASK>` (positional, required): Task name/slug.

**Options:**

- `-m, --model <MODEL>`: Filter status to a specific model. Repeat for
  multiple models.

**Examples:**

```bash
# Full status for a task
kaggle b t status my-task

# Filter to specific models
kaggle b t status my-task -m gemini-2.5-pro
kaggle b t status my-task -m gemini-2.5-pro -m claude-sonnet-4
```

**Purpose:** Check task creation state and model run progress or errors.

**Output format:**
```
Task:     my-task
Status:   COMPLETED
Created:  2026-04-28 18:13:04
Task URL: https://www.kaggle.com/...

Model                     Status      Started               Ended
--------------------------------------------------------------------------
gemini-2.5-pro            COMPLETED   2026-04-28 18:13:04   2026-04-28 18:14:00
claude-sonnet-4           ERRORED     2026-04-28 18:13:04   2026-04-28 18:13:04

Errors:
  [claude-sonnet-4]
    Traceback (most recent call last):
      ...
    ValueError: some error
```

If no runs exist: `No runs yet. Use 'kaggle b t run my-task' to start one.`

### `kaggle benchmarks tasks download`

Downloads completed or errored run outputs.

**Usage:**

```bash
kaggle benchmarks tasks download <TASK> [options]
```

**Arguments:**

- `<TASK>` (positional, required): Task name/slug.

**Examples:**

```bash
# Download all terminal run outputs (completed and errored)
kaggle b t download my-task

# Download for specific model(s)
kaggle b t download my-task -m gemini-2.5-pro

# Download to a custom directory
kaggle b t download my-task -o ./results

# Download with source notebooks included
kaggle b t download my-task --include-source
```

**Options:**
- `-m, --model <MODEL>` (e.g. `gemini-2.5-pro`): Download only for a specific model. Repeat for multiple models.
- `-o, --output <DIRECTORY>`: Output directory (default: current directory)
- `-s, --include-source`: Also download the kernel session's source notebooks (`__notebook__.ipynb`, `__notebook_source__.ipynb`)
- `-f, --force`: Force re-download of already completed runs, overwriting local files

**Purpose:** Retrieve benchmark output artifacts for local inspection.

**Output directory structure:**
```
<output>/<task>/<version>/<model>/<run_id>/    (version is "unset" if unavailable)
   ├── output files...
```

**Behavior details:**
- Downloads outputs for all runs in a **terminal state** — this includes both `COMPLETED` and `ERRORED` runs (errored runs may still have partial output)
- Downloads zip archives and extracts them automatically
- Progress is rendered as a table with `Model | File | Size | Progress` columns. `Size` is the extracted on-disk size of the run's output directory.
- Already-downloaded runs are skipped (use `--force` to re-download) and shown as a `Cached` row in the table
- Corrupt zips show as a `Bad zip` row; the raw `.zip` is kept on disk and the next run is processed
- If `-s` is passed but a cached run's directory lacks source notebooks (`__notebook__.ipynb` / `__notebook_source__.ipynb`), the cached row is left untouched and a tip is printed after the summary: `Tip: N cached run(s) lack source notebooks. Re-run with -f -s to fetch them.` — use `-f -s` together to backfill source notebooks into existing cached runs.
- No downloadable runs (all still in progress): `No downloadable runs yet — N run(s) still in progress. Use 'kaggle b t status my-task' to check progress.`
- No runs at all: `No runs found for task 'my-task'. Use 'kaggle b t run my-task' to start one.`

### `kaggle benchmarks tasks log`

Views logs for benchmark task runs.

**Usage:**

```bash
kaggle benchmarks tasks log <TASK> [options]
```

**Examples:**

```bash
# Show logs for all runs of a task
kaggle b t log my-task

# Show logs for a specific model's run(s)
kaggle b t log my-task -m gemini-2.5-pro

# Show logs for multiple models
kaggle b t logs my-task -m gemini-2.5-pro -m claude-sonnet-4
```

**Arguments:**
- `<TASK>` (positional, required): Task name/slug (e.g. `my-task`)

**Options:**
- `-m, --model <MODEL>` (e.g. `gemini-2.5-pro`): Filter logs to a specific model. Repeat for multiple models. If omitted, logs for all runs are shown.

**Aliases:** `log`, `logs`

**Purpose:** Inspect per-run execution logs for debugging.

**Behavior details:**
- Each run's logs are printed with a header including run state: `═══ Logs for gemini-2.5-pro (Run 456) [COMPLETED] ═══`
- Each run ends with a line count footer: `═══ (42 lines) ═══`
- A summary is printed at the end: `Showed logs for N run(s) across N model(s).`
- Runs are logged **sequentially** in the loop: If the first run is active, the CLI blocks and streams it in real-time until completion before printing or streaming the next run's logs. This prevents terminal log interleaving.
- Active runs: Logs are streamed in real-time via Server-Sent Events (SSE)
- Completed runs: The persisted log file is returned and printed
- No runs found: `No runs found for task 'my-task'. Use 'kaggle b t run my-task' to start one.`

## Additional Commands

### `kaggle benchmarks tasks list`

**Usage:**

```bash
kaggle benchmarks tasks list [options]
```

**Options:**

- `--name-regex <REGEX>`: Filter task names by regular expression.
- `--status <STATUS>`: Filter by creation status: `queued`, `running`, `completed`, `errored`.
- `--page-size <SIZE>`: Tasks per page in the interactive pager. Defaults to 20.
- `--all`: Print every task at once and skip the interactive pager.

**Examples:**

```bash
# List all your tasks
kaggle b t list

# Filter by name (regex)
kaggle b t list --name-regex "^math"

# Filter by status
kaggle b t list --status completed

# Combine filters
kaggle b t list --name-regex "^math" --status errored

# Use a smaller interactive page size
kaggle b t list --page-size 5

# Print all tasks without the pager
kaggle b t list --all
```

**Purpose:** Review owned benchmark tasks and filter by task name or creation state.

**Output:** Aligned table with columns: Task, Version (or `unset`), Status, Created.

### `kaggle benchmarks tasks models`

Lists available benchmark models.

**Usage:**

```bash
kaggle benchmarks tasks models
```

**Options:**

- No visible options.

**Examples:**

```bash
kaggle b t models
```

**Purpose:** Find model slugs accepted by `kaggle benchmarks tasks run`.

**Output:** Table with columns: Slug, Display Name

### `kaggle benchmarks tasks delete`

Deletes a benchmark task when server support is available.

**Usage:**

```bash
kaggle benchmarks tasks delete <TASK> [options]
```

**Arguments:**

- `<TASK>` (positional, required): Task name/slug.

**Options:**

- `-y, --yes`: Skip confirmation.

**Examples:**

```bash
kaggle b t delete my-task
kaggle b t delete my-task -y   # skip confirmation
```

**Purpose:** Request task deletion. Current server behavior does not support it.

**Note:** Delete is not yet supported by the server. Currently prints: `Delete is not supported by the server yet.`

### `kaggle benchmarks tasks publish`

Publishes a benchmark task.

**Usage:**

```bash
kaggle benchmarks tasks publish <TASK> [options]
```

**Arguments:**

- `<TASK>` (positional, required): Task name/slug.

**Examples:**

```bash
# Publish a task and its backing notebook (default)
kaggle b t publish my-task

# Publish without the backing notebook
kaggle b t publish my-task --no-publish-backing-notebook
```

**Options:**
- `--no-publish-backing-notebook`: Do not publish the backing notebook (it is published by default).

**Purpose:** Make a benchmark task public, optionally without publishing its
backing notebook.

**Notes:**
- Idempotent: re-publishing an already-public task prints a message and returns successfully.
- Unpublishing is not supported through this command.
- If task not found, raises `ValueError: Task 'my-task' not found. Check the task name and try again. Use 'kaggle b t list' to see your tasks.`

### `kaggle benchmarks topics list`

Lists discussion topics for a benchmark.

**Usage:**

```bash
kaggle benchmarks topics list [BENCHMARK] [options]
```

**Arguments:**

- `[BENCHMARK]`: Benchmark reference.

**Options:**

- `--sort-by <SORT>`: One of `hot`, `top`, `new`, `recent`, `active`, `relevance`.
- `-s, --search <TERM>`: Search text.
- `--page-size <SIZE>`: Number of topics to return.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle benchmarks topics list my-benchmark
kaggle b topics list my-benchmark --sort-by recent --page-size 50
```

**Purpose:** Browse benchmark discussions before opening a specific topic.

`kaggle benchmarks topics` without `list` works as a shortcut for
listing topics.

### `kaggle benchmarks topics show`

Shows a benchmark discussion topic and comments in tree form.

**Usage:**

```bash
kaggle benchmarks topics show <TOPIC_REF> [TOPIC_ID] [options]
```

**Arguments:**

- `<TOPIC_REF>`: Topic reference, or benchmark reference when using the
  two-argument form.
- `[TOPIC_ID]`: Optional topic ID for the two-argument form.

**Options:**

- `--page-size <SIZE>`: Number of comments to return.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle benchmarks topics show my-benchmark/12345
kaggle benchmarks topics show my-benchmark 12345
```

**Purpose:** Read a benchmark discussion topic and its comments.

## Task Name Normalization

Task names are automatically normalized to URL-safe slugs:
- `my_task` → `my-task`
- `My Test Task` → `my-test-task`
- `My Task` → `my-task`

When the CLI normalizes a name, it prints a yellow warning:
```
⚠ Warning: task name 'My Test Task' was normalized to slug 'my-test-task'.
  Use 'my-test-task' in future commands.
```

The slug must match between the `@task(name=...)` decorator in the file and the CLI command. Comparison is done on slugified names, so `@task(name="My Task")` matches `kaggle b t push my-task -f file.py`.

## Model Slug Normalization

Benchmark model names are automatically normalized on both input and output. This makes it easy to pass various formats interchangeably while keeping displays and directories clean.

- **Flexible Inputs**: The CLI accepts model names in several formats:
  - **Canonical Slugs (recommended)**: `gemini-2.5-pro` or `claude-sonnet-4`
  - **With Provider Prefix**: `google/gemini-2.5-pro` or `anthropic/claude-sonnet-4`
  - **With Version/Proxy `@` symbols**: `anthropic/claude-haiku-4-5@20251001` or `claude-sonnet-4-6@default`
- **Unified Normalization**: The client automatically strips any provider prefix (e.g., `google/` or `anthropic/`) and replaces `@` characters with `-` to match the server's canonical database slug format.
- **Clean Outputs**:
  - **Status Display**: Tables and error logs display the canonical, hyphenated slugs (e.g., `claude-haiku-4-5-20251001` and `gemini-2.0-flash-lite-001`) for readability.
  - **Hierarchical Downloads**: Run outputs are extracted into clean folders using the canonical slugs (e.g., `./<task>/<version>/claude-haiku-4-5-20251001/<run_id>/`), with no `@` or `/` symbols in folder names.

## Common Workflows

### Full End-to-End Workflow

```bash
# 1. Setup
kaggle b init -y

# 2. Write your task in task.py (see task file format above)

# 3. Push
kaggle b t push my-task -f task.py --wait

# 4. Run against models
kaggle b t run my-task -m gemini-2.5-pro -m claude-sonnet-4 --wait

# 5. Check status
kaggle b t status my-task

# 6. Download results
kaggle b t download my-task -o ./results
```

### Local Iteration Loop

Before pushing to the server, you can test your task locally against the Model Proxy to catch errors early. This avoids the push → run → wait → download round-trip for every change.

**1. Get credentials:**
```bash
kaggle b init -y
# or just: kaggle b auth -y
```

**2. Load env vars and run locally:**
```bash
# Source the .env file to set MODEL_PROXY_URL, MODEL_PROXY_API_KEY, etc.
set -a && source .env && set +a

# Run your task file directly with Python
python task.py
```

**3. Check the output:**
- A successful run produces a `.run.json` file in the current directory
- Assertions print pass/fail inline so you can iterate on prompts and thresholds
- Errors surface immediately in your terminal — no need to wait for server execution

**4. Once satisfied, push to the server:**
```bash
kaggle b t push my-task -f task.py --wait && \
kaggle b t run my-task -m gemini-2.5-pro --wait && \
kaggle b t download my-task -o ./results
```

**⚠ Note:** Local runs use the `LLM_DEFAULT` model from your `.env` file. Server runs use whatever model(s) you specify with `-m`. Behavior may differ between models, so always validate against your target model(s) on the server after local iteration.

### Quick Push-Run-Download

```bash
# Push and wait, then run and wait, all in sequence
kaggle b t push my-task -f task.py --wait && \
kaggle b t run my-task -m gemini-2.5-pro --wait && \
kaggle b t download my-task -o ./results
```

### Testing a Task That Intentionally Errors

```python
# t.py
# %%
import kaggle_benchmarks as kbench

# %%
@kbench.task()
def d(llm):
    raise ValueError("intentional error")

# %%
d.run(kbench.llm)
```

```bash
# Push succeeds (error only triggers at run time)
kaggle b t push d -f t.py --wait

# Run — will complete with ERRORED status
kaggle b t run d -m gemini-3-flash-preview --wait

# Status shows clean table + separate Errors section
kaggle b t status d
```
