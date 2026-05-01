# Kaggle Benchmarks CLI Reference

This reference covers how to use the `kaggle` CLI to manage Kaggle Benchmark tasks — pushing task files, running them against LLM models, checking status, and downloading results.

## Prerequisites

- Python 3.11+
- `kaggle` CLI installed (`pip install kaggle` or `pip install -e .` from source)
- Valid Kaggle credentials: `KAGGLE_API_TOKEN` env var, `~/.kaggle/access_token` file, or OAuth via `kaggle auth login`

## Command Hierarchy

```
kaggle benchmarks (alias: kaggle b)
├── auth              — Fetch Model Proxy credentials
├── init              — Fetch credentials + setup local dev environment
└── tasks (alias: t)  — Manage benchmark tasks
    ├── push          — Upload a task from a .py file
    ├── run           — Run a task against model(s)
    ├── list          — List your benchmark tasks
    ├── status        — Show task details and per-model run status
    ├── download      — Download completed run outputs
    ├── models        — List available benchmark models
    └── delete        — Delete a task (not yet supported by server)
```

## Setup & Authentication

### Initialize a Benchmark Project

The `init` command fetches Model Proxy credentials, writes default environment variables, generates a starter example task file, and a syntax reference document.

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

### Fetch Only Auth Credentials

If you just need the Model Proxy token (without the extra env vars and example files):

```bash
# Refresh only the 3 credential variables (MODEL_PROXY_URL, MODEL_PROXY_API_KEY, MODEL_PROXY_EXPIRY_TIME)
kaggle b auth -y

# Or write to a custom env file:
# kaggle b auth -y --env-file custom.env
```

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
@kbench.task(name="my-test-task")
def my_test_task(llm):
    response = llm.prompt("What is 2 + 2?")
    kbench.assertions.assert_in("4", response, expectation="Should contain 4")

my_test_task.run(kbench.llm)
```

**Task name defaults:** If you omit the `name=` argument from `@kbench.task()`, the task name defaults to the function name, title-cased with underscores replaced by spaces. For example, `@kbench.task()` on a function named `my_eval` produces the task name `"My Eval"`, which is slugified to `my-eval`.

**Task file format rules:**
- Must be a `.py` file
- Uses "percent format" — `# %%` cell markers separate notebook cells. Each `# %%` starts a new cell. The CLI converts the file to `.ipynb` using `jupytext` with this format.
- IPython magics (`%`, `!`, `%%`) are stripped during AST validation but kept in the final notebook for server execution
- The task name is normalized to a URL-safe slug (e.g. `"My Test Task"` → `my-test-task`)
- The slug used in the CLI must match a `@task` decorator in the file

### Step 2: Push the Task

```bash
# Push and wait for server-side creation to complete (recommended)
kaggle b t push my-task -f task.py --wait

# Push with timeout (60s) and custom poll interval (5s)
kaggle b t push my-task -f task.py --wait 60 --poll-interval 5

# Push without waiting (fire-and-forget; check status with `kaggle b t status`)
# kaggle b t push my-task -f task.py
```

**Arguments:**
- `<TASK>` (positional, required): Task name/slug
- `-f, --file <FILE>` (required): Path to the `.py` source file

**Options:**
- `--wait [TIMEOUT]`: Wait for creation to complete. `--wait` alone = wait indefinitely. `--wait 60` = timeout after 60s.
- `--poll-interval <SECONDS>`: Seconds between status polls (default: `10`)

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

### Step 3: Run the Task Against Models

```bash
# Run with interactive model selection (paginated picker)
kaggle b t run my-task

# Run against specific models
kaggle b t run my-task -m google/gemini-2.5-pro anthropic/claude-sonnet-4

# Run against a model and wait for completion
kaggle b t run my-task -m google/gemini-2.5-pro --wait

# Run with timeout and custom poll interval
kaggle b t run my-task -m google/gemini-2.5-pro --wait 30 --poll-interval 5
```

**Arguments:**
- `<TASK>` (positional, required): Task name/slug

**Options:**
- `-m, --model <MODEL> [MODEL ...]`: One or more model slugs. If omitted, shows interactive picker.
- `--wait [TIMEOUT]`: Wait for runs to complete. `0` or omit value = indefinite.
- `--poll-interval <SECONDS>`: Seconds between polls (default: `10`)

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

### Step 4: Check Status

```bash
# Full status for a task
kaggle b t status my-task

# Filter to specific models
kaggle b t status my-task -m google/gemini-2.5-pro
kaggle b t status my-task -m google/gemini-2.5-pro anthropic/claude-sonnet-4
```

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

### Step 5: Download Results

```bash
# Download all terminal run outputs (completed and errored)
kaggle b t download my-task

# Download for specific model(s)
kaggle b t download my-task -m google/gemini-2.5-pro

# Download to a custom directory
kaggle b t download my-task -o ./results
```

**Options:**
- `-m, --model <MODEL> [MODEL ...]`: Download only for specific models
- `-o, --output <DIRECTORY>`: Output directory (default: current directory)

**Output directory structure:**
```
<output>/<task>/<model>/<run_id>/
   ├── output files...
```

**Behavior details:**
- Downloads outputs for all runs in a **terminal state** — this includes both `COMPLETED` and `ERRORED` runs (errored runs may still have partial output)
- Downloads zip archives and extracts them automatically
- Already-downloaded runs are skipped: `Skipping gemini-2.5-pro (run 123) — already downloaded to ./my-task/gemini-2.5-pro/123`
- Corrupt zips: Warning printed, raw `.zip` file kept, continues with other models
- No downloadable runs (all still in progress): `No downloadable runs yet — N run(s) still in progress. Use 'kaggle b t status my-task' to check progress.`
- No runs at all: `No runs found for task 'my-task'. Use 'kaggle b t run my-task' to start one.`

## Additional Commands

### List Tasks

```bash
# List all your tasks
kaggle b t list

# Filter by name (regex)
kaggle b t list --name-regex "^math"

# Filter by status
kaggle b t list --status completed

# Combine filters
kaggle b t list --name-regex "^math" --status errored
```

**Status filter values:** `queued`, `running`, `completed`, `errored`

**Output:** Aligned table with columns: Task, Status, Created

### List Available Models

```bash
kaggle b t models
```

**Output:** Table with columns: Slug, Display Name

### Delete a Task

```bash
kaggle b t delete my-task
kaggle b t delete my-task -y   # skip confirmation
```

**Note:** Delete is not yet supported by the server. Currently prints: `Delete is not supported by the server yet.`

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

## Model Slug Format

Models use an `owner/model-name` format:
- `google/gemini-2.5-pro`
- `anthropic/claude-sonnet-4`
- `openai/gpt-oss-120b`

When displaying model names, the owner prefix is stripped for readability (e.g. `gemini-2.5-pro`).

The server may sometimes return model slugs in different formats (e.g. `anthropic/claude-sonnet-4-6@default`). The CLI handles this with client-side normalization, replacing `@` with `-` for matching.

## Common Workflows

### Full End-to-End Workflow

```bash
# 1. Setup
kaggle b init -y

# 2. Write your task in task.py (see task file format above)

# 3. Push
kaggle b t push my-task -f task.py --wait

# 4. Run against models
kaggle b t run my-task -m google/gemini-2.5-pro anthropic/claude-sonnet-4 --wait

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
kaggle b t run my-task -m google/gemini-2.5-pro --wait && \
kaggle b t download my-task -o ./results
```

**⚠ Note:** Local runs use the `LLM_DEFAULT` model from your `.env` file. Server runs use whatever model(s) you specify with `-m`. Behavior may differ between models, so always validate against your target model(s) on the server after local iteration.

### Quick Push-Run-Download

```bash
# Push and wait, then run and wait, all in sequence
kaggle b t push my-task -f task.py --wait && \
kaggle b t run my-task -m google/gemini-2.5-pro --wait && \
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
kaggle b t run d -m google/gemini-3-flash-preview --wait

# Status shows clean table + separate Errors section
kaggle b t status d
```
