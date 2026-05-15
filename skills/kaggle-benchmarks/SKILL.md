---
name: kaggle-benchmarks
description: >
  How to write, push, run, and manage Kaggle Benchmark tasks using the kaggle
  CLI and the kaggle-benchmarks Python SDK. Activate this skill when the user
  wants to create a benchmark task, push a task file, run benchmarks against
  LLM models, check run status, download results, or troubleshoot benchmark
  workflows. Keywords: kaggle benchmarks, benchmark task, kbench, model proxy,
  push task, run task, benchmark status, benchmark download.
metadata:
  author: kaggle
  version: "0.1"
---

# Kaggle Benchmarks CLI Reference

This reference covers how to use the `kaggle` CLI to manage Kaggle Benchmark tasks — pushing task files, running them against LLM models, checking status, and downloading results.

## Official resources

- **kaggle-benchmarks SDK repo:** https://github.com/Kaggle/kaggle-benchmarks — full source, API reference, and examples for the `kaggle-benchmarks` Python library used to write task files
- **DeepWiki documentation:** https://deepwiki.com/Kaggle/kaggle-benchmarks — auto-generated documentation for the SDK

## Prerequisites

- Python 3.11+
- `kaggle` CLI installed (`pip install kaggle` or `pip install -e .` from source)
- `kaggle-benchmarks` SDK installed (`pip install kaggle-benchmarks`)
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
- `<TASK>` (positional, required): Task slug (must match a `@task` decorator name in the file)
- `-f, --file <FILE>` (required): Path to the `.py` task file

### Step 3: Run the Task

```bash
# Run against the default model
kaggle b t run my-task

# Run against specific models
kaggle b t run my-task -m google/gemini-3-flash-preview -m openai/gpt-4o

# List available models
kaggle b t models
```

### Step 4: Check Status

```bash
# Show task details and per-model run status
kaggle b t status my-task
```

### Step 5: Download Results

```bash
# Download completed run outputs
kaggle b t download my-task

# Download to a specific directory
kaggle b t download my-task -o ./results/
```

## Common Issues & Troubleshooting

- **"No run file produced"**: Ensure your task calls `.run(kbench.llm)` — without it, push succeeds but no results are recorded
- **Token expired**: Re-run `kaggle b auth -y` or `kaggle b init -y` to refresh Model Proxy credentials
- **Task slug mismatch**: The slug in `kaggle b t push <SLUG>` must exactly match a `@task(name="<SLUG>")` decorator in your file
- **Cell format errors**: Ensure `# %%` markers are present — the CLI converts percent-format `.py` to `.ipynb`
