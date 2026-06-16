# Kernels CLI Reference

Use `kaggle kernels` or alias `kaggle k` to list kernels, inspect output files,
initialize metadata, push notebook/script code, pull code, download output,
inspect run status, stream logs, and delete kernels.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials for push/pull/output/status/logs/delete.
- For push flows, prepare a folder containing `kernel-metadata.json`. Use
  `kaggle kernels init -p <FOLDER>` to generate a starter file.

## Command Hierarchy

```text
kaggle kernels (alias: kaggle k)
├── list
├── files
├── init
├── push | update
├── pull | get
├── output
├── status
├── logs
└── delete
```

Note: `get` is an alias for `pull`; `update` is an alias for `push`.

## `kaggle kernels list`

Lists available kernels.

**Usage:**

```bash
kaggle kernels list [options]
```

**Options:**

- `-m, --mine`: Only your kernels.
- `-p, --page <PAGE>`: Page number.
- `--page-size <SIZE>`: Page size, default 20.
- `-s, --search <TERM>`: Search text.
- `--parent <KERNEL>`: Children of a parent kernel.
- `--competition <SLUG>`: Kernels for a competition.
- `--dataset <OWNER/SLUG>`: Kernels for a dataset.
- `--user <USER>`: Kernels by user.
- `--language <LANG>`: `all`, `python`, `r`, `sqlite`, `julia`.
- `--kernel-type <TYPE>`: `all`, `script`, `notebook`.
- `--output-type <TYPE>`: `all`, `visualization`, `data`.
- `--sort-by <SORT>`: Sort order.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle kernels list
kaggle kernels list --user kaggle --language python --sort-by dateRun
kaggle k list -m -v
```

**Purpose:** Find kernel handles in `<owner>/<kernel-slug>` format.

## `kaggle kernels files`

Lists output files for a kernel.

**Usage:**

```bash
kaggle kernels files [KERNEL] [options]
```

**Options:**

- `--page-size <SIZE>`: Page size.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle kernels files kerneler/sqlite-global-default -v --page-size 1
```

**Purpose:** Inspect generated output before downloading.

## `kaggle kernels init`

Creates a starter `kernel-metadata.json`.

**Usage:**

```bash
kaggle kernels init [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder where metadata will be written.

**Examples:**

```bash
kaggle kernels init -p my-kernel
```

**Purpose:** Bootstrap local metadata for `kaggle kernels push`.

## `kaggle kernels push`

Pushes code to a kernel and runs it.

**Usage:**

```bash
kaggle kernels push [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder containing files and `kernel-metadata.json`.
- `-t, --timeout <SECONDS>`: Limit run time, bounded by Kaggle's maximum.
- `--accelerator <ACCELERATOR>`: Accelerator type for the kernel run.

**Examples:**

```bash
kaggle kernels push -p my-kernel --timeout 3600 --accelerator gpu
kaggle kernels update -p my-kernel
```

**Purpose:** Upload local notebook/script code and create a new kernel version.

## `kaggle kernels pull`

Pulls code from a kernel.

**Usage:**

```bash
kaggle kernels pull [KERNEL] [options]
```

**Options:**

- `-p, --path <PATH>`: Download folder.
- `-w, --wp`: Download to current working path.
- `-m, --metadata`: Generate metadata when pulling.
**Examples:**

```bash
kaggle kernels pull owner/kernel-slug -p pulled
kaggle kernels get owner/kernel-slug -w -m
kaggle k pull owner/kernel-slug/3 -w -m
```

**Purpose:** Retrieve notebook/script source from Kaggle.

**Notes:** Kernel references may include an optional version:
`<owner>/<kernel-name>/<version>`.

## `kaggle kernels output`

Downloads output from the latest kernel run.

**Usage:**

```bash
kaggle kernels output [KERNEL] [options]
```

**Options:**

- `-p, --path <PATH>`: Download folder.
- `-w, --wp`: Download to current working path.
- `-o, --force`: Force download.
- `-q, --quiet`: Suppress progress output.
- `--file-pattern <REGEX>`: Download only matching output files.
- `--page-size <SIZE>`: Output files per page.
- `--page-token <TOKEN>`: Download from one output page.

**Examples:**

```bash
kaggle kernels output owner/kernel-slug -p output
kaggle k output owner/kernel-slug --file-pattern ".*\\.png$"
```

**Purpose:** Retrieve generated files such as submissions, images, or processed
data.

**Notes:** Without `--page-token`, output download scans available pages so
`--file-pattern` can match files beyond the first page.

## `kaggle kernels status`

Displays status of the latest kernel run.

**Usage:**

```bash
kaggle kernels status [KERNEL]
```

**Options:**

- No visible options. If `[KERNEL]` is omitted, local `kernel-metadata.json`
  supplies the kernel reference.

**Examples:**

```bash
kaggle kernels status owner/kernel-slug
```

**Purpose:** Check whether the latest run is queued, running, complete, or
errored.

## `kaggle kernels logs`

Prints execution logs from the latest kernel run.

**Usage:**

```bash
kaggle kernels logs [KERNEL] [options]
```

**Options:**

- `-f, --follow`: Continuously poll and print new log lines.
- `--interval <SECONDS>`: Poll interval for follow mode. Default 5.

**Examples:**

```bash
kaggle kernels logs owner/kernel-slug
kaggle k logs owner/kernel-slug --follow --interval 10
```

**Purpose:** Debug kernel execution from CLI logs.

## `kaggle kernels delete`

Deletes a kernel.

**Usage:**

```bash
kaggle kernels delete <KERNEL> [options]
```

**Options:**

- `-y, --yes`: Skip confirmation.

**Examples:**

```bash
kaggle kernels delete owner/kernel-slug -y
```

**Purpose:** Remove a kernel from Kaggle.

## Error Scenarios And Notes

- Kernel handles are `<owner>/<kernel-name>` or `<owner>/<kernel-name>/<version>`.
- Push requires a metadata file and valid code file path in that metadata.
- `logs --follow` polls repeatedly; use `--interval` to avoid excessive calls.
