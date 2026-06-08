# Files CLI Reference

Use `kaggle files` to upload local files or directories to a Kaggle inbox path.
This command is separate from dataset/model/kernel uploads and uses the inbox
file API.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials.
- Local filesystem paths to upload.

## Command Hierarchy

```text
kaggle files
└── upload | u
```

## `kaggle files upload`

Uploads one or more local paths to the server inbox.

**Usage:**

```bash
kaggle files upload [options] <LOCAL_PATH> [LOCAL_PATH ...]
```

**Arguments:**

- `<LOCAL_PATH>`: One or more local files or directories. Each path creates a
  separate inbox file.

**Options:**

- `-i, --inbox-path <PATH>`: Virtual path on the server where uploaded files are stored.
- `--no-resume`: Skip resumable uploads.
- `--no-compress`: Do not zip directories; upload directories as tar archives.

**Examples:**

```bash
kaggle files upload report.csv
kaggle files upload -i experiments/run-1 output.csv metrics.json
kaggle files upload --no-resume --no-compress local-directory
kaggle files u -i scratch data.zip
```

**Purpose:** Upload arbitrary local files to an inbox location.

## Behavior Details

- The command accepts up to `MAX_NUM_INBOX_FILES_TO_UPLOAD`, currently 1000
  local paths.
- Each provided local path is uploaded independently.
- Directory paths are compressed by default as zip archives.
- With `--no-compress`, directory paths are uploaded using tar mode.
- Uploads use resumable upload machinery unless `--no-resume` is set.
- After uploading each blob, the CLI creates an inbox file and prints
  `Inbox file created: <name>`.

## Error Scenarios And Notes

- Paths that fail upload are skipped by the upload helper; successfully uploaded
  paths still proceed to inbox file creation.
- This command does not create Kaggle Datasets. Use `kaggle datasets create` or
  `kaggle datasets version` for dataset uploads.
- This command does not accept `-q/--quiet`; progress/noise behavior comes from
  the upload behavior.
