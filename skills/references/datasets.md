# Datasets CLI Reference

Use `kaggle datasets` or alias `kaggle d` to search datasets, list and download
files, initialize metadata, create datasets, create versions, update metadata,
check creation status, delete datasets, and browse dataset discussions.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials for create, version, metadata update, delete, and most
  authenticated list/download scenarios.
- For create/version flows, prepare a folder with `dataset-metadata.json`. Use
  `kaggle datasets init -p <FOLDER>` to generate a starter file.

## Command Hierarchy

```text
kaggle datasets (alias: kaggle d)
├── list
├── files
├── download
├── init
├── create
├── version
├── metadata
├── status
├── delete
└── topics
    ├── list
    └── show
```

## `kaggle datasets list`

Lists available datasets.

**Usage:**

```bash
kaggle datasets list [options]
```

**Options:**

- `--sort-by <SORT>`: Sort order.
- `--size <SIZE>`: Deprecated. Use `--max-size` and `--min-size`.
- `--file-type <TYPE>`: File type filter.
- `--license <LICENSE>`: License filter.
- `--tags <TAGS>`: Comma-separated tag ids.
- `-s, --search <TERM>`: Search text.
- `-m, --mine`: Only your datasets.
- `--user <USER>`: Datasets owned by a user or organization.
- `-p, --page <PAGE>`: Page number.
- `--max-size <BYTES>`: Maximum dataset size.
- `--min-size <BYTES>`: Minimum dataset size.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle datasets list
kaggle datasets list -s "bird observation" --file-type csv
kaggle d list --user kaggle --sort-by votes -v
```

**Purpose:** Find dataset handles in `<owner>/<dataset-slug>` format.

## `kaggle datasets files`

Lists files in a dataset.

**Usage:**

```bash
kaggle datasets files [DATASET] [options]
```

**Arguments:**

- `[DATASET]`: Dataset handle in `<owner>/<dataset-name>` format.

**Options:**

- `--page-token <TOKEN>`: Page token.
- `--page-size <SIZE>`: Page size, default 20.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle datasets files kaggle/titanic --page-size 7
```

**Purpose:** Inspect files before downloading.

## `kaggle datasets download`

Downloads dataset files.

**Usage:**

```bash
kaggle datasets download [DATASET] [options]
```

**Options:**

- `-f, --file <NAME>`: Download one file. Downloads all files when omitted.
- `-p, --path <PATH>`: Download directory.
- `-w, --wp`: Download to current working path.
- `--unzip`: Unzip the downloaded archive and delete the zip.
- `-o, --force`: Force download even if local file looks current.
- `-q, --quiet`: Suppress progress output.

**Examples:**

```bash
kaggle datasets download kaggle/titanic
kaggle d download kaggle/titanic -f train.csv -p data --unzip
```

**Purpose:** Retrieve dataset files for local work.

## `kaggle datasets init`

Creates a starter `dataset-metadata.json`.

**Usage:**

```bash
kaggle datasets init [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder to write metadata to. Defaults to current directory.

**Examples:**

```bash
kaggle datasets init -p my-dataset
```

**Purpose:** Bootstrap metadata before creating a dataset.

## `kaggle datasets create`

Creates a new dataset from local files and metadata.

**Usage:**

```bash
kaggle datasets create [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder containing files and `dataset-metadata.json`.
- `-u, --public`: Create publicly. Default is private.
- `-q, --quiet`: Suppress progress output.
- `-t, --keep-tabular`: Do not convert tabular files to CSV.
- `-r, --dir-mode <skip|zip|tar>`: Directory handling. Default `skip`.

**Examples:**

```bash
kaggle datasets create -p my-dataset -u -q -t -r skip
```

**Purpose:** Upload local files and metadata to create a Kaggle dataset.

## `kaggle datasets version`

Creates a new version of an existing dataset.

**Usage:**

```bash
kaggle datasets version -m <MESSAGE> [options]
```

**Options:**

- `-m, --message <MESSAGE>`: Required version notes.
- `-p, --path <FOLDER>`: Folder containing updated files and metadata.
- `-q, --quiet`: Suppress progress output.
- `-t, --keep-tabular`: Do not convert tabular files to CSV.
- `-r, --dir-mode <skip|zip|tar>`: Directory handling. Default `skip`.
- `-d, --delete-old-versions`: Delete old versions of this dataset.

**Examples:**

```bash
kaggle datasets version -p my-dataset -m "Updated data" -q -t -r skip
```

**Purpose:** Publish updated dataset files or metadata as a new version.

## `kaggle datasets metadata`

Downloads or updates dataset metadata.

**Usage:**

```bash
kaggle datasets metadata [DATASET] [options]
```

**Options:**

- `--update`: Update server metadata from local metadata.
- `-p, --path <PATH>`: Metadata folder. Defaults to current directory.

**Examples:**

```bash
kaggle datasets metadata kaggle/titanic -p metadata
kaggle datasets metadata kaggle/titanic --update -p metadata
```

**Purpose:** Retrieve or update `dataset-metadata.json`.

**Notes:** Metadata update also supports selected dataset metadata fields and
dataset cover image files supported by the CLI.

## `kaggle datasets status`

Gets creation status for a dataset.

**Usage:**

```bash
kaggle datasets status [DATASET] [options]
```

**Options:**

- `--format <FORMAT>`: Plain status by default. `json`,
  `json(current_version_number)`, and field selection are supported.

**Examples:**

```bash
kaggle datasets status owner/dataset
kaggle d status owner/dataset --format json
```

**Purpose:** Check whether dataset creation/versioning is complete.

## `kaggle datasets delete`

Deletes a dataset.

**Usage:**

```bash
kaggle datasets delete <DATASET> [options]
```

**Options:**

- `-y, --yes`: Skip confirmation.

**Examples:**

```bash
kaggle datasets delete owner/dataset -y
```

**Purpose:** Remove a dataset from Kaggle.

## Dataset Discussion Commands

### `kaggle datasets topics list`

Lists discussion topics for a dataset.

**Usage:**

```bash
kaggle datasets topics list [DATASET] [options]
```

**Arguments:**

- `[DATASET]`: Dataset handle in `<owner>/<dataset-slug>` format.

**Options:**

- `--sort-by <SORT>`: One of `hot`, `top`, `new`, `recent`, `active`, `relevance`.
- `-s, --search <TERM>`: Search text.
- `--page-size <SIZE>`: Number of topics to return.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle datasets topics list zillow/zecon
kaggle datasets topics list zillow/zecon --sort-by recent --page-size 50
```

**Purpose:** Browse dataset discussions before opening a specific topic.

`kaggle datasets topics` without `list` works as a shortcut for listing
topics.

### `kaggle datasets topics show`

Shows a dataset discussion topic and comments in tree form.

**Usage:**

```bash
kaggle datasets topics show <TOPIC_REF> [TOPIC_ID] [options]
```

**Arguments:**

- `<TOPIC_REF>`: Topic reference, or dataset handle when using the two-argument
  form.
- `[TOPIC_ID]`: Optional topic ID for the two-argument form.

**Options:**

- `--page-size <SIZE>`: Number of comments to return.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle datasets topics show zillow/zecon/12345
kaggle datasets topics show zillow/zecon 12345
```

**Purpose:** Read a dataset discussion topic and its comments.

## Error Scenarios And Notes

- Dataset handles should be `<owner>/<dataset-slug>`.
- Create/version commands require local metadata in the upload folder.
- Directory uploads follow `--dir-mode`; `skip` ignores directories, `zip`
  compresses them, and `tar` uploads an uncompressed archive.
