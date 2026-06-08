# Model Variation Versions CLI Reference

Use `kaggle models variations versions` to manage numbered snapshots of a model
variation's files. The CLI also accepts nested alias `v` under variations.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials for private model variations and mutation commands.
- An existing model variation in `<owner>/<model>/<framework>/<variation>` form.

## Command Hierarchy

```text
kaggle models variations versions (alias under variations: v)
├── list
├── create
├── download
├── files
└── delete
```

Note: do not recommend `kaggle models variations versions init`; use
`kaggle models variations init` to create variation metadata.

## `kaggle models variations versions list`

Lists versions for a model variation.

**Usage:**

```bash
kaggle models variations versions list <MODEL_VARIATION> [options]
```

**Options:**

- `--page-size <SIZE>`: Page size, default 20.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle models variations versions list google/gemma/pytorch/7b -v
```

**Purpose:** Inspect available numbered versions before selecting one.

## `kaggle models variations versions create`

Creates a new version for a model variation.

**Usage:**

```bash
kaggle models variations versions create <MODEL_VARIATION> [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder containing files for the new version.
- `-n, --version-notes <NOTES>`: Version notes.
- `-q, --quiet`: Suppress progress output.
- `-r, --dir-mode <skip|zip|tar>`: Directory handling. Default `skip`.

**Examples:**

```bash
kaggle models variations versions create owner/model/jax/main -p tmp -n "Updated files" -q -r skip
```

**Purpose:** Upload a new file snapshot for an existing variation.

## `kaggle models variations versions download`

Downloads files for a specific variation version.

**Usage:**

```bash
kaggle models variations versions download <MODEL_VARIATION_VERSION> [options]
```

**Arguments:**

- `<MODEL_VARIATION_VERSION>`:
  `<owner>/<model>/<framework>/<variation>/<version-number>`.

**Options:**

- `-p, --path <PATH>`: Download folder.
- `--untar`: Untar the downloaded archive and delete the tar file.
- `-f, --force`: Force download.
- `-q, --quiet`: Suppress progress output.

**Examples:**

```bash
kaggle models variations versions download google/gemma/pytorch/7b/2 -p models --untar
```

**Purpose:** Retrieve files for a specific version.

## `kaggle models variations versions files`

Lists files for a specific variation version.

**Usage:**

```bash
kaggle models variations versions files <MODEL_VARIATION_VERSION> [options]
```

**Options:**

- `--page-size <SIZE>`: Page size, default 20.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle models variations versions files google/gemma/pytorch/7b/2 -v --page-size 3
```

**Purpose:** Inspect files inside a numbered version.

## `kaggle models variations versions delete`

Deletes a model variation version.

**Usage:**

```bash
kaggle models variations versions delete <MODEL_VARIATION_VERSION> [options]
```

**Options:**

- `-y, --yes`: Skip confirmation.

**Examples:**

```bash
kaggle models variations versions delete owner/model/jax/main/2 -y
```

**Purpose:** Remove a specific version snapshot.

## Notes

- Version handles use
  `<owner>/<model-name>/<framework>/<variation-slug>/<version-number>`.
- Directory uploads follow `--dir-mode`; `skip` ignores directories, `zip`
  compresses them, and `tar` uploads an uncompressed archive.
