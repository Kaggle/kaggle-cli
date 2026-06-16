# Model Variations CLI Reference

Use `kaggle models variations` to manage framework-specific variations of a
Kaggle Model. The CLI also accepts legacy aliases
`kaggle models instances`, `kaggle models v`, and `kaggle models i`.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials for private models and mutation commands.
- An existing model for create flows.
- For init/create/update flows, use `model-instance-metadata.json`. Use
  `kaggle models variations init -p <FOLDER>` to generate a starter file.

## Command Hierarchy

```text
kaggle models variations (aliases: instances, v, i)
├── get
├── init
├── create
├── files
├── list
├── update
├── delete
└── versions | v
    ├── list
    ├── create
    ├── download
    ├── files
    └── delete
```

## `kaggle models variations init`

Creates a starter `model-instance-metadata.json`.

**Usage:**

```bash
kaggle models variations init [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder where metadata will be written.

**Examples:**

```bash
kaggle models variations init -p my-model/main
```

**Purpose:** Bootstrap metadata for a model variation.

## `kaggle models variations create`

Creates a new model variation and first version.

**Usage:**

```bash
kaggle models variations create [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder containing files and `model-instance-metadata.json`.
- `-q, --quiet`: Suppress progress output.
- `-r, --dir-mode <skip|zip|tar>`: Directory handling. Default `skip`.

**Examples:**

```bash
kaggle models variations create -p tmp -q -r skip
```

**Purpose:** Upload variation metadata and files under an existing model.

## `kaggle models variations get`

Gets model variation metadata.

**Usage:**

```bash
kaggle models variations get <MODEL_VARIATION> [options]
```

**Arguments:**

- `<MODEL_VARIATION>`: `<owner>/<model>/<framework>/<variation-slug>`.

**Options:**

- `-p, --path <FOLDER>`: Folder to download metadata to.

**Examples:**

```bash
kaggle models variations get google/gemma/pytorch/7b -p metadata
```

**Purpose:** Download metadata for a variation.

## `kaggle models variations files`

Lists files for the current version of a model variation.

**Usage:**

```bash
kaggle models variations files <MODEL_VARIATION> [options]
```

**Options:**

- `--page-size <SIZE>`: Page size, default 20.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle models variations files google/gemma/pytorch/7b -v --page-size 5
```

**Purpose:** Inspect files associated with the latest variation version.

## `kaggle models variations list`

Lists variations of a model.

**Usage:**

```bash
kaggle models variations list <MODEL> [options]
```

**Options:**

- `--page-size <SIZE>`: Page size, default 20.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle models variations list google/gemma
```

**Purpose:** See framework-specific variations under a model.

## `kaggle models variations update`

Updates variation metadata.

**Usage:**

```bash
kaggle models variations update [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder containing updated `model-instance-metadata.json`.

**Examples:**

```bash
kaggle models variations update -p tmp
```

**Purpose:** Change variation metadata. This does not upload new files or create
a version; use `models variations versions create` for file updates. The
variation identity is read from `ownerSlug`, `modelSlug`, `framework`, and
`instanceSlug` inside `model-instance-metadata.json`.

## `kaggle models variations delete`

Deletes a model variation.

**Usage:**

```bash
kaggle models variations delete <MODEL_VARIATION> [options]
```

**Options:**

- `-y, --yes`: Skip confirmation.

**Examples:**

```bash
kaggle models variations delete owner/model/jax/main -y
```

**Purpose:** Remove a model variation.

## Notes

- Variation handles use `<owner>/<model-name>/<framework>/<variation-slug>`.
- `instances` is accepted as a legacy spelling for `variations`.
- Directory uploads follow `--dir-mode`; `skip` ignores directories, `zip`
  compresses them, and `tar` uploads an uncompressed archive.
