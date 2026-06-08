# Models CLI Reference

Use `kaggle models` or alias `kaggle m` to list, initialize, create, retrieve,
update, and delete Kaggle Model records, and to browse model discussion topics.
Model files normally live under model variations; see the model variation
references for variation and version upload/download flows.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials for create/update/delete and private model access.
- For create/update flows, prepare `model-metadata.json`. Use
  `kaggle models init -p <FOLDER>` to generate a starter file.

## Command Hierarchy

```text
kaggle models (alias: kaggle m)
├── list
├── init
├── create
├── get
├── update
├── delete
├── topics
│   ├── list
│   └── show
└── variations | instances | v | i
```

## `kaggle models list`

Lists public models.

**Usage:**

```bash
kaggle models list [options]
```

**Options:**

- `--sort-by <SORT>`: Sort order.
- `-s, --search <TERM>`: Search text.
- `--owner <OWNER>`: Models owned by a user or organization.
- `--page-size <SIZE>`: Page size.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.

**Examples:**

```bash
kaggle models list
kaggle models list --owner google --sort-by downloadCount
kaggle m list -s gemma -v
```

**Purpose:** Find model handles in `<owner>/<model-name>` format.

## `kaggle models init`

Creates a starter `model-metadata.json`.

**Usage:**

```bash
kaggle models init [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder where metadata will be written.

**Examples:**

```bash
kaggle models init -p my-model
```

**Purpose:** Bootstrap model metadata before creation.

## `kaggle models create`

Creates a new Kaggle Model.

**Usage:**

```bash
kaggle models create [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder containing `model-metadata.json`.

**Examples:**

```bash
kaggle models create -p my-model
```

**Purpose:** Create a model record. Files are managed through variations and
versions.

## `kaggle models get`

Gets model metadata.

**Usage:**

```bash
kaggle models get <MODEL> [options]
```

**Arguments:**

- `<MODEL>`: Model handle in `<owner>/<model-name>` format.

**Options:**

- `-p, --path <FOLDER>`: Folder to download model metadata to.

**Examples:**

```bash
kaggle models get google/gemma -p metadata
```

**Purpose:** Download metadata for an existing model.

## `kaggle models update`

Updates model metadata.

**Usage:**

```bash
kaggle models update [options]
```

**Options:**

- `-p, --path <FOLDER>`: Folder containing updated `model-metadata.json`.

**Examples:**

```bash
kaggle models update -p my-model
```

**Purpose:** Change model metadata without changing variation files. The model
identity is read from `ownerSlug` and `slug` inside `model-metadata.json`.

## `kaggle models delete`

Deletes a model.

**Usage:**

```bash
kaggle models delete <MODEL> [options]
```

**Options:**

- `-y, --yes`: Skip confirmation.

**Examples:**

```bash
kaggle models delete owner/model-slug -y
```

**Purpose:** Remove a model and its associated resources.

## Model Discussion Commands

### `kaggle models topics list`

Lists discussion topics for a model.

**Usage:**

```bash
kaggle models topics list [MODEL] [options]
```

**Arguments:**

- `[MODEL]`: Model handle in `<owner>/<model-slug>` format.

**Options:**

- `--sort-by <SORT>`: One of `hot`, `top`, `new`, `recent`, `active`, `relevance`.
- `-s, --search <TERM>`: Search text.
- `--page-size <SIZE>`: Number of topics to return.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle models topics list owner/model-slug
kaggle models topics list owner/model-slug --sort-by recent --page-size 50
```

**Purpose:** Browse model discussions before opening a specific topic.

`kaggle models topics` without `list` works as a shortcut for listing
topics.

### `kaggle models topics show`

Shows a model discussion topic and comments in tree form.

**Usage:**

```bash
kaggle models topics show <TOPIC_REF> [TOPIC_ID] [options]
```

**Arguments:**

- `<TOPIC_REF>`: Topic reference, or model handle when using the two-argument
  form.
- `[TOPIC_ID]`: Optional topic ID for the two-argument form.

**Options:**

- `--page-size <SIZE>`: Number of comments to return.
- `--page-token <TOKEN>`: Page token.
- `-v, --csv`: Print CSV.
- `-q, --quiet`: Suppress extra output.

**Examples:**

```bash
kaggle models topics show owner/model-slug/12345
kaggle models topics show owner/model-slug 12345
```

**Purpose:** Read a model discussion topic and its comments.

## Notes

- The CLI supports both `models variations` and legacy `models instances`
  naming. They behave as the same model variation command family.
- Use `model_variations.md` for creating and managing framework-specific model
  variations.
