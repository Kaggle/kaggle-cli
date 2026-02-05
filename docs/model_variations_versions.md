# Model Variation Versions Commands

Commands for managing versions of a specific Kaggle Model Variation. Each version represents a snapshot of the model variation files at a point in time.

## `kaggle models variations versions create`

Creates a new version of an existing model variation.

**Usage:**

```bash
kaggle models variations versions create <MODEL_VARIATION> -p <FOLDER_PATH> [options]
```

**Arguments:**

*   `<MODEL_VARIATION>`: The target model variation URL suffix for the new version (format: `owner/model-slug/framework/variation-slug`, e.g., `$KAGGLE_DEVELOPER/test-model/jax/main`).

**Options:**

*   `-p, --path <FOLDER_PATH>`: Path to the folder containing the files for this new version (defaults to the current directory).
*   `-n, --version-notes <NOTES>`: Notes describing this version.
*   `-q, --quiet`: Suppress verbose output.
*   `-r, --dir-mode <MODE>`: How to handle directories within the upload: `skip` (ignore), `zip` (compressed upload), `tar` (uncompressed upload) (default: `skip`).

**Example:**

Create a new version for the model variation `$KAGGLE_DEVELOPER/test-model/jax/main` using files from the `tmp` folder, with version notes "Updated model files", quietly, and skipping subdirectories:

```bash
# Ensure tmp folder contains the new files for the version, e.g., data_v2.csv
# echo "e,f,g,h" > tmp/data_v2.csv

kaggle models variations versions create $KAGGLE_DEVELOPER/test-model/jax/main -p tmp -n "Updated model files" -q -r skip
```

**Purpose:**

This command uploads a new set of files to an existing model variation, creating a new, numbered version. This allows you to track changes and revert to previous versions of your model variation files.

## `kaggle models variations versions download`

Downloads files for a specific version of a model variation.

**Usage:**

```bash
kaggle models variations versions download <MODEL_VARIATION_VERSION> [options]
```

**Arguments:**

*   `<MODEL_VARIATION_VERSION>`: Model variation version URL suffix in the format `owner/model-slug/framework/variation-slug/version-number` (e.g., `$KAGGLE_DEVELOPER/test-model/jax/main/1`).

**Options:**

*   `-p, --path <PATH>`: Folder to download files to (defaults to current directory).
*   `--untar`: Untar the downloaded file if it's a `.tar` archive (deletes the `.tar` file afterwards).
*   `--unzip`: Unzip the downloaded file if it's a `.zip` archive (deletes the `.zip` file afterwards).
*   `-f, --force`: Force download, overwriting existing files.
*   `-q, --quiet`: Suppress verbose output.

**Example:**

Download version 1 of the model variation `$KAGGLE_DEVELOPER/test-model/jax/main` into the `tmp` folder, untar if applicable, force overwrite, and do it quietly:

```bash
kaggle models variations versions download $KAGGLE_DEVELOPER/test-model/jax/main/1 -p tmp -q -f --untar
```

**Purpose:**

This command allows you to retrieve the specific files associated with a particular version of a model variation.

## `kaggle models variations versions files`

Lists files for a specific version of a model variation.

**Usage:**

```bash
kaggle models variations versions files <MODEL_VARIATION_VERSION> [options]
```

**Arguments:**

*   `<MODEL_VARIATION_VERSION>`: Model variation version URL suffix (e.g., `google/gemma/pytorch/7b/2`).

**Options:**

*   `-v, --csv`: Print results in CSV format.
*   `--page-size <SIZE>`: Number of items per page (default: 20).
*   `--page-token <TOKEN>`: Page token for results paging.

**Example:**

List the first 3 files for version 2 of the model variation `google/gemma/pytorch/7b` in CSV format:

```bash
kaggle models variations versions files google/gemma/pytorch/7b/2 -v --page-size=3
```

**Purpose:**

Use this command to see the individual files that constitute a specific version of a model variation before downloading.

## `kaggle models variations versions delete`

Deletes a specific version of a model variation from Kaggle.

**Usage:**

```bash
kaggle models variations versions delete <MODEL_VARIATION_VERSION> [options]
```

**Arguments:**

*   `<MODEL_VARIATION_VERSION>`: Model variation version URL suffix in the format `owner/model-slug/framework/variation-slug/version-number` (e.g., `$KAGGLE_DEVELOPER/test-model/jax/main/1`).

**Options:**

*   `-y, --yes`: Automatically confirm deletion without prompting.

**Example:**

Delete version 1 of the model variation `$KAGGLE_DEVELOPER/test-model/jax/main` and automatically confirm:

```bash
kaggle models variations versions delete $KAGGLE_DEVELOPER/test-model/jax/main/1 -y
```

**Purpose:**

This command permanently removes a specific version of your model variation from Kaggle. Use with caution. If it's the only version, this may lead to the deletion of the model variation itself if no other versions exist.
