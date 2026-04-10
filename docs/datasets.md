# Datasets Commands

Commands for interacting with Kaggle datasets.

## `kaggle datasets list`

Lists available datasets.

**Usage:**

```bash
kaggle datasets list [options]
```

**Options:**

*   `--sort-by <SORT_BY>`: Sort results. Valid options: `hottest`, `votes`, `updated`, `active` (default: `hottest`).
*   `--size <SIZE_CATEGORY>`: DEPRECATED. Use `--min-size` and `--max-size`.
*   `--file-type <FILE_TYPE>`: Filter by file type. Valid options: `all`, `csv`, `sqlite`, `json`, `bigQuery`.
*   `--license <LICENSE_NAME>`: Filter by license. Valid options: `all`, `cc`, `gpl`, `odb`, `other`.
*   `--tags <TAG_IDS>`: Filter by tags (comma-separated tag IDs).
*   `-s, --search <SEARCH_TERM>`: Search term.
*   `-m, --mine`: Display only your datasets.
*   `--user <USER>`: Filter by a specific user or organization.
*   `-p, --page <PAGE>`: Page number for results (default: 1).
*   `-v, --csv`: Print results in CSV format.
*   `--max-size <BYTES>`: Maximum dataset size in bytes.
*   `--min-size <BYTES>`: Minimum dataset size in bytes.

**Examples:**

1.  List your own datasets:

    ```bash
    kaggle datasets list -m
    ```

2.  List CSV datasets, page 2, sorted by last updated, containing "student" in their title, with size between 13000 and 15000 bytes:

    ```bash
    kaggle datasets list --file-type csv --page 2 --sort-by updated -s student --min-size 13000 --max-size 15000
    ```

3.  List datasets with an ODB license, tagged with "internet", and matching the search term "telco":

    ```bash
    kaggle datasets list --license odb --tags internet --search telco
    ```

**Purpose:**

This command helps you find datasets on Kaggle based on various criteria like owner, file type, tags, and size.

## `kaggle datasets files`

Lists files for a specific dataset.

**Usage:**

```bash
kaggle datasets files <DATASET> [options]
```

**Arguments:**

*   `<DATASET>`: Dataset URL suffix in the format `owner/dataset-name` (e.g., `kerneler/brazilian-bird-observation-metadata-from-wikiaves`).

**Options:**

*   `-v, --csv`: Print results in CSV format.
*   `--page-token <PAGE_TOKEN>`: Page token for results paging.
*   `--page-size <PAGE_SIZE>`: Number of items to show on a page (default: 20, max: 200).

**Example:**

List the first 7 files for the dataset `kerneler/brazilian-bird-observation-metadata-from-wikiaves`:

```bash
kaggle datasets files kerneler/brazilian-bird-observation-metadata-from-wikiaves --page-size=7
```

**Purpose:**

Use this command to see the individual files within a dataset before downloading.

## `kaggle datasets download`

Downloads dataset files.

**Usage:**

```bash
kaggle datasets download <DATASET> [options]
```

**Arguments:**

*   `<DATASET>`: Dataset URL suffix (e.g., `willianoliveiragibin/pixar-films`).

**Options:**

*   `-f, --file <FILE_NAME>`: Specific file to download (downloads all if not specified).
*   `-p, --path <PATH>`: Folder to download files to (defaults to current directory).
*   `-w, --wp`: Download files to the current working path.
*   `--unzip`: Unzip the downloaded file (deletes the .zip file afterwards).
*   `-o, --force`: Force download, overwriting existing files.
*   `-q, --quiet`: Suppress verbose output.

**Examples:**

1.  Download all files for the dataset `willianoliveiragibin/pixar-films`:

    ```bash
    kaggle datasets download -d willianoliveiragibin/pixar-films
    ```

2.  Download the dataset `goefft/public-datasets-with-file-types-and-columns`, unzip it into the `tmp` folder, overwriting if necessary, and suppress output:

    ```bash
    kaggle datasets download goefft/public-datasets-with-file-types-and-columns -p tmp --unzip -o -q
    ```

3.  Download the specific file `dataset_results.csv` from `goefft/public-datasets-with-file-types-and-columns` to the current working directory, quietly, and force overwrite:

    ```bash
    kaggle datasets download goefft/public-datasets-with-file-types-and-columns -f dataset_results.csv -w -q -o
    ```

**Purpose:**

This command allows you to retrieve dataset files for local use.

## `kaggle datasets init`

Initializes a metadata file (`dataset-metadata.json`) for creating a new dataset. See [metadata file format](./datasets_metadata.md).

**Usage:**

```bash
kaggle datasets init -p <FOLDER_PATH>
```

**Options:**

*   `-p, --path <FOLDER_PATH>`: The path to the folder where the `dataset-metadata.json` file will be created (defaults to the current directory).

**Example:**

Initialize a dataset metadata file in the `tests/dataset` folder:

```bash
kaggle datasets init -p tests/dataset
```

**Purpose:**

This command creates a template `dataset-metadata.json` file that you need to edit before creating a new dataset on Kaggle. This file contains information like the dataset title, ID (slug), and licenses.

## `kaggle datasets create`

Creates a new dataset on Kaggle.

**Usage:**

```bash
kaggle datasets create -p <FOLDER_PATH> [options]
```

**Options:**

*   `-p, --path <FOLDER_PATH>`: Path to the folder containing the data files and the `dataset-metadata.json` file (defaults to the current directory).
*   `-u, --public`: Make the dataset public (default is private).
*   `-q, --quiet`: Suppress verbose output.
*   `-t, --keep-tabular`: Do not convert tabular files to CSV (default is to convert).
*   `-r, --dir-mode <MODE>`: How to handle directories: `skip` (ignore), `zip` (compressed upload), `tar` (uncompressed upload) (default: `skip`).

**Example:**

Create a new public dataset from the files in `tests/dataset`, quietly, without converting tabular files, and skipping subdirectories. (Assumes `dataset-metadata.json` in `tests/dataset` has been properly edited with title and slug):

```bash
# Example: Edit dataset-metadata.json first
# sed -i 's/INSERT_TITLE_HERE/My Dataset Title/' tests/dataset/dataset-metadata.json
# sed -i 's/INSERT_SLUG_HERE/my-dataset-slug/' tests/dataset/dataset-metadata.json

kaggle datasets create -p tests/dataset --public -q -t -r skip
```

**Purpose:**

This command uploads your local data files and the associated metadata to create a new dataset on Kaggle.

## `kaggle datasets version`

Creates a new version of an existing dataset.

**Usage:**

```bash
kaggle datasets version -p <FOLDER_PATH> -m <VERSION_NOTES> [options]
```

**Options:**

*   `-p, --path <FOLDER_PATH>`: Path to the folder containing the updated data files and `dataset-metadata.json` (defaults to current directory).
*   `-m, --message <VERSION_NOTES>`: (Required) Message describing the new version.
*   `-q, --quiet`: Suppress verbose output.
*   `-t, --keep-tabular`: Do not convert tabular files to CSV.
*   `-r, --dir-mode <MODE>`: Directory handling mode (`skip`, `zip`, `tar`).
*   `-d, --delete-old-versions`: Delete old versions of this dataset.

**Example:**

Create a new version of a dataset using files from `tests/dataset` with version notes "Updated data", quietly, keeping tabular formats, skipping directories, and deleting old versions:

```bash
kaggle datasets version -m "Updated data" -p tests/dataset -q -t -r skip -d
```

**Purpose:**

Use this command to update an existing dataset with new files or metadata changes.

## `kaggle datasets metadata`

Downloads metadata for a dataset or updates existing from local metadata.

**Usage:**

```bash
kaggle datasets metadata <DATASET> [options]
```

**Arguments:**

*   `<DATASET>`: Dataset URL suffix (e.g., `goefft/public-datasets-with-file-types-and-columns`).

**Options:**

*   `-p, --path <PATH>`: Directory to download/update metadata file (`dataset-metadata.json`). Defaults to current working directory.
*   `--update`: Update the existing dataset version's metadata using the contents of the local metadata JSON file. (e.g. "push" from local)

**Example:**

Download metadata for the dataset `goefft/public-datasets-with-file-types-and-columns` into the `tests/dataset` folder:

```bash
kaggle datasets metadata goefft/public-datasets-with-file-types-and-columns -p tests/dataset
```

**Purpose:**

This command allows you to fetch the `dataset-metadata.json` file for an existing dataset, which can be useful for inspection or as a template for creating a new version.

## `kaggle datasets status`

Gets the creation status of a dataset.

**Usage:**

```bash
kaggle datasets status <DATASET>
```

**Arguments:**

*   `<DATASET>`: Dataset URL suffix (e.g., `goefft/public-datasets-with-file-types-and-columns`).

**Example:**

Get the status of the dataset `goefft/public-datasets-with-file-types-and-columns`:

```bash
kaggle datasets status goefft/public-datasets-with-file-types-and-columns
```

**Purpose:**

After creating or updating a dataset, this command helps you check if the process was successful or if there were any issues.

## `kaggle datasets delete`

Deletes a dataset from Kaggle.

**Usage:**

```bash
kaggle datasets delete <DATASET> [options]
```

**Arguments:**

*   `<DATASET>`: Dataset URL suffix (e.g., `username/dataset-slug`).

**Options:**

*   `-y, --yes`: Automatically confirm deletion without prompting.

**Example:**

Delete the dataset `username/dataset-slug` and automatically confirm:

```bash
kaggle datasets delete username/dataset-slug --yes
```

**Purpose:**

This command permanently removes one of your datasets from Kaggle. Use with caution.
