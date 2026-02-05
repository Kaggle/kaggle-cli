# Model Variation Commands

Commands for interacting with variations of Kaggle Models. A model variation typically represents a specific framework of a parent model.

## `kaggle models variations init`

Initializes a metadata file (`model-instance-metadata.json`) for creating a new model variation.
Note that the name of the file reflects the old name for a variation, which was "instance".

**Usage:**

```bash
kaggle models variations init -p <FOLDER_PATH>
```

**Options:**

*   `-p, --path <FOLDER_PATH>`: The path to the folder where the `model-instance-metadata.json` file will be created (defaults to the current directory).

**Example:**

Initialize a model variation metadata file in the `tmp` folder:

```bash
kaggle models variations init -p tmp
```

**Purpose:**

This command creates a template `model-instance-metadata.json` file. You must edit this file with details such as the owner slug, the parent model slug, the variation (or instance) slug (URL-friendly name for this variations), and the framework (e.g., `tensorflow`, `pytorch`, `jax`, `sklearn`) before creating the variation.

## `kaggle models variations create`

Creates a new model variation under an existing model on Kaggle.

**Usage:**

```bash
kaggle models variations create -p <FOLDER_PATH> [options]
```

**Options:**

*   `-p, --path <FOLDER_PATH>`: Path to the folder containing the model variation files and the `model-instance-metadata.json` file (defaults to the current directory).
*   `-q, --quiet`: Suppress verbose output.
*   `-r, --dir-mode <MODE>`: How to handle directories within the upload: `skip` (ignore), `zip` (compressed upload), `tar` (uncompressed upload) (default: `skip`).

**Example:**

Create a new model variation using the metadata and files in the `tmp` folder, quietly, skipping subdirectories. (Assumes `model-instance-metadata.json` in `tmp` has been properly edited):

```bash
# Example: Edit model-instance-metadata.json first
# sed -i 's/INSERT_OWNER_SLUG_HERE/your-username/' tmp/model-instance-metadata.json
# sed -i 's/INSERT_EXISTING_MODEL_SLUG_HERE/parent-model-slug/' tmp/model-instance-metadata.json
# sed -i 's/INSERT_INSTANCE_SLUG_HERE/my-variation-slug/' tmp/model-instance-metadata.json
# sed -i 's/INSERT_FRAMEWORK_HERE/jax/' tmp/model-instance-metadata.json
# echo "a,b,c,d" > tmp/data.csv # Example model file

kaggle models variations create -p tmp -q -r skip
```

**Purpose:**

This command uploads your local model files (e.g., weights, architecture definition) and the associated variation metadata to create a new variation under a specified parent model on Kaggle. This effectively creates the first version of this model variation.

## `kaggle models variations get`

Downloads the `model-instance-metadata.json` file for an existing model variation.

**Usage:**

```bash
kaggle models variations get <MODEL_VARIATION> -p <FOLDER_PATH>
```

**Arguments:**

*   `<MODEL_VARIATION>`: Model variation URL suffix in the format `owner/model-slug/framework/variation-slug` (e.g., `$KAGGLE_DEVELOPER/test-model/jax/main`).

**Options:**

*   `-p, --path <FOLDER_PATH>`: Folder to download the `model-instance-metadata.json` file to.

**Example:**

Download the metadata for model variation `$KAGGLE_DEVELOPER/test-model/jax/main` into the `tmp` folder:

```bash
kaggle models variations get $KAGGLE_DEVELOPER/test-model/jax/main -p tmp
```

**Purpose:**

This command retrieves the metadata file for an existing model variation. This can be useful for inspection or as a basis for an update.

## `kaggle models variations files`

Lists files for the current version of a model variation.

**Usage:**

```bash
kaggle models variations files <MODEL_VARIATION> [options]
```

**Arguments:**

*   `<MODEL_VARIATION>`: Model variation URL suffix (e.g., `$KAGGLE_DEVELOPER/test-model/jax/main`).

**Options:**

*   `-v, --csv`: Print results in CSV format.
*   `--page-size <SIZE>`: Number of items per page (default: 20).
*   `--page-token <TOKEN>`: Page token for results paging.

**Example:**

List the first 5 files for the model variation `$KAGGLE_DEVELOPER/test-model/jax/main` in CSV format:

```bash
kaggle models variations files $KAGGLE_DEVELOPER/test-model/jax/main -v --page-size 5
```

**Purpose:**

Use this command to see the files associated with the latest version of a specific model variation.

## `kaggle models variations update`

Updates an existing model variation on Kaggle using a local `model-instance-metadata.json` file.

**Usage:**

```bash
kaggle models variations update -p <FOLDER_PATH>
```

**Options:**

*   `-p, --path <FOLDER_PATH>`: Path to the folder containing the `model-instance-metadata.json` file with the updated information (defaults to the current directory). Note: This command only updates the metadata of the variation, not the files. To update files, create a new version.

**Example:**

Update the model variation whose details are in `tmp/model-instance-metadata.json` (ensure the slugs and owner in the JSON match an existing model variation):

```bash
kaggle models variations update -p tmp
```

**Purpose:**

Use this command to change the metadata of an existing model variation, such as its description or other fields defined in the `model-instance-metadata.json` file. This does not upload new files or create a new version.

## `kaggle models variations delete`

Deletes a model variation from Kaggle.

**Usage:**

```bash
kaggle models variations delete <MODEL_VARIATION> [options]
```

**Arguments:**

*   `<MODEL_VARIATION>`: Model variation URL suffix in the format `owner/model-slug/framework/variation-slug` (e.g., `$KAGGLE_DEVELOPER/test-model/jax/main`).

**Options:**

*   `-y, --yes`: Automatically confirm deletion without prompting.

**Example:**

Delete the model variation `$KAGGLE_DEVELOPER/test-model/jax/main` and automatically confirm:

```bash
kaggle models variations delete $KAGGLE_DEVELOPER/test-model/jax/main -y
```

**Purpose:**

This command permanently removes one of your model variations (and all its versions) from Kaggle. Use with caution.
