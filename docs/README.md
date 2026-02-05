# Kaggle CLI Documentation

Welcome to the Kaggle CLI documentation. This guide provides detailed information on how to use the Kaggle command-line interface to interact with Kaggle's platform.

## Installation

Note: Ensure you have Python 3.11+ and the package manager `pip` installed.

Install the `kaggle` package with [pip](https://pypi.org/project/pip/):

```sh
pip install kaggle
```

If you run into a `Command kaggle not found` error, ensure that your Python executable scripts are in your $PATH. For a local user install on Linux, the default location is `~/.local/bin`. On Windows, the default location is `$PYTHON_HOME/Scripts`.

## Authentication

First, you will need a Kaggle account. You can sign up [here](https://www.kaggle.com/account/login).

After login, you can download your Kaggle API credentials at https://www.kaggle.com/settings by clicking on the "Generate New Token" button under the "API" section.

### Option 1: Environment variable

```sh
export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx # Copied from the settings UI
```

### Option 2: API token file

Store your Kaggle API token obtained from your [Kaggle account settings page](https://www.kaggle.com/settings) in a file at `~/.kaggle/access_token`.

### Option 3: Legacy API credentials file

From your [Kaggle account settings page](https://www.kaggle.com/settings), under "Legacy API Credentials", click on the "Create Legacy API Key" button to generate a `kaggle.json` file and store it at `~/.kaggle/kaggle.json`.

## CLI Usage

Run the following command to list the available commands:

```sh
kaggle --help
```

The Kaggle CLI is organized into several command groups:

*   [Competitions](./competitions.md): Manage and participate in Kaggle competitions.
*   [Datasets](./datasets.md): Search, download, and manage Kaggle datasets.
*   [Kernels](./kernels.md): Interact with Kaggle Kernels (notebooks and scripts).
*   [Models](./models.md): Manage your Kaggle Models.
*   [Model Variations](./model_variations.md): Manage variations of your Kaggle Models.
*   [Model Variation Versions](./model_variations_versions.md): Manage versions of your Kaggle Model Variations.
*   [Configuration](./configuration.md): Configure the Kaggle CLI.

## Tutorials

Explore these tutorials to learn how to perform common tasks:

*   [Tutorials](./tutorials.md)
    *   [Create a Dataset](./tutorials.md#tutorial-create-a-dataset)
    *   [Find and Download a Dataset](./tutorials.md#tutorial-find-and-download-a-dataset)
    *   [Create a Model](./tutorials.md#tutorial-create-a-model)
    *   [Create a Model Variation](./tutorials.md#tutorial-create-a-model-variation)
    *   [Create a Model Variation Version](./tutorials.md#tutorial-create-a-model-variation-version)
    *   [How to Submit to a Competition](./tutorials.md#tutorial-how-to-submit-to-a-competition)
    *   [How to Submit to a Code Competition](./tutorials.md#tutorial-how-to-submit-to-a-code-competition)