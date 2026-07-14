# Kaggle CLI

The official CLI to interact with [Kaggle](https://www.kaggle.com).

---

[User documentation](docs/README.md)

---

## Key Features

Some of the key features are:

* List competitions, download competition data, submit to a competition.
* List, create, update, download or delete datasets.
* List, create, update, download or delete models & model variations.
* List, update & run, download code & output or delete kernels (notebooks).
* Browse and read discussion forums.

## Installation

Install the `kaggle` package with [pip](https://pypi.org/project/pip/):

```sh
pip install kaggle
```

Additional installation instructions can be found [here](docs/README.md#installation).

## Quick start

Explore the available commands by running:

```sh
kaggle --help
```

See the [User documentation](docs/README.md) for more examples & tutorials.

## Hosting a competition

End-to-end host commands — scaffold a new competition, author its pages, and
launch it — are documented in
[docs/competition_creation.md](docs/competition_creation.md). Covers
`kaggle competitions init`, `create`, `pages create`, and `launch`.

## Development

### `kagglesdk` Updates

New features that interact with `kaggle.com` probably require changes to the Python library, `kagglesdk`.
Make sure to bump the minimum version required for `kagglesdk` in the `dependencies` list specified in
[pyproject.toml][pyproject.toml]]. Make sure the required version is available on the
[pypi.org kagglesdk project](https://pypi.org/project/kagglesdk/#history).

### Prerequisites

We use [hatch](https://hatch.pypa.io) to manage this project.

Follow these [instructions](https://hatch.pypa.io/latest/install/) to install it.

### Run `kaggle` from source

#### Option 1: Execute a one-liner of code from the command line

```sh
hatch run kaggle datasets list
```

#### Option 2: Run many commands in a shell

```sh
hatch shell

# Inside the shell, you can run many commands
kaggle datasets list
kaggle competitions list
...
```

### Lint / Format

```sh
# Lint check
hatch run lint:style
hatch run lint:typing
hatch run lint:all     # for both

# Format
hatch run lint:fmt
```

### Tests

Note: These tests are not true unit tests and are calling the Kaggle web server.

```sh
# Run against kaggle.com
hatch run test:prod

# Run against a local web server (Kaggle engineers only)
hatch run test:local
```

### Integration Tests

To run integration tests on your local machine, you need to set up your Kaggle credentials. You can do this by following the [authentication instructions](docs/README.md#authentication).

After setting up your credentials, you can run the integration tests as follows:

```sh
hatch run test:integration
```

### Code Coverage

We measure code coverage using `pytest-cov`.

To run unit tests with coverage and generate reports:

```sh
hatch run test:cov
```

This generates:
*   Terminal output with a coverage summary.
*   `coverage.xml` (XML report in the root, used by IDE integrations).
*   `htmlcov/index.html` (HTML report for browser viewing).

#### Editor Integration

##### VSCode
Install the **Coverage Gutters** extension. After running the coverage command, click the **Watch** button in the status bar to see coverage indicators in the editor margins.

##### JetBrains Rider
With the **Python** plugin installed:
*   **Run with Coverage:** Create a Pytest run configuration and click the shield icon ("Run with Coverage").
*   **Import Report:** Go to **Tools** -> **Show Code Coverage Data**, click **Add** (+), and select `coverage.xml`.

### Running `hatch` commands inside Docker

This is useful to run in a consistent environment and easily switch between Python versions.

The following shows how to run `hatch run lint:all` but this also works for any other hatch commands:

```
# Use default Python version
./docker-hatch run lint:all
```

## Changelog

See [CHANGELOG](CHANGELOG.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

The Kaggle CLI is released under the [Apache 2.0 license](LICENSE.txt).
