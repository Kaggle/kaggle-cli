# Code Coverage

We measure code coverage using `pytest-cov`. This helps us identify untested parts of the codebase.

## Generating Coverage Reports

To run the unit tests and generate coverage reports, run:

```bash
hatch run test:cov
```

This command will:
1. Run all unit tests.
2. Output a coverage summary to the terminal.
3. Generate an XML report at `coverage.xml`.
4. Generate an HTML report in the `htmlcov/` directory.

The generated report files are configured to be ignored by Git.

---

## Viewing Coverage in Editors

### VSCode

1. Install the [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) extension.
2. Run `hatch run test:cov` to generate `coverage.xml`.
3. Open a source file (e.g., `src/kaggle/cli.py`).
4. Click the **Watch** button in the VSCode status bar, or run `Coverage Gutters: Watch` from the Command Palette.
5. You will see green (covered) and red (uncovered) indicators in the gutter next to line numbers.

### JetBrains Rider

Rider supports coverage via the Python plugin:

#### Option 1: Run with Coverage (Recommended)
1. Configure a Pytest run configuration for your tests.
2. Click the **Run with Coverage** icon (shield) in the top-right toolbar.
3. Rider will run the tests and display coverage gutters automatically.

#### Option 2: Import coverage.xml
1. Run `hatch run test:cov` to generate `coverage.xml`.
2. In Rider, go to **Tools** -> **Show Code Coverage Data**.
3. Click the **Add** (+) button.
4. Select `coverage.xml` from the project root.
5. The coverage data will be overlaid on your source files.
