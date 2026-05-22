#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure we are in the tests directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running fast unit tests with pytest..."
(cd "$SCRIPT_DIR/.." && pytest)

# Run live integration tests only if stdin is a TTY (invoked interactively)
if [ -t 0 ]; then
  echo "Running live integration tests (unit_tests.py)..."
  python3 -m unittest unit_tests.py
fi

echo "All tests passed!"
