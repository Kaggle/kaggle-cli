# coding=utf-8
"""Allow running the kaggle CLI via ``python -m kaggle``.

This makes it easy to debug in IntelliJ by creating a run configuration
with Module mode set to ``kaggle`` and passing CLI arguments in Parameters.
"""

from kaggle.cli import main

if __name__ == "__main__":
    main()
