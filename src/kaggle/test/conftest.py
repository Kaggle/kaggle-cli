"""Shared test configuration for kaggle CLI tests.

Must be set before importing kaggle, which calls api.authenticate() at
module level.  Fake legacy credentials keep authenticate() off the network;
removing KAGGLE_API_TOKEN prevents _introspect_token() from being called.
We also patch get_access_token_from_env so the ~/.kaggle/access_token file
doesn't trigger token introspection.
"""

import os
from unittest.mock import patch

os.environ.pop("KAGGLE_API_TOKEN", None)
os.environ["KAGGLE_USERNAME"] = "testuser"
os.environ["KAGGLE_KEY"] = "testkey"

with patch("kagglesdk.get_access_token_from_env", return_value=(None, None)):
    import kaggle  # noqa: F401 — triggers authenticate()
