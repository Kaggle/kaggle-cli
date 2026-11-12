import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kaggle.api.kaggle_api_extended import KaggleApi

# (cli wrapper, library method it calls, positional args for the wrapper)
WRAPPERS = [
    ("dataset_create_version_cli", "dataset_create_version", ("folder", "notes")),
    ("dataset_create_new_cli", "dataset_create_new", ("folder",)),
    ("kernels_push_cli", "kernels_push", ("folder", None, None)),
    ("model_create_new_cli", "model_create_new", ("folder",)),
    ("model_delete_cli", "model_delete", ("owner/model", True)),
    ("model_update_cli", "model_update", ("folder",)),
    ("model_instance_create_cli", "model_instance_create", ("folder",)),
    ("model_instance_delete_cli", "model_instance_delete", ("o/m/f/i", True)),
    ("model_instance_update_cli", "model_instance_update", ("folder",)),
    ("model_instance_version_create_cli", "model_instance_version_create", ("o/m/f/i", "folder")),
    ("model_instance_version_delete_cli", "model_instance_version_delete", ("o/m/f/i/1", True)),
]


class TestCliWriteErrors(unittest.TestCase):
    """A server-side error must reach cli.main() as ValueError so the command exits 1, not 0."""

    def test_server_error_raises(self):
        api = KaggleApi.__new__(KaggleApi)
        result = SimpleNamespace(error="boom", id=0, status="error", invalidTags=None, ref="", url="")
        for wrapper, method, args in WRAPPERS:
            with self.subTest(wrapper=wrapper), patch.object(KaggleApi, method, return_value=result):
                with self.assertRaisesRegex(ValueError, "error: boom"):
                    getattr(api, wrapper)(*args)


if __name__ == "__main__":
    unittest.main()
