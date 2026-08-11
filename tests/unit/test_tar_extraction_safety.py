import os
import shutil
import tempfile
import unittest
import tarfile
import io
from unittest.mock import MagicMock, patch

from kaggle.api.kaggle_api_extended import KaggleApi, safe_extract_tar


class TestSafeExtractTar(unittest.TestCase):
    """Tests for safe_extract_tar, the tarfile.extractall() traversal guard."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Extraction target nested one level under temp_dir, so a
        # "../evil.txt" member has somewhere plausible (temp_dir itself) to
        # escape to if the guard fails.
        self.extract_dir = os.path.join(self.temp_dir, "extract")
        os.makedirs(self.extract_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _build_tar(self, name, data=b"payload"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        return tarfile.open(fileobj=buf, mode="r")

    def test_rejects_path_traversal_member(self):
        t = self._build_tar("../evil.txt")
        try:
            with self.assertRaises((ValueError, tarfile.TarError)):
                safe_extract_tar(t, self.extract_dir)
        finally:
            t.close()

        escaped_path = os.path.join(self.temp_dir, "evil.txt")
        self.assertFalse(os.path.exists(escaped_path))
        self.assertEqual(os.listdir(self.extract_dir), [])

    def test_extracts_well_behaved_member(self):
        t = self._build_tar("safe.txt", data=b"hello")
        try:
            safe_extract_tar(t, self.extract_dir)
        finally:
            t.close()

        extracted = os.path.join(self.extract_dir, "safe.txt")
        self.assertTrue(os.path.exists(extracted))
        with open(extracted, "rb") as f:
            self.assertEqual(f.read(), b"hello")


class TestModelInstanceVersionDownloadTarSafety(unittest.TestCase):
    """Exercises the public model_instance_version_download path (kaggle_api_extended.py:8554)
    to confirm the safe_extract_tar wiring, not just the helper in isolation."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}
        self.api.already_printed_version_warning = True
        self.temp_dir = tempfile.mkdtemp()
        # Extraction target nested one level under temp_dir, so a
        # "../evil.txt" member has somewhere plausible (temp_dir itself) to
        # escape to if the guard fails.
        self.extract_dir = os.path.join(self.temp_dir, "extract")
        os.makedirs(self.extract_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch.object(KaggleApi, "download_needed", return_value=True)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_download_untar_rejects_path_traversal_member(self, mock_client, mock_download_needed):
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            data = b"evil payload"
            info = tarfile.TarInfo(name="../evil.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_bytes = tar_buffer.getvalue()

        def side_effect_download(response, outfile, http_client, quiet, show_progress):
            with open(outfile, "wb") as f:
                f.write(tar_bytes)

        mock_kaggle = MagicMock()
        mock_kaggle.models.model_api_client.download_model_instance_version.return_value = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        version_str = "owner/model/keras/instance/2"

        with patch.object(KaggleApi, "download_file", side_effect=side_effect_download):
            with self.assertRaises(ValueError):
                self.api.model_instance_version_download(version_str, path=self.extract_dir, untar=True)

        escaped_path = os.path.join(self.temp_dir, "evil.txt")
        self.assertFalse(os.path.exists(escaped_path))
        self.assertEqual(os.listdir(self.extract_dir), ["model.tar.gz"])


if __name__ == "__main__":
    unittest.main()
