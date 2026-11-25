import unittest
from unittest.mock import MagicMock, patch
import os
import json
import tempfile
import shutil
import struct
import zlib

# Ensure parent directory is in path for imports
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_types import DatasetSettings, DatasetSettingsFile, DatasetSettingsFileColumn
from kagglesdk.datasets.types.dataset_api_service import ApiUpdateDatasetMetadataRequest
from kagglesdk.users.types.users_enums import CollaboratorType


def _make_api():
    api = KaggleApi.__new__(KaggleApi)
    api.already_printed_version_warning = True
    api.config_values = {"username": "owner"}
    return api


class TestDatasetMetadataUpdate(unittest.TestCase):
    def setUp(self):
        self.api = _make_api()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_data_succeeds(self, mock_build):
        # Prepare metadata file with 'data'
        metadata = {
            "title": "New Title",
            "data": [
                {
                    "name": "file.csv",
                    "description": "file desc",
                    "columns": [{"name": "col1", "description": "col1 desc", "type": "string"}],
                }
            ],
        }
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        # Assertions
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.assert_called_once()
        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        self.assertIsInstance(call_args, ApiUpdateDatasetMetadataRequest)
        self.assertEqual(call_args.settings.title, "New Title")
        self.assertEqual(len(call_args.settings.data), 1)
        self.assertEqual(call_args.settings.data[0].name, "file.csv")
        self.assertEqual(call_args.settings.data[0].description, "file desc")
        self.assertEqual(len(call_args.settings.data[0].columns), 1)
        self.assertEqual(call_args.settings.data[0].columns[0].name, "col1")
        self.assertEqual(call_args.settings.data[0].columns[0].description, "col1 desc")

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_resources_succeeds(self, mock_build):
        # Prepare metadata file with 'resources'
        metadata = {
            "title": "New Title",
            "resources": [
                {
                    "path": "file.csv",
                    "description": "file desc",
                    "schema": {
                        "fields": [
                            {"name": "col1", "description": "col1 desc", "type": "string"},
                            {"name": "col2", "title": "col2 desc", "type": "integer"},  # test title fallback
                        ]
                    },
                }
            ],
        }
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        # Assertions
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.assert_called_once()
        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        self.assertIsInstance(call_args, ApiUpdateDatasetMetadataRequest)
        self.assertEqual(call_args.settings.title, "New Title")
        self.assertEqual(len(call_args.settings.data), 1)
        self.assertEqual(call_args.settings.data[0].name, "file.csv")
        self.assertEqual(call_args.settings.data[0].description, "file desc")
        self.assertEqual(len(call_args.settings.data[0].columns), 2)
        self.assertEqual(call_args.settings.data[0].columns[0].name, "col1")
        self.assertEqual(call_args.settings.data[0].columns[0].description, "col1 desc")
        self.assertEqual(call_args.settings.data[0].columns[1].name, "col2")
        self.assertEqual(call_args.settings.data[0].columns[1].description, "col2 desc")  # title mapped to description

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_collaborator_converts_role_to_enum(self, mock_build):
        # A single collaborator with a string role should be converted to the CollaboratorType enum.
        metadata = {
            "title": "New Title",
            "collaborators": [{"username": "bob", "role": "writer"}],
        }
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        self.assertEqual(len(call_args.settings.collaborators), 1)
        self.assertEqual(call_args.settings.collaborators[0].username, "bob")
        self.assertEqual(call_args.settings.collaborators[0].role, CollaboratorType.WRITER)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_multiple_collaborators_converts_roles(self, mock_build):
        # Multiple collaborators with mixed-case roles should each be converted independently.
        metadata = {
            "title": "New Title",
            "collaborators": [
                {"username": "bob", "role": "writer"},
                {"username": "alice", "role": "reader"},
                {"username": "carol", "role": "ADMIN"},
            ],
        }
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        collaborators = call_args.settings.collaborators
        self.assertEqual(len(collaborators), 3)
        self.assertEqual(collaborators[0].username, "bob")
        self.assertEqual(collaborators[0].role, CollaboratorType.WRITER)
        self.assertEqual(collaborators[1].username, "alice")
        self.assertEqual(collaborators[1].role, CollaboratorType.READER)
        self.assertEqual(collaborators[2].username, "carol")
        self.assertEqual(collaborators[2].role, CollaboratorType.ADMIN)

    def test_metadata_update_with_invalid_collaborator_role_fails(self):
        # An unrecognized role should raise rather than silently pass a bad value to the SDK.
        metadata = {
            "title": "New Title",
            "collaborators": [{"username": "bob", "role": "not-a-role"}],
        }
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        with self.assertRaises(KeyError):
            self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

    def test_process_column_has_description_uses_description(self):
        col_dict = {"name": "col", "description": "desc", "type": "string"}
        processed = self.api.process_column(col_dict)
        self.assertEqual(processed.description, "desc")

    def test_process_column_has_title_only_falls_back_to_title(self):
        col_dict = {"name": "col", "title": "desc", "type": "string"}
        processed = self.api.process_column(col_dict)
        self.assertEqual(processed.description, "desc")

    def test_process_column_has_both_prefers_description(self):
        col_dict = {"name": "col", "description": "desc", "title": "ignored", "type": "string"}
        processed = self.api.process_column(col_dict)
        self.assertEqual(processed.description, "desc")

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_image_succeeds(self, mock_build, mock_upload_file):
        metadata = {"title": "New Title", "image": "cover.png"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        image_file = os.path.join(self.temp_dir, "cover.png")
        open(image_file, "w").close()

        mock_upload_file_result = MagicMock()
        mock_upload_file_result.token = "image-token-123"
        mock_upload_file.return_value = mock_upload_file_result

        mock_kaggle = MagicMock()
        mock_response = MagicMock()
        mock_response.errors = []
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = mock_response
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        mock_upload_file.assert_called_once()
        self.assertEqual(mock_upload_file.call_args[0][0], "cover.png")
        self.assertEqual(mock_upload_file.call_args[0][1], image_file)

        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.assert_called_once()
        call_args = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        self.assertIsInstance(call_args, ApiUpdateDatasetMetadataRequest)
        self.assertEqual(call_args.settings.title, "New Title")
        self.assertIsNotNone(call_args.settings.image)
        self.assertEqual(call_args.settings.image.token, "image-token-123")
        self.assertEqual(len(call_args.settings.image.crop_rectangles), 2)
        self.assertEqual(call_args.settings.image.crop_rectangles[0].title, "cover image")
        self.assertEqual(call_args.settings.image.crop_rectangles[1].title, "thumbnail")

    def test_metadata_update_with_image_file_not_found_fails(self):
        metadata = {"image": "non-existent.png"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        with self.assertRaises(ValueError) as context:
            self.api.dataset_metadata_update("owner/dataset", self.temp_dir)
        self.assertIn("Image file was not found", str(context.exception))

    def test_metadata_update_with_image_invalid_extension_fails(self):
        metadata = {"image": "cover.txt"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        image_file = os.path.join(self.temp_dir, "cover.txt")
        open(image_file, "w").close()

        with self.assertRaises(ValueError) as context:
            self.api.dataset_metadata_update("owner/dataset", self.temp_dir)
        self.assertIn("Image file requires an extension of", str(context.exception))

    @patch.object(KaggleApi, "_upload_file", return_value=None)
    @patch.object(KaggleApi, "build_kaggle_client")
    def test_metadata_update_with_image_upload_failure_fails(self, mock_build, mock_upload_file):
        metadata = {"image": "cover.png"}
        meta_file = os.path.join(self.temp_dir, "dataset-metadata.json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        image_file = os.path.join(self.temp_dir, "cover.png")
        open(image_file, "w").close()

        with self.assertRaises(ValueError) as context:
            self.api.dataset_metadata_update("owner/dataset", self.temp_dir)
        self.assertIn("Error uploading image file", str(context.exception))

    @patch.object(KaggleApi, "_upload_file")
    @patch.object(KaggleApi, "build_kaggle_client")
    def _crops_for_uploaded_image(self, image_bytes, mock_build, mock_upload_file):
        with open(os.path.join(self.temp_dir, "dataset-metadata.json"), "w") as f:
            json.dump({"title": "New Title"}, f)
        with open(os.path.join(self.temp_dir, "dataset-cover-image.png"), "wb") as f:
            f.write(image_bytes)
        mock_upload_file.return_value = MagicMock(token="image-token-123")
        mock_kaggle = MagicMock()
        mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.return_value = MagicMock(errors=[])
        mock_build.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_build.return_value.__exit__ = MagicMock(return_value=False)

        self.api.dataset_metadata_update("owner/dataset", self.temp_dir)

        request = mock_kaggle.datasets.dataset_api_client.update_dataset_metadata.call_args[0][0]
        return [(r.title, r.left, r.top, r.width, r.height) for r in request.settings.image.crop_rectangles]

    def test_metadata_update_with_minimum_size_image_uses_documented_crops(self):
        self.assertEqual(
            self._crops_for_uploaded_image(_png(560, 280)),
            [("cover image", 0, 0, 560, 280), ("thumbnail", 140, 0, 280, 280)],
        )

    def test_metadata_update_with_large_image_crops_whole_image(self):
        self.assertEqual(
            self._crops_for_uploaded_image(_png(1600, 800)),
            [("cover image", 0, 0, 1600, 800), ("thumbnail", 400, 0, 800, 800)],
        )

    def test_metadata_update_with_unreadable_image_uses_documented_crops(self):
        self.assertEqual(
            self._crops_for_uploaded_image(b"not an image"),
            [("cover image", 0, 0, 560, 280), ("thumbnail", 140, 0, 280, 280)],
        )

    def test_metadata_update_with_one_pixel_wide_image_uses_documented_crops(self):
        self.assertEqual(
            self._crops_for_uploaded_image(_png(1, 1)),
            [("cover image", 0, 0, 560, 280), ("thumbnail", 140, 0, 280, 280)],
        )

    def test_cover_crop_rectangles_are_centered_and_fit_the_image(self):
        def crops(size):
            return [(r.left, r.top, r.width, r.height) for r in KaggleApi._cover_crop_rectangles(size)]

        self.assertEqual(crops((1000, 800)), [(0, 150, 1000, 500), (100, 0, 800, 800)])  # taller than 2:1
        self.assertEqual(crops((2000, 500)), [(500, 0, 1000, 500), (750, 0, 500, 500)])  # wider than 2:1
        self.assertEqual(crops((999, 700)), [(0, 100, 998, 499), (149, 0, 700, 700)])  # odd width stays exactly 2:1
        self.assertEqual(crops((2, 1)), [(0, 0, 2, 1), (0, 0, 1, 1)])  # smallest valid 2:1 image
        self.assertEqual(crops((1, 500)), [(0, 0, 560, 280), (140, 0, 280, 280)])  # too narrow for a positive 2:1 crop
        self.assertEqual(crops((0, 280)), [(0, 0, 560, 280), (140, 0, 280, 280)])
        self.assertEqual(crops(None), [(0, 0, 560, 280), (140, 0, 280, 280)])

    def test_image_dimensions_reads_png_jpeg_and_webp_headers(self):
        jfif = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
        dht = b"\xff\xc4" + struct.pack(">H", 5) + b"\x00" * 3  # a non-frame marker in the 0xC0-0xCF range
        progressive_frame = b"\xff\xc2" + struct.pack(">HBHH", 17, 8, 365, 731) + b"\x00" * 9
        riff = b"RIFF" + struct.pack("<I", 100) + b"WEBP"
        cases = {
            "a.png": _png(731, 365),
            "b.jpg": b"\xff\xd8" + jfif + dht + progressive_frame + b"\xff\xd9",
            "stuffed.jpg": b"\xff\xd8" + jfif + b"\xff\x00" + progressive_frame + b"\xff\xd9",
            "c.webp": riff
            + b"VP8 "
            + struct.pack("<I", 20)
            + b"\x00\x00\x00\x9d\x01\x2a"
            + struct.pack("<HH", 731, 365),
            "d.webp": riff + b"VP8L" + struct.pack("<I", 20) + b"\x2f" + struct.pack("<I", 730 | (364 << 14)),
            "e.webp": riff
            + b"VP8X"
            + struct.pack("<I", 10)
            + b"\x10\x00\x00\x00"
            + (730).to_bytes(3, "little")
            + (364).to_bytes(3, "little"),
        }
        for name, data in cases.items():
            path = os.path.join(self.temp_dir, name)
            with open(path, "wb") as f:
                f.write(data)
            self.assertEqual(KaggleApi._image_dimensions(path), (731, 365), name)

        corrupt = {
            "zero_width.png": b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">IIBBBBB", 0, 365, 8, 2, 0, 0, 0),
            "overflow_width.png": b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">IIBBBBB", 0x80000000, 365, 8, 2, 0, 0, 0),
            "truncated.jpg": b"\xff\xd8\xff\xe0\x00",
            # a segment length below 2 (it counts its own two bytes)
            "short_length.jpg": b"\xff\xd8\xff\xe0\x00\x01" + progressive_frame,
            # an SOF segment too short (< 7 bytes) to hold precision, height and width
            "short_sof.jpg": b"\xff\xd8\xff\xc0\x00\x06\x08\x01\x6d\x02" + progressive_frame,
            # end of image or start of scan before any frame header: stop instead of scanning trailing data
            "eoi_first.jpg": b"\xff\xd8" + jfif + b"\xff\xd9" + progressive_frame,
            "scan_first.jpg": b"\xff\xd8" + jfif + b"\xff\xda" + struct.pack(">H", 8) + b"\x00" * 6 + progressive_frame,
            # DNL-deferred height (Y=0 in SOF) has no usable height in the frame header
            "dnl_zero_height.jpg": b"\xff\xd8\xff\xc0" + struct.pack(">HBHH", 11, 8, 0, 731) + b"\x00" * 3,
            "bad_vp8_sync.webp": riff
            + b"VP8 "
            + struct.pack("<I", 20)
            + b"\x00\x00\x00\x00\x00\x00"
            + struct.pack("<HH", 731, 365),
            "interframe_vp8.webp": riff
            + b"VP8 "
            + struct.pack("<I", 20)
            + b"\x01\x00\x00\x9d\x01\x2a"
            + struct.pack("<HH", 731, 365),
            "zero_vp8.webp": riff
            + b"VP8 "
            + struct.pack("<I", 20)
            + b"\x00\x00\x00\x9d\x01\x2a"
            + struct.pack("<HH", 0, 365),
            "bad_vp8l_sig.webp": riff
            + b"VP8L"
            + struct.pack("<I", 20)
            + b"\x00"
            + struct.pack("<I", 730 | (364 << 14)),
            "truncated_vp8l.webp": riff + b"VP8L" + struct.pack("<I", 20) + b"\x2f",
            "truncated_vp8x.webp": riff + b"VP8X" + struct.pack("<I", 10) + b"\x10\x00",
        }
        for name, data in corrupt.items():
            path = os.path.join(self.temp_dir, name)
            with open(path, "wb") as f:
                f.write(data)
            self.assertIsNone(KaggleApi._image_dimensions(path), name)


def _png(width, height):
    """A valid 1-bit grayscale PNG of the given size."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + b"\x00" * ((width + 7) // 8) for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


if __name__ == "__main__":
    unittest.main()
