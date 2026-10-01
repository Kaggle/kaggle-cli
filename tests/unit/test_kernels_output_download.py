# coding=utf-8
"""``kernels_output`` saves each file through ``KaggleApi.download_file``, the path dataset and
competition files already take: streamed in chunks instead of held in memory, checked against
Content-Length, and resumed with a Range request when the connection drops.

The output listing is mocked; the file bodies come from a real in-process HTTP server that
advertises ``Accept-Ranges: bytes`` like the signed storage URLs in a kernel's output listing.
"""

import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

sys.path.insert(0, "../..")

from kaggle.api.kaggle_api_extended import KaggleApi


class _Origin:
    def __init__(self, blob):
        self.blob = blob
        self.drop_first_after = None  # bytes sent before the first full GET is cut off
        self.ranges = []  # Range header of every GET, None for a full one


def _make_handler(origin):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            origin.ranges.append(rng)
            start = int(rng[len("bytes=") :].split("-")[0]) if rng else 0
            body = origin.blob[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", "Wed, 01 Jan 2025 00:00:00 GMT")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(origin.blob) - 1}/{len(origin.blob)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not rng and origin.drop_first_after is not None:
                self.wfile.write(body[: origin.drop_first_after])
                self.wfile.flush()
                origin.drop_first_after = None
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            self.wfile.write(body)

    return Handler


class TestKernelsOutputDownload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin = _Origin(b"")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(cls.origin))
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/frames/0001.png"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.origin.blob = os.urandom(3 * 1024 * 1024)
        self.origin.drop_first_after = None
        self.origin.ranges = []
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.config_values = {"username": "testuser"}

    def _kernels_output(self, target_dir):
        listing = MagicMock()
        listing.files = [MagicMock(file_name="frames/0001.png", url=self.url)]
        listing.next_page_token = ""
        listing.log = None
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.list_kernel_session_output.return_value = listing
        with (
            patch.object(KaggleApi, "build_kaggle_client") as mock_client,
            patch.object(KaggleApi, "download_file", autospec=True, side_effect=KaggleApi.download_file) as spy,
            patch("time.sleep"),
        ):
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            outfiles, _ = self.api.kernels_output("owner/kernel-slug", target_dir, quiet=True)
        self.download_file = spy
        return outfiles

    def test_output_file_goes_through_download_file(self):
        with tempfile.TemporaryDirectory() as target_dir:
            outfiles = self._kernels_output(target_dir)
            with open(outfiles[0], "rb") as f:
                self.assertEqual(f.read(), self.origin.blob)
            self.download_file.assert_called_once()
            self.assertEqual(self.origin.ranges, [None])

    def test_dropped_connection_resumes_from_the_bytes_already_written(self):
        # The connection dies halfway through the second 1 MiB chunk: the first chunk is
        # on disk, so the retry asks only for the rest.
        self.origin.drop_first_after = 1536 * 1024
        with tempfile.TemporaryDirectory() as target_dir:
            outfiles = self._kernels_output(target_dir)
            with open(outfiles[0], "rb") as f:
                self.assertEqual(f.read(), self.origin.blob)
            self.assertEqual(self.origin.ranges, [None, f"bytes={1024 * 1024}-"])
            self.assertFalse(os.path.exists(outfiles[0] + ".kaggle-partial"))


if __name__ == "__main__":
    unittest.main()
