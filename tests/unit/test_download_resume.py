# coding=utf-8
"""Regression tests for safe download resume in ``KaggleApi.download_file``.

These guard against the object-identity resume bug where a stale local file
from a *different* version of a remote object was blindly appended to (remote
grew -> silent corruption) or kept as-is (remote shrank -> silent staleness),
because resume was gated only on file size with no identity check.

The tests drive ``download_file`` against a real in-process HTTP server that
behaves like GCS-backed Kaggle downloads: it advertises ``Accept-Ranges: bytes``
and honours both ``Range`` and ``If-Range``.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, "../..")

import requests

from kaggle.api.kaggle_api_extended import KaggleApi


class _Origin:
    """Mutable remote-object state shared with the request handler."""

    def __init__(self, blob=b"", etag=None, last_modified="Wed, 01 Jan 2025 00:00:00 GMT"):
        self.blob = blob
        self.etag = etag
        self.last_modified = last_modified
        self.range_requests = 0  # how many GETs carried a Range header
        self.if_range_seen = []  # If-Range values observed
        self.deny_status = None  # when set, every GET fails with this status


def _make_handler(origin):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _common_headers(self):
            self.send_header("Accept-Ranges", "bytes")
            if origin.etag is not None:
                self.send_header("ETag", origin.etag)
            self.send_header("Last-Modified", origin.last_modified)

        def do_GET(self):
            if origin.deny_status is not None:
                body = b"<?xml version='1.0' encoding='UTF-8'?><Error><Code>SignatureDoesNotMatch</Code></Error>"
                self.send_response(origin.deny_status)
                self.send_header("Content-Type", "application/xml; charset=UTF-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            blob = origin.blob
            total = len(blob)
            rng = self.headers.get("Range")
            if_range = self.headers.get("If-Range")

            if rng and rng.startswith("bytes="):
                origin.range_requests += 1
                origin.if_range_seen.append(if_range)
                start = int(rng[len("bytes=") :].split("-")[0])

                # If-Range miss (object changed): serve the whole current object.
                validators = {v for v in (origin.etag, origin.last_modified) if v}
                if if_range is not None and if_range not in validators:
                    self._send_full(blob)
                    return

                if start >= total:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{total}")
                    self.end_headers()
                    return

                body = blob[start:]
                self.send_response(206)
                self._common_headers()
                self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_full(blob)

        def _send_full(self, blob):
            self.send_response(200)
            self._common_headers()
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    return Handler


class _FailingResponse:
    """A stand-in initial response whose body dies mid-stream (network error)."""

    def __init__(self, url, headers, fail_after):
        self.url = url
        self.headers = headers
        self.status_code = 200
        self._fail_after = fail_after
        self.request = type("R", (), {"method": "GET", "headers": {}})()

    def iter_content(self, chunk_size):
        blob = self.headers["_blob"]
        yield blob[: self._fail_after]
        raise requests.exceptions.ConnectionError("simulated mid-stream failure")


class DownloadResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin = _Origin()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(cls.origin))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.tmpdir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.tmpdir, "data.bin")
        # reset shared origin counters
        self.origin.range_requests = 0
        self.origin.if_range_seen = []
        self.origin.deny_status = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/data.bin"

    def _set_remote(self, blob, etag='"v1"'):
        self.origin.blob = blob
        self.origin.etag = etag

    def _seed_file(self, data):
        with open(self.outfile, "wb") as f:
            f.write(data)

    def _seed_marker(self, validator, size):
        with open(self.api._resume_marker_path(self.outfile), "w") as f:
            json.dump({"validator": validator, "size": size}, f)

    def _run(self, resume=True):
        resp = requests.get(self.url, stream=True)
        self.api.download_file(resp, self.outfile, http_client=None, quiet=True, resume=resume)
        with open(self.outfile, "rb") as f:
            return f.read()

    def _marker_exists(self):
        return os.path.isfile(self.api._resume_marker_path(self.outfile))

    # --- The two reported scenarios ------------------------------------------

    def test_remote_grew_does_not_corrupt_stale_complete_file(self):
        """Remote grew 100->150; a stale complete file with no marker must be
        fully re-downloaded, never appended to."""
        self._set_remote(b"B" * 150)
        self._seed_file(b"A" * 100)  # stale, no marker
        result = self._run()
        self.assertEqual(result, b"B" * 150)
        self.assertEqual(self.origin.range_requests, 0)  # no resume attempted
        self.assertFalse(self._marker_exists())

    def test_remote_shrank_does_not_keep_stale_file(self):
        """Remote shrank 100->80; a stale larger file with no marker must be
        replaced by the new content, not kept."""
        self._set_remote(b"B" * 80)
        self._seed_file(b"A" * 100)  # stale, no marker
        result = self._run()
        self.assertEqual(result, b"B" * 80)
        self.assertFalse(self._marker_exists())

    # --- Legitimate resume is preserved --------------------------------------

    def test_genuine_resume_appends_with_matching_marker(self):
        """A real interrupted partial (with a matching marker) is resumed via a
        Range request, appending only the missing tail."""
        blob = b"B" * 150
        self._set_remote(blob, etag='"v1"')
        self._seed_file(blob[:60])
        self._seed_marker('"v1"', 150)
        result = self._run()
        self.assertEqual(result, blob)
        self.assertEqual(self.origin.range_requests, 1)  # resume happened
        self.assertEqual(self.origin.if_range_seen, ['"v1"'])
        self.assertFalse(self._marker_exists())  # cleared on success

    def test_partial_with_mismatched_marker_restarts(self):
        """A partial whose marker identifies a *different* object is discarded
        and restarted rather than appended to."""
        self._set_remote(b"B" * 150, etag='"v2"')
        self._seed_file(b"A" * 60)  # bytes from an old object
        self._seed_marker('"v1"', 150)  # marker for the OLD object
        result = self._run()
        self.assertEqual(result, b"B" * 150)
        self.assertEqual(self.origin.range_requests, 0)  # no unsafe append
        self.assertFalse(self._marker_exists())

    def test_if_range_miss_falls_back_to_overwrite(self):
        """Marker matches the initial response, but the object changes before the
        Range request; If-Range triggers a 200 and we overwrite instead of append."""
        self._set_remote(b"B" * 150, etag='"v1"')
        self._seed_file(b"B" * 60)
        self._seed_marker('"v1"', 150)
        resp = requests.get(self.url, stream=True)  # captures etag "v1", size 150
        # Object changes (same size, new content + etag) before the resume request.
        self._set_remote(b"C" * 150, etag='"v2"')
        self.api.download_file(resp, self.outfile, http_client=None, quiet=True, resume=True)
        with open(self.outfile, "rb") as f:
            result = f.read()
        self.assertEqual(result, b"C" * 150)  # overwritten, not "BBB...CCC"
        self.assertFalse(self._marker_exists())

    def test_http_error_on_resume_preserves_partial(self):
        """An HTTP error during the Range resume (e.g. 403 once the signed URL
        has expired) must not overwrite the partial file with the error body,
        and must leave resume state usable for a later attempt."""
        blob = b"B" * 2000
        self._set_remote(blob, etag='"v1"')
        self._seed_file(blob[:600])
        self._seed_marker('"v1"', 2000)

        resp = requests.get(self.url, stream=True)  # valid initial response
        self.origin.deny_status = 403  # signed URL expires before the resume

        with self.assertRaises(requests.exceptions.HTTPError):
            self.api.download_file(resp, self.outfile, http_client=None, quiet=True, resume=True, max_retries=0)

        # The partial bytes survive untouched...
        with open(self.outfile, "rb") as f:
            self.assertEqual(f.read(), blob[:600])
        # ...and the marker still describes the real object, not the error body.
        self.assertTrue(self._marker_exists())
        with open(self.api._resume_marker_path(self.outfile)) as f:
            self.assertEqual(json.load(f), {"validator": '"v1"', "size": 2000})

        # A later run against a working URL therefore resumes to correct content.
        self.origin.deny_status = None
        self.assertEqual(self._run(), blob)

    def test_fresh_download_writes_and_clears_marker(self):
        self._set_remote(b"B" * 120)
        result = self._run()
        self.assertEqual(result, b"B" * 120)
        self.assertFalse(self._marker_exists())

    def test_no_resume_when_resume_false(self):
        """resume=False (i.e. --force) always downloads fresh, ignoring markers."""
        self._set_remote(b"B" * 150, etag='"v1"')
        self._seed_file(b"B" * 60)
        self._seed_marker('"v1"', 150)
        result = self._run(resume=False)
        self.assertEqual(result, b"B" * 150)
        self.assertEqual(self.origin.range_requests, 0)

    def test_in_process_retry_resumes(self):
        """A mid-stream network failure resumes from the bytes already written,
        within the same call, using If-Range."""
        blob = b"B" * 150
        self._set_remote(blob, etag='"v1"')
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": "150",
            "ETag": '"v1"',
            "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT",
            "_blob": blob,
        }
        failing = _FailingResponse(self.url, headers, fail_after=60)
        with patch("kaggle.api.kaggle_api_extended.time.sleep"):
            self.api.download_file(failing, self.outfile, http_client=None, quiet=True, resume=True)
        with open(self.outfile, "rb") as f:
            result = f.read()
        self.assertEqual(result, blob)
        self.assertEqual(self.origin.range_requests, 1)  # resumed after failure
        self.assertFalse(self._marker_exists())


class CanResumePartialUnitTest(unittest.TestCase):
    """Fast, server-free unit tests for the resume-eligibility predicate."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.tmpdir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.tmpdir, "data.bin")

    def _seed(self, data, marker=None):
        with open(self.outfile, "wb") as f:
            f.write(data)
        if marker is not None:
            with open(self.api._resume_marker_path(self.outfile), "w") as f:
                json.dump(marker, f)

    def test_true_when_marker_matches_and_partial_shorter(self):
        self._seed(b"x" * 60, {"validator": '"v1"', "size": 150})
        self.assertTrue(self.api._can_resume_partial(self.outfile, True, True, '"v1"', 150))

    def test_false_without_resume(self):
        self._seed(b"x" * 60, {"validator": '"v1"', "size": 150})
        self.assertFalse(self.api._can_resume_partial(self.outfile, False, True, '"v1"', 150))

    def test_false_without_range_support(self):
        self._seed(b"x" * 60, {"validator": '"v1"', "size": 150})
        self.assertFalse(self.api._can_resume_partial(self.outfile, True, False, '"v1"', 150))

    def test_false_without_validator(self):
        self._seed(b"x" * 60, {"validator": '"v1"', "size": 150})
        self.assertFalse(self.api._can_resume_partial(self.outfile, True, True, None, 150))

    def test_false_when_no_marker(self):
        self._seed(b"x" * 60)  # no marker
        self.assertFalse(self.api._can_resume_partial(self.outfile, True, True, '"v1"', 150))

    def test_false_on_validator_mismatch(self):
        self._seed(b"x" * 60, {"validator": '"v1"', "size": 150})
        self.assertFalse(self.api._can_resume_partial(self.outfile, True, True, '"v2"', 150))

    def test_false_on_size_mismatch(self):
        self._seed(b"x" * 60, {"validator": '"v1"', "size": 150})
        self.assertFalse(self.api._can_resume_partial(self.outfile, True, True, '"v1"', 200))

    def test_false_when_partial_not_shorter(self):
        self._seed(b"x" * 150, {"validator": '"v1"', "size": 150})
        self.assertFalse(self.api._can_resume_partial(self.outfile, True, True, '"v1"', 150))

    def test_false_when_file_missing(self):
        self.assertFalse(self.api._can_resume_partial(self.outfile, True, True, '"v1"', 150))

    def test_write_skips_marker_without_validator(self):
        self.api._write_resume_marker(self.outfile, None, 150)
        self.assertFalse(os.path.isfile(self.api._resume_marker_path(self.outfile)))


if __name__ == "__main__":
    unittest.main()
