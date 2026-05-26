# coding=utf-8
import io
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, "..")

from kaggle.api.kaggle_api_extended import KaggleApi


def _mock_quota(used_hours, total_hours):
    quota = MagicMock()
    quota.time_used = timedelta(hours=used_hours)
    quota.total_time_allowed = timedelta(hours=total_hours)
    return quota


def _build_response(gpu=None, tpu=None, refresh_time=None):
    response = MagicMock()
    response.gpu_quota = gpu
    response.tpu_quota = tpu
    response.quota_refresh_time = refresh_time
    return response


class TestQuota(unittest.TestCase):
    """Tests for the quota_view and quota_view_cli methods."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_quota_view_returns_response(self, mock_client):
        expected = _build_response(gpu=_mock_quota(5, 30), tpu=_mock_quota(0, 20))
        mock_kaggle = MagicMock()
        mock_kaggle.kernels.kernels_api_client.get_accelerator_quota_statistics.return_value = expected
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.quota_view()
        self.assertIs(result, expected)

    @patch.object(KaggleApi, "quota_view")
    def test_quota_view_cli_table(self, mock_view):
        mock_view.return_value = _build_response(
            gpu=_mock_quota(5, 30),
            tpu=_mock_quota(2, 20),
            refresh_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.quota_view_cli()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("GPU", output)
        self.assertIn("TPU", output)
        self.assertIn("5.00h", output)
        self.assertIn("25.00h", output)  # GPU remaining: 30 - 5
        self.assertIn("18.00h", output)  # TPU remaining: 20 - 2
        self.assertIn("2026-06-01", output)

    @patch.object(KaggleApi, "quota_view")
    def test_quota_view_cli_csv(self, mock_view):
        mock_view.return_value = _build_response(
            gpu=_mock_quota(5, 30),
            tpu=_mock_quota(2, 20),
        )

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.quota_view_cli(csv_display=True)
        finally:
            sys.stdout = sys.__stdout__

        lines = [line for line in captured.getvalue().splitlines() if line]
        self.assertEqual(lines[0], "resource,used,remaining,total,refreshAt")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("GPU,"))
        self.assertTrue(lines[2].startswith("TPU,"))

    @patch.object(KaggleApi, "quota_view")
    def test_quota_view_cli_skips_missing_accelerator(self, mock_view):
        mock_view.return_value = _build_response(gpu=_mock_quota(1, 30), tpu=None)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.quota_view_cli()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("GPU", output)
        self.assertNotIn("TPU", output)

    @patch.object(KaggleApi, "quota_view")
    def test_quota_view_cli_no_quotas(self, mock_view):
        mock_view.return_value = _build_response(gpu=None, tpu=None)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.quota_view_cli()
        finally:
            sys.stdout = sys.__stdout__

        self.assertIn("No quota information available", captured.getvalue())

    @patch.object(KaggleApi, "quota_view")
    def test_quota_view_cli_clamps_negative_remaining(self, mock_view):
        # User over their quota — remaining should be 0, not negative.
        mock_view.return_value = _build_response(gpu=_mock_quota(35, 30))

        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.quota_view_cli()
        finally:
            sys.stdout = sys.__stdout__

        self.assertIn("0.00h", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
