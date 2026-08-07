# coding=utf-8
import io
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from requests import HTTPError

sys.path.insert(0, "../..")

from kagglesdk.models.types.model_enums import ModelProxyQuotaRefillPeriod

from kaggle.api.kaggle_api_extended import KaggleApi


def _mock_balance(used, total, refill_period, refill_time=None):
    balance = MagicMock()
    balance.quota_used = used
    balance.total_quota_allowed = total
    balance.refill_period = refill_period
    balance.refill_time = refill_time
    return balance


def _build_response(balances):
    response = MagicMock()
    response.quota_balances = balances
    return response


def _http_error(status_code):
    response = MagicMock()
    response.status_code = status_code
    return HTTPError(response=response)


class TestBenchmarksQuota(unittest.TestCase):
    """Tests for the benchmarks_quota and benchmarks_quota_cli methods."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def _patch_client(self, mock_client, quotas_client):
        mock_kaggle = MagicMock()
        mock_kaggle.models.model_proxy_api_client = quotas_client
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_kaggle)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_benchmarks_quota_returns_response(self, mock_client):
        expected = _build_response([_mock_balance(1.2, 5.0, ModelProxyQuotaRefillPeriod.DAILY)])
        quotas_client = MagicMock()
        quotas_client.get_model_proxy_quotas.return_value = expected
        self._patch_client(mock_client, quotas_client)

        result = self.api.benchmarks_quota()

        self.assertIs(result, expected)
        quotas_client.get_model_proxy_quotas.assert_called_once()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_benchmarks_quota_404_raises_friendly_error(self, mock_client):
        quotas_client = MagicMock()
        quotas_client.get_model_proxy_quotas.side_effect = _http_error(404)
        self._patch_client(mock_client, quotas_client)

        with pytest.raises(ValueError, match="Endpoint not found"):
            self.api.benchmarks_quota()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_benchmarks_quota_403_raises_friendly_error(self, mock_client):
        quotas_client = MagicMock()
        quotas_client.get_model_proxy_quotas.side_effect = _http_error(403)
        self._patch_client(mock_client, quotas_client)

        with pytest.raises(ValueError, match="Authentication failed"):
            self.api.benchmarks_quota()

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_benchmarks_quota_other_error_propagates(self, mock_client):
        quotas_client = MagicMock()
        quotas_client.get_model_proxy_quotas.side_effect = _http_error(500)
        self._patch_client(mock_client, quotas_client)

        with pytest.raises(HTTPError):
            self.api.benchmarks_quota()

    def _run_cli(self, **kwargs):
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.api.benchmarks_quota_cli(**kwargs)
        finally:
            sys.stdout = sys.__stdout__
        return captured.getvalue()

    @patch.object(KaggleApi, "benchmarks_quota")
    def test_benchmarks_quota_cli_table(self, mock_view):
        mock_view.return_value = _build_response(
            [
                _mock_balance(
                    1.2,
                    5.0,
                    ModelProxyQuotaRefillPeriod.DAILY,
                    refill_time=datetime(2026, 8, 8, tzinfo=timezone.utc),
                ),
                _mock_balance(
                    14.5,
                    100.0,
                    ModelProxyQuotaRefillPeriod.MONTHLY,
                    refill_time=datetime(2026, 9, 1, tzinfo=timezone.utc),
                ),
            ]
        )

        output = self._run_cli()

        self.assertIn("period", output)
        self.assertIn("refillAt", output)
        self.assertIn("Daily", output)
        self.assertIn("Monthly", output)
        # Remaining is derived: total - used.
        self.assertIn("$3.80", output)
        self.assertIn("$85.50", output)
        self.assertIn("2026-08-08T00:00:00+00:00", output)

    @patch.object(KaggleApi, "benchmarks_quota")
    def test_benchmarks_quota_cli_clamps_negative_remaining(self, mock_view):
        """Overage must not render as a negative remaining balance."""
        mock_view.return_value = _build_response([_mock_balance(7.5, 5.0, ModelProxyQuotaRefillPeriod.DAILY)])

        output = self._run_cli()

        self.assertIn("$0.00", output)
        self.assertNotIn("-$", output)

    @patch.object(KaggleApi, "benchmarks_quota")
    def test_benchmarks_quota_cli_missing_refill_time(self, mock_view):
        mock_view.return_value = _build_response([_mock_balance(1.0, 5.0, ModelProxyQuotaRefillPeriod.DAILY)])

        output = self._run_cli()

        self.assertIn("Daily", output)
        self.assertIn("$4.00", output)

    @patch.object(KaggleApi, "benchmarks_quota")
    def test_benchmarks_quota_cli_unspecified_period(self, mock_view):
        mock_view.return_value = _build_response(
            [_mock_balance(1.0, 5.0, ModelProxyQuotaRefillPeriod.REFILL_PERIOD_UNSPECIFIED)]
        )

        output = self._run_cli()

        self.assertIn("Unknown", output)

    @patch.object(KaggleApi, "benchmarks_quota")
    def test_benchmarks_quota_cli_no_balances(self, mock_view):
        mock_view.return_value = _build_response([])

        output = self._run_cli()

        self.assertIn("No quota information available", output)

    @patch.object(KaggleApi, "benchmarks_quota")
    def test_benchmarks_quota_cli_csv(self, mock_view):
        mock_view.return_value = _build_response([_mock_balance(1.2, 5.0, ModelProxyQuotaRefillPeriod.DAILY)])

        output = self._run_cli(csv_display=True)

        self.assertIn("period,used,remaining,total,refillAt", output)
        self.assertIn("Daily,$1.20,$3.80,$5.00", output)


if __name__ == "__main__":
    unittest.main()
