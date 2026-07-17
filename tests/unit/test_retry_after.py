# coding=utf-8
"""Unit tests for Retry-After header handling (issue #938).

These tests use unittest.mock to simulate HTTP 429 responses and verify
that the retry logic respects the Retry-After header.
"""

import unittest
from typing import Any, cast
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import urllib3.exceptions as urllib3_exceptions
from requests.exceptions import ConnectionError, ConnectTimeout, HTTPError
from requests.models import Response

from kaggle.api.kaggle_api_extended import KaggleApi


class TestIsRateLimited(unittest.TestCase):
    """Tests for KaggleApi._is_rate_limited()."""

    def test_returns_true_for_429_http_error(self):
        response = Response()
        response.status_code = 429
        error = HTTPError(response=response)
        self.assertTrue(KaggleApi._is_rate_limited(error))

    def test_returns_false_for_500_http_error(self):
        response = Response()
        response.status_code = 500
        error = HTTPError(response=response)
        self.assertFalse(KaggleApi._is_rate_limited(error))

    def test_returns_false_for_non_http_error(self):
        self.assertFalse(KaggleApi._is_rate_limited(ValueError("oops")))

    def test_returns_false_for_none_response(self):
        error = HTTPError(response=None)
        self.assertFalse(KaggleApi._is_rate_limited(error))


class TestGetRetryAfterDelay(unittest.TestCase):
    """Tests for KaggleApi._get_retry_after_delay()."""

    def _make_response(self, retry_after_value=None):
        response = Response()
        response.status_code = 429
        if retry_after_value is not None:
            response.headers["Retry-After"] = retry_after_value
        return response

    def test_returns_none_when_header_absent(self):
        response = self._make_response()
        self.assertIsNone(KaggleApi._get_retry_after_delay(response))

    def test_parses_integer_seconds(self):
        response = self._make_response("120")
        delay = KaggleApi._get_retry_after_delay(response)
        assert delay is not None
        self.assertAlmostEqual(delay, 120.0)

    def test_parses_float_seconds(self):
        response = self._make_response("30.5")
        delay = KaggleApi._get_retry_after_delay(response)
        assert delay is not None
        self.assertAlmostEqual(delay, 30.5)

    def test_negative_value_clamped_to_zero(self):
        response = self._make_response("-5")
        delay = KaggleApi._get_retry_after_delay(response)
        assert delay is not None
        self.assertAlmostEqual(delay, 0.0)

    def test_parses_http_date(self):
        future = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=60)
        date_str = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        response = self._make_response(date_str)
        delay = KaggleApi._get_retry_after_delay(response)
        assert delay is not None
        # Allow some tolerance for time elapsed during test
        self.assertAlmostEqual(delay, 60.0, delta=2.0)

    def test_past_http_date_clamped_to_zero(self):
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=60)
        date_str = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
        response = self._make_response(date_str)
        delay = KaggleApi._get_retry_after_delay(response)
        assert delay is not None
        self.assertAlmostEqual(delay, 0.0)

    def test_returns_none_for_garbage(self):
        response = self._make_response("not-a-number-or-date")
        self.assertIsNone(KaggleApi._get_retry_after_delay(response))


class TestIsRetriable(unittest.TestCase):
    """Tests for KaggleApi._is_retriable() with rate-limiting."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)

    def test_429_is_retriable(self):
        response = Response()
        response.status_code = 429
        error = HTTPError(response=response)
        self.assertTrue(self.api._is_retriable(error))

    def test_403_is_not_retriable(self):
        response = Response()
        response.status_code = 403
        error = HTTPError(response=response)
        self.assertFalse(self.api._is_retriable(error))

    def test_connection_error_is_retriable(self):
        self.assertTrue(self.api._is_retriable(ConnectionError("boom")))

    def test_connect_timeout_is_retriable(self):
        self.assertTrue(self.api._is_retriable(ConnectTimeout("slow")))

    def test_urllib3_protocol_error_is_retriable(self):
        self.assertTrue(self.api._is_retriable(urllib3_exceptions.ProtocolError("broken")))

    def test_value_error_is_not_retriable(self):
        self.assertFalse(self.api._is_retriable(ValueError("nope")))


class TestWithRetryRateLimiting(unittest.TestCase):
    """Tests for KaggleApi.with_retry() handling 429 responses."""

    def setUp(self):
        self.api = KaggleApi.__new__(KaggleApi)
        self.api.logger = MagicMock()

    def _make_429_error(self, retry_after=None):
        response = Response()
        response.status_code = 429
        if retry_after is not None:
            response.headers["Retry-After"] = str(retry_after)
        error = HTTPError(response=response)
        return error

    @patch("kaggle.api.kaggle_api_extended.time.sleep")
    @patch("builtins.print")
    def test_respects_retry_after_header(self, mock_print, mock_sleep):
        error = self._make_429_error(retry_after=42)

        call_count = 0

        def failing_then_succeeding(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error
            return "success"

        wrapped = self.api.with_retry(failing_then_succeeding, max_retries=3)
        result = wrapped()

        self.assertEqual(result, "success")
        # Should have slept for 42 seconds (from Retry-After)
        mock_sleep.assert_called_once_with(42.0)
        # Logger should have been called with rate-limit info
        logger = cast(Any, self.api.logger)
        logger.info.assert_called()
        log_msg = logger.info.call_args[0][0]
        self.assertIn("Retry-After", log_msg)

    @patch("kaggle.api.kaggle_api_extended.time.sleep")
    @patch("builtins.print")
    def test_falls_back_to_backoff_without_retry_after(self, mock_print, mock_sleep):
        error = self._make_429_error(retry_after=None)

        call_count = 0

        def failing_then_succeeding(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error
            return "success"

        wrapped = self.api.with_retry(failing_then_succeeding, max_retries=3)
        result = wrapped()

        self.assertEqual(result, "success")
        # Should have slept for some backoff delay (not the retry-after value)
        mock_sleep.assert_called_once()
        # Logger should mention missing Retry-After
        logger = cast(Any, self.api.logger)
        log_msg = logger.info.call_args[0][0]
        self.assertIn("No valid Retry-After", log_msg)

    @patch("kaggle.api.kaggle_api_extended.time.sleep")
    @patch("builtins.print")
    def test_connection_error_is_retried(self, mock_print, mock_sleep):
        call_count = 0

        def failing_then_succeeding(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection dropped")
            return "success"

        wrapped = self.api.with_retry(failing_then_succeeding, max_retries=3)
        result = wrapped()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)
        mock_sleep.assert_called_once()

    @patch("kaggle.api.kaggle_api_extended.time.sleep")
    @patch("builtins.print")
    def test_connect_timeout_is_retried(self, mock_print, mock_sleep):
        call_count = 0

        def failing_then_succeeding(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectTimeout("timed out")
            return "success"

        wrapped = self.api.with_retry(failing_then_succeeding, max_retries=3)
        result = wrapped()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)
        mock_sleep.assert_called_once()

    @patch("kaggle.api.kaggle_api_extended.time.sleep")
    @patch("builtins.print")
    def test_urllib3_protocol_error_is_retried(self, mock_print, mock_sleep):
        call_count = 0

        def failing_then_succeeding(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib3_exceptions.ProtocolError("connection aborted")
            return "success"

        wrapped = self.api.with_retry(failing_then_succeeding, max_retries=3)
        result = wrapped()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)
        mock_sleep.assert_called_once()

    @patch("kaggle.api.kaggle_api_extended.time.sleep")
    @patch("builtins.print")
    def test_non_retriable_error_fails_immediately(self, mock_print, mock_sleep):
        call_count = 0

        def always_failing(*args):
            nonlocal call_count
            call_count += 1
            raise ValueError("not retriable")

        wrapped = self.api.with_retry(always_failing, max_retries=3)

        with self.assertRaises(ValueError):
            wrapped()

        # Should not retry or sleep for a non-retriable exception.
        self.assertEqual(call_count, 1)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
