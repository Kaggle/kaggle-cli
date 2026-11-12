from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "answer, default_to_yes, expected",
    [
        ("", True, True),
        (" Y \r", True, True),
        ("NO", True, False),
    ],
)
def test_confirmation_single_answer(api, answer, default_to_yes, expected):
    with patch("builtins.input", return_value=answer):
        assert api.confirmation("do it", default_to_yes=default_to_yes) is expected


def test_confirmation_reprompts_after_invalid_answer(api):
    # Without default_to_yes, Enter is not an answer either.
    with patch("builtins.input", side_effect=["", "x", "y"]) as mock_input:
        assert api.confirmation("do it") is True
    assert mock_input.call_count == 3


def test_confirmation_closed_stdin_cancels(api):
    with patch("builtins.input", side_effect=["x", EOFError]):
        assert api.confirmation("do it") is False
