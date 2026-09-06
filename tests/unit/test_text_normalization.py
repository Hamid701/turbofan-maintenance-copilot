"""Tests for conservative PDF text normalization."""

from turbofan_copilot.ingestion.text_normalization import normalize_page_text


def test_normalizes_line_endings_trailing_whitespace_and_confirmed_label() -> None:
    raw_text = "6-1  \r\nHeading\t\rBody text  "

    normalized = normalize_page_text(raw_text, printed_page_label="6-1")

    assert normalized == "Heading\nBody text"


def test_preserves_label_that_is_not_the_first_nonempty_line() -> None:
    raw_text = "Heading\n6-1\nmetal- \nto-metal  "

    normalized = normalize_page_text(raw_text, printed_page_label="6-1")

    assert normalized == "Heading\n6-1\nmetal-\nto-metal"


def test_normalization_is_idempotent() -> None:
    raw_text = "6-1\r\nOil cooler  \r\noperating limits\t"

    normalized_once = normalize_page_text(raw_text, printed_page_label="6-1")
    normalized_twice = normalize_page_text(normalized_once, printed_page_label="6-1")

    assert normalized_twice == normalized_once
