"""Conservative normalization for raw text extracted from PDF pages."""


def normalize_page_text(text: str, printed_page_label: str | None) -> str:
    """Normalize safe formatting artifacts without rewriting technical content."""
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in normalized_newlines.split("\n")]

    if printed_page_label is not None:
        for index, line in enumerate(lines):
            if line.strip():
                if line == printed_page_label:
                    del lines[index]
                break

    return "\n".join(lines)
