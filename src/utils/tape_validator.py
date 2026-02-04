"""Tape number validation and correction logic."""


def parse_tape(tape_str: str) -> tuple[int, int] | None:
    """Parse tape string 'X/Y' into (X, Y) tuple."""
    if not tape_str:
        return None
    try:
        parts = tape_str.split("/")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except (ValueError, AttributeError):
        pass
    return None


def validate_and_correct_tape(
    ocr_tape: str | None,
    prev_tape_x: int,
    prev_tape_y: int,
    has_end_of_tape: bool
) -> tuple[int, int, bool]:
    """
    Validate OCR-detected tape and correct if needed.

    Args:
        ocr_tape: Tape string from OCR (e.g., "67/1")
        prev_tape_x: Previous tape number (cassette)
        prev_tape_y: Previous page in cassette
        has_end_of_tape: True if this page has END OF TAPE marker

    Returns:
        (tape_x, tape_y, was_corrected): Validated tape numbers and correction flag
    """
    # Calculate expected tape (dead reckoning)
    if has_end_of_tape:
        expected_x = prev_tape_x + 1
        expected_y = 1
    else:
        expected_x = prev_tape_x
        expected_y = prev_tape_y + 1

    # If no OCR tape, use expected
    if not ocr_tape:
        return (expected_x, expected_y, False)

    # Parse OCR tape
    parsed = parse_tape(ocr_tape)
    if not parsed:
        return (expected_x, expected_y, True)  # OCR failed to parse

    ocr_x, ocr_y = parsed

    # Exact match - trust OCR
    if ocr_x == expected_x and ocr_y == expected_y:
        return (ocr_x, ocr_y, False)

    # Special case: END OF TAPE should reset to X+1/1
    if has_end_of_tape:
        if ocr_y == 1 and ocr_x in (expected_x - 1, expected_x, expected_x + 1):
            # OCR says X/1 which makes sense for new tape
            return (ocr_x, 1, False)
        else:
            # OCR is wrong, use expected
            return (expected_x, expected_y, True)

    # Check if OCR is plausible (exact Y match, X may vary by 1)
    # Y must match exactly (pages only increment forward)
    if ocr_y == expected_y and abs(ocr_x - expected_x) <= 1:
        # Y is correct, X is close - trust OCR
        return (ocr_x, ocr_y, False)

    # Check for OCR digit errors in X (only when Y matches expected)
    # Common: first digit wrong (66 → 6, 67 → 6)
    if ocr_y == expected_y:  # Y is correct, only X might have errors
        # Try to fix first digit of X
        if str(ocr_x) in str(expected_x):  # "6" is in "66" or "67"
            # OCR probably dropped a digit - use expected X and Y
            return (expected_x, expected_y, True)

        # Try to fix last digit of X
        ocr_x_str = str(ocr_x)
        expected_x_str = str(expected_x)
        if len(ocr_x_str) == len(expected_x_str):
            # Same number of digits, check if only one digit differs
            diff_count = sum(a != b for a, b in zip(ocr_x_str, expected_x_str))
            if diff_count == 1:
                # Single digit OCR error - use expected X and Y
                return (expected_x, expected_y, True)

    # Check if this is a new tape starting (Y=1)
    if ocr_y == 1 and expected_y == 1:
        # Both OCR and logic agree it's a new tape - validate X
        if expected_x - 1 <= ocr_x <= expected_x + 2:
            # X is close to expected - trust OCR
            return (ocr_x, 1, False)

    # OCR doesn't make sense - use expected (dead reckoning)
    return (expected_x, expected_y, True)


def format_tape(tape_x: int, tape_y: int) -> str:
    """Format tape as 'X/Y' string."""
    return f"{tape_x}/{tape_y}"
