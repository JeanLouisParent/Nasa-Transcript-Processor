"""
Cleaning functions for OCR blocks.
"""
import re
from .patterns import (
    SPEAKER_TOKEN_RE,
    ANNOTATION_FRAGMENT_RE,
    PAREN_RADIO_CALL_RE
)
from .utils import (
    clean_leading_footer_noise,
    is_garbled_page_tape_label,
)

EMBEDDED_TIMESTAMP_RE = re.compile(
    r"\(?\b[0-9OIil]{1,2}[:\s-][0-9OIil]{1,2}[:\s-][0-9OIil]{1,2}[ .:\-][0-9OIil]{1,2}\b\)?"
)

def _normalize_embedded_timestamp(raw: str) -> str | None:
    """
    Normalizes a potentially corrupted OCR timestamp fragment into DD HH MM SS format.

    Args:
        raw: The raw text fragment containing potential digits.

    Returns:
        Formatted timestamp string or None if insufficient digits are found.
    """
    chars = re.findall(r"[0-9OIilCSB]", raw)
    if len(chars) < 8:
        return None
    digits: list[str] = []
    for c in chars[:8]:
        if c in ("O", "o", "C", "c"):
            digits.append("0")
        elif c in ("I", "i", "l"):
            digits.append("1")
        elif c in ("S", "s"):
            digits.append("5")
        elif c in ("B", "b"):
            digits.append("8")
        else:
            digits.append(c)
    parts = ["".join(digits[i:i + 2]) for i in range(0, 8, 2)]
    return " ".join(parts)


def split_embedded_timestamp_blocks(blocks: list[dict]) -> list[dict]:
    """
    Identifies and extracts timestamps accidentally merged into communication text.

    Useful for fixing OCR artifacts where a new line's timestamp is joined 
    to the end of the previous line. Skips timestamps part of TIG declarations.

    Args:
        blocks: List of communication block dictionaries.

    Returns:
        Flattened list of blocks with embedded segments split into new blocks.
    """
    output: list[dict] = []
    for block in blocks:
        text = block.get("text")
        if not text:
            output.append(block)
            continue

        matches = list(EMBEDDED_TIMESTAMP_RE.finditer(text))
        if not matches:
            output.append(block)
            continue

        # Filter out matches that are part of TIG declarations
        filtered_matches = []
        for match in matches:
            start = match.start()
            prefix_start = max(0, start - 10)
            prefix = text[prefix_start:start]
            if not re.search(r'\bTIG\s+\d{1,3}\s*$', prefix, re.IGNORECASE):
                filtered_matches.append(match)

        matches = filtered_matches
        if not matches:
            output.append(block)
            continue

        prefix = text[:matches[0].start()].strip()
        if prefix:
            base_block = dict(block)
            base_block["text"] = prefix
            output.append(base_block)

        for idx, match in enumerate(matches):
            ts = _normalize_embedded_timestamp(match.group(0))
            speaker = None
            seg_start = match.end()
            if not ts:
                continue
            seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            seg_text = text[seg_start:seg_end].strip()

            location = None
            if seg_text:
                tokens = seg_text.split()
                if speaker is None and tokens and SPEAKER_TOKEN_RE.match(tokens[0]):
                    candidate_speaker = tokens[0].rstrip(":")
                    if not re.match(r'^(NOUN|VERB)\s*\d', candidate_speaker, re.IGNORECASE):
                        speaker = candidate_speaker
                        seg_text = " ".join(tokens[1:]).strip()
                    else:
                        pass
                if seg_text.startswith("("):
                    loc_match = re.match(r"^\(([^)]+)\)\s*(.*)$", seg_text)
                    if loc_match:
                        location = loc_match.group(1).strip().upper()
                        seg_text = loc_match.group(2).strip()

            seg_text = seg_text.lstrip(":-").strip()

            new_block = {"type": "comm", "timestamp": ts}
            if speaker:
                new_block["speaker"] = speaker
            if location:
                new_block["location"] = location
            if seg_text:
                new_block["text"] = seg_text
            output.append(new_block)

    return output


def merge_duplicate_comm_timestamps(blocks: list[dict]) -> list[dict]:
    """
    Merges consecutive blocks that share the same timestamp.

    Consolidates text fragments and metadata when the OCR engine splits 
    a single dialogue line into multiple segments with identical timecodes.

    Args:
        blocks: List of communication blocks.

    Returns:
        List of blocks with consecutive duplicates merged.
    """
    def merge_text(prev_text: str, new_text: str) -> str:
        if not prev_text:
            return new_text
        if not new_text:
            return prev_text
        if new_text in prev_text:
            return prev_text
        if prev_text in new_text:
            return new_text
        prev_words = prev_text.split()
        new_words = new_text.split()
        max_k = min(12, len(prev_words), len(new_words))
        for k in range(max_k, 0, -1):
            if prev_words[-k:] == new_words[:k]:
                return " ".join(prev_words + new_words[k:])
        return f"{prev_text} {new_text}".strip()

    merged: list[dict] = []
    for block in blocks:
        if (
            merged
            and block.get("type") == "comm"
            and merged[-1].get("type") == "comm"
            and block.get("timestamp")
            and block.get("timestamp") == merged[-1].get("timestamp")
        ):
            prev = merged[-1]
            prev_speaker = prev.get("speaker")
            block_speaker = block.get("speaker")
            if not block_speaker or not prev_speaker or block_speaker == prev_speaker:
                if not prev_speaker and block_speaker:
                    prev["speaker"] = block_speaker
                if not prev.get("location") and block.get("location"):
                    prev["location"] = block.get("location")
                if block.get("text"):
                    prev_text = prev.get("text", "")
                    new_text = block["text"]
                    if prev_text and new_text:
                        prev_words = set(prev_text.lower().split())
                        new_words = set(new_text.lower().split())
                        if prev_words and new_words:
                            overlap = len(prev_words & new_words) / min(len(prev_words), len(new_words))
                            if overlap < 0.3:
                                merged.append(block)
                                continue
                        prev["text"] = new_text if len(new_text) > len(prev_text) else prev_text
                    else:
                        prev["text"] = merge_text(prev_text, new_text)
                if not prev.get("timestamp_correction") and block.get("timestamp_correction"):
                    prev["timestamp_correction"] = block.get("timestamp_correction")
                continue
        merged.append(block)
    return merged


def remove_repeated_phrases(text: str) -> str:
    """
    Detects and removes identical word sequences repeated immediately within a line.

    Args:
        text: Dialogue line to clean.

    Returns:
        Cleaned text with stutter/repetitions removed.
    """
    words = text.split()
    if len(words) < 12:
        return text
    max_span = 12
    min_span = 6
    window = 20
    for span in range(max_span, min_span - 1, -1):
        for i in range(0, len(words) - 2 * span + 1):
            seq = words[i:i + span]
            for j in range(i + span, min(len(words) - span + 1, i + span + window)):
                if words[j:j + span] == seq:
                    new_words = words[:j] + words[j + span:]
                    return " ".join(new_words)
    return text


def merge_nearby_duplicate_timestamps(blocks: list[dict], window: int = 4) -> list[dict]:
    """
    Merges blocks with the same timestamp occurring within a small lookahead window.

    Handles cases where non-consecutive blocks (e.g., separated by fragments)
    belong to the same dialogue line.

    Args:
        blocks: List of communication blocks.
        window: Number of blocks to look ahead for merging.

    Returns:
        List of blocks with nearby duplicates merged.
    """
    merged: list[dict] = []
    for block in blocks:
        if block.get("type") != "comm" or not block.get("timestamp"):
            merged.append(block)
            continue
        ts = block.get("timestamp")
        speaker = block.get("speaker")
        merged_idx = None
        for back in range(1, min(window, len(merged)) + 1):
            prev = merged[-back]
            if prev.get("type") != "comm":
                continue
            if prev.get("timestamp") != ts:
                continue
            prev_speaker = prev.get("speaker")
            if prev_speaker and speaker and prev_speaker != speaker:
                short_text = (block.get("text") or "").strip()
                if len(short_text) > 20:
                    continue
            merged_idx = len(merged) - back
            break
        if merged_idx is None:
            merged.append(block)
            continue
        prev = merged[merged_idx]
        if not prev.get("speaker") and speaker:
            prev["speaker"] = speaker
        if not prev.get("location") and block.get("location"):
            prev["location"] = block.get("location")
        if block.get("text"):
            prev_text = prev.get("text", "")
            new_text = block["text"]
            if prev_text and new_text:
                prev_words = set(prev_text.lower().split())
                new_words = set(new_text.lower().split())
                if prev_words and new_words:
                    overlap = len(prev_words & new_words) / min(len(prev_words), len(new_words))
                    if overlap < 0.3:
                        merged.append(block)
                        continue
            prev["text"] = new_text if len(new_text) > len(prev_text) else prev_text
        if not prev.get("timestamp_correction") and block.get("timestamp_correction"):
            prev["timestamp_correction"] = block.get("timestamp_correction")
    return merged


def clean_or_merge_continuations(blocks: list[dict]) -> list[dict]:
    """
    Appends continuation blocks to the preceding communication block.

    Validates that the continuation is not just noise or a footer artifact
    before merging the text content.

    Args:
        blocks: List of communication and continuation blocks.

    Returns:
        List of blocks with continuations merged into their parents.
    """
    cleaned: list[dict] = []
    for idx, block in enumerate(blocks):
        if block.get("type") != "continuation":
            cleaned.append(block)
            continue

        text = clean_leading_footer_noise((block.get("text") or "").strip())
        if not text:
            continue
        if is_garbled_page_tape_label(text):
            continue
        block["text"] = text

        if EMBEDDED_TIMESTAMP_RE.search(text):
            cleaned.append(block)
            continue

        if cleaned:
            prev = cleaned[-1]
            prev_text = (prev.get("text") or "").strip()
            if prev_text == text or text in prev_text:
                continue
            if prev_text:
                prev["text"] = (prev_text + " " + text).strip()
            else:
                prev["text"] = text
        else:
            cleaned.append(block)

    return cleaned


def normalize_parenthesized_radio_calls(text: str) -> str:
    """
    Standardizes parenthesized radio terminology (e.g., "(OVER)" -> "OVER").

    Fixes common OCR artifacts where dialogue terminators are incorrectly 
    wrapped or partially captured.

    Args:
        text: The dialogue line to normalize.

    Returns:
        Text with radio calls normalized.
    """
    if not text:
        return text

    def repl(match: re.Match) -> str:
        phrase = match.group(1).upper()
        punct = match.group(2) or ""
        return f"{phrase}{punct}"

    normalized = PAREN_RADIO_CALL_RE.sub(repl, text)
    normalized = re.sub(
        r"\(\s*(OVER|OUT|ROGER|COPY|WILCO|GO AHEAD|SAY AGAIN|STAND BY|STANDBY)\b",
        lambda m: m.group(1).upper(),
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(OVER|OUT|ROGER|COPY|WILCO|GO AHEAD|SAY AGAIN|STAND BY|STANDBY)\s*\)",
        lambda m: m.group(1).upper(),
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    return normalized


def merge_inline_annotations(blocks: list[dict], inline_terms: list[str] | None = None) -> list[dict]:
    """
    Merges specific technical annotations into the preceding communication block.

    Args:
        blocks: List of communication and annotation blocks.
        inline_terms: Allowlist of annotation terms to merge (e.g., ["LAUGHTER"]).

    Returns:
        List of blocks with allowed inline annotations merged.
    """
    if not inline_terms:
        return blocks
    terms = {t.upper().strip() for t in inline_terms if t}
    cleaned: list[dict] = []
    for block in blocks:
        if block.get("type") != "annotation":
            cleaned.append(block)
            continue

        text = (block.get("text") or "").strip()
        if not text:
            continue

        normalized = re.sub(r"[^A-Z0-9 ]+", "", text.upper()).strip()
        if normalized in terms:
            if cleaned and cleaned[-1].get("type") == "comm":
                prev_text = (cleaned[-1].get("text") or "").strip()
                if prev_text:
                    cleaned[-1]["text"] = f"{prev_text} {text}".strip()
                else:
                    cleaned[-1]["text"] = text
            else:
                cleaned.append({"type": "comm", "text": text})
            continue

        cleaned.append(block)

    return cleaned


def merge_fragment_annotations(blocks: list[dict]) -> list[dict]:
    """
    Merges short technical text fragments into the preceding block.

    Useful for capturing orphaned words or technical codes (e.g., "S-IVB.")
    that were correctly identified as annotations but belong to the dialogue.

    Args:
        blocks: List of communication and annotation blocks.

    Returns:
        List of blocks with technical fragments merged.
    """
    cleaned: list[dict] = []
    for block in blocks:
        if block.get("type") != "annotation":
            cleaned.append(block)
            continue

        text = (block.get("text") or "").strip()
        if not text:
            continue

        upper = text.upper()
        if (
            cleaned
            and cleaned[-1].get("type") in ("comm", "continuation")
            and ANNOTATION_FRAGMENT_RE.match(upper)
            and "END OF TAPE" not in upper
            and "LUNAR REV" not in upper
            and "REST PERIOD" not in upper
            and "GOSS NET" not in upper
            and " TAPE " not in f" {upper} "
            and " PAGE " not in f" {upper} "
        ):
            prev_text = (cleaned[-1].get("text") or "").strip()
            cleaned[-1]["text"] = f"{prev_text} {text}".strip() if prev_text else text
            continue

        cleaned.append(block)

    return cleaned
