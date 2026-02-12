"""
Payload merging logic for combining multi-pass OCR results.
"""

import difflib
import re

def normalize_text_for_match(text: str) -> str:
    """
    Simplifies text for robust similarity comparison.

    Args:
        text: Raw string to normalize.

    Returns:
        Lowercase string with collapsed whitespace.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def text_quality_score(text: str) -> float:
    """
    Heuristically estimates the quality of OCR'd text.

    Considers character composition, word count, and presence of 
    unlikely symbols.

    Args:
        text: String to evaluate.

    Returns:
        A score where higher indicates better quality.
    """
    if not text:
        return 0.0
    total = len(text)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = sum(c.isspace() for c in text)
    punctuation = sum(c in ".,;:'\"-()/?!" for c in text)
    common = letters + digits + spaces + punctuation
    weird = max(0, total - common)
    words = re.findall(r"[A-Za-z]{2,}", text)
    score = (letters / total) + (len(words) / 8.0)
    if weird:
        score -= min(0.5, weird / total)
    return score


def should_insert_continuation(text: str) -> bool:
    """
    Determines if a text fragment is substantive enough to be a continuation.

    Args:
        text: Text fragment to check.

    Returns:
        True if the fragment likely contains valid dialogue.
    """
    if not text:
        return False
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 3:
        return False
    if len(text.strip()) < 20:
        return False
    return True


def find_comm_index(
    blocks: list[dict],
    timestamp: str,
    speaker: str = "",
    location: str = "",
    first: bool = True
) -> int | None:
    """
    Locates a specific communication block within a list.

    Args:
        blocks: List of blocks to search.
        timestamp: Targeted timecode.
        speaker: Optional speaker code filter.
        location: Optional location code filter.
        first: If True, returns the first match; otherwise the last.

    Returns:
        Index of the matching block, or None if not found.
    """
    if not timestamp:
        return None
    speaker = speaker.upper().strip()
    location = location.upper().strip()
    matches = []
    for idx, block in enumerate(blocks):
        if block.get("type") != "comm":
            continue
        if block.get("timestamp") != timestamp:
            continue
        if speaker and block.get("speaker", "").upper() != speaker:
            continue
        if location and block.get("location", "").upper() != location:
            continue
        matches.append(idx)
    if matches:
        return matches[0] if first else matches[-1]
    if speaker or location:
        for idx, block in enumerate(blocks):
            if block.get("type") == "comm" and block.get("timestamp") == timestamp:
                return idx
    return None


def merge_payloads(preferred: dict, fallback: dict) -> dict:
    """
    Merge two page payloads, preferring blocks from the 'preferred' payload
    but filling in gaps or replacing low-quality text from 'fallback'.
    """
    preferred_blocks = list(preferred.get("blocks", []))
    fallback_blocks = fallback.get("blocks", [])
    if not fallback_blocks:
        return preferred

    preferred_texts = set()
    preferred_text_list = []
    for block in preferred_blocks:
        text = block.get("text")
        if text:
            preferred_texts.add(normalize_text_for_match(text))
            preferred_text_list.append(text)

    prev_ts_list = []
    prev_ts = None
    for block in fallback_blocks:
        if block.get("type") == "comm" and block.get("timestamp"):
            prev_ts = block.get("timestamp")
        prev_ts_list.append(prev_ts)

    next_ts_list = [None] * len(fallback_blocks)
    next_ts = None
    for idx in range(len(fallback_blocks) - 1, -1, -1):
        block = fallback_blocks[idx]
        if block.get("type") == "comm" and block.get("timestamp"):
            next_ts = block.get("timestamp")
        next_ts_list[idx] = next_ts

    def word_overlap_ratio(a: str, b: str) -> float:
        a_words = set(re.findall(r"[A-Za-z]+", a.lower()))
        b_words = set(re.findall(r"[A-Za-z]+", b.lower()))
        if not a_words and not b_words:
            return 0.0
        return len(a_words & b_words) / max(1, len(a_words | b_words))

    def is_near_duplicate(text: str) -> bool:
        if not text:
            return False
        norm = normalize_text_for_match(text)
        for other in preferred_text_list:
            ratio = difflib.SequenceMatcher(None, norm, normalize_text_for_match(other)).ratio()
            if ratio >= 0.9:
                return True
        return False

    for idx, fallback_block in enumerate(fallback_blocks):
        fb_text = fallback_block.get("text", "")
        if fallback_block.get("type") == "comm" and fallback_block.get("timestamp"):
            target_idx = find_comm_index(
                preferred_blocks,
                fallback_block.get("timestamp", ""),
                fallback_block.get("speaker", ""),
                fallback_block.get("location", ""),
                first=True
            )
            if target_idx is not None and fb_text:
                preferred_block = preferred_blocks[target_idx]
                pref_text = preferred_block.get("text", "")

                # Check if texts are significantly different (different quotes at same timestamp)
                if pref_text and fb_text:
                    pref_words = set(pref_text.lower().split())
                    fb_words = set(fb_text.lower().split())
                    if pref_words and fb_words:
                        overlap = len(pref_words & fb_words) / min(len(pref_words), len(fb_words))
                        if overlap < 0.3:
                            # Very different texts - insert as separate block instead of replacing
                            preferred_blocks.insert(target_idx + 1, fallback_block)
                            preferred_texts.add(normalize_text_for_match(fb_text))
                            preferred_text_list.append(fb_text)
                            continue

                pref_score = text_quality_score(pref_text)
                fb_score = text_quality_score(fb_text)
                if (
                    pref_text
                    and fb_text
                    and pref_text in fb_text
                    and len(fb_text) - len(pref_text) >= 8
                ):
                    preferred_block["text"] = fb_text
                    if not preferred_block.get("speaker") and fallback_block.get("speaker"):
                        preferred_block["speaker"] = fallback_block.get("speaker")
                    if not preferred_block.get("location") and fallback_block.get("location"):
                        preferred_block["location"] = fallback_block.get("location")
                elif (
                    pref_text
                    and fb_text
                    and len(fb_text) - len(pref_text) >= 8
                    and word_overlap_ratio(pref_text, fb_text) >= 0.85
                ):
                    preferred_block["text"] = fb_text
                    if not preferred_block.get("speaker") and fallback_block.get("speaker"):
                        preferred_block["speaker"] = fallback_block.get("speaker")
                    if not preferred_block.get("location") and fallback_block.get("location"):
                        preferred_block["location"] = fallback_block.get("location")
                elif (
                    pref_text
                    and pref_score >= 0.8
                    and len(pref_text.strip()) <= 15
                    and len(fb_text.strip()) >= 25
                    and word_overlap_ratio(pref_text, fb_text) < 0.2
                ):
                    preferred_blocks.insert(target_idx + 1, {"type": "continuation", "text": fb_text})
                    preferred_texts.add(normalize_text_for_match(fb_text))
                elif (
                    not pref_text
                    or pref_score < 0.6
                    or fb_score > pref_score + 0.4
                ):
                    preferred_block["text"] = fb_text
                    if not preferred_block.get("speaker") and fallback_block.get("speaker"):
                        preferred_block["speaker"] = fallback_block.get("speaker")
                    if not preferred_block.get("location") and fallback_block.get("location"):
                        preferred_block["location"] = fallback_block.get("location")
            continue

        if fallback_block.get("type") == "continuation":
            if not should_insert_continuation(fb_text):
                continue
            if normalize_text_for_match(fb_text) in preferred_texts:
                continue
            if is_near_duplicate(fb_text):
                continue
            prev_ts = prev_ts_list[idx]
            next_ts = next_ts_list[idx]
            insert_idx = None
            if prev_ts:
                insert_idx = find_comm_index(preferred_blocks, prev_ts, first=False)
                if insert_idx is not None:
                    insert_idx += 1
            if insert_idx is None and next_ts:
                insert_idx = find_comm_index(preferred_blocks, next_ts, first=True)
            if insert_idx is None:
                insert_idx = 0
            preferred_blocks.insert(insert_idx, {"type": "continuation", "text": fb_text})
            preferred_texts.add(normalize_text_for_match(fb_text))

    preferred["blocks"] = preferred_blocks
    return preferred
