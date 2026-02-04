"""
Build final page JSON from parsed rows.
"""

import difflib
import re
from pathlib import Path

from src.correctors.speaker_corrector import SpeakerCorrector
from src.correctors.text_corrector import TextCorrector
from src.correctors.timestamp_corrector import TimestampCorrector

from .patterns import (
    HEADER_PAGE_ONLY_RE,
    HEADER_TAPE_ONLY_RE,
    HEADER_TAPE_SIMPLE_RE,
    HEADER_TAPE_PAGE_ONLY_RE,
    LUNAR_REV_RE,
    REST_PERIOD_RE,
    NO_COMM_RE,
    END_OF_TAPE_KEYWORD,
    GOSS_NET_RE,
    SPEAKER_TOKEN_RE,
)
from .utils import (
    fuzzy_find,
    is_transcription_header,
    is_goss_net_noise,
    is_not1_footer_noise,
    clean_trailing_footer,
    clean_leading_footer_noise,
    extract_header_metadata,
)


EMBEDDED_TIMESTAMP_RE = re.compile(
    r"\(?\b[0-9OIil]{1,2}[:\s-][0-9OIil]{1,2}[:\s-][0-9OIil]{1,2}[ .:\-][0-9OIil]{1,2}\b\)?"
)
SECONDARY_EMBEDDED_RE = re.compile(
    r"(?<!\d)"
    r"([0-9OIilCSB]{1,2})[^0-9OIilCSB]+"
    r"([0-9OIilCSB]{1,2})[^0-9OIilCSB]+"
    r"([0-9OIilCSB]{1,2})[^0-9OIilCSB]+"
    r"([0-9OIilCSB]{1,2})"
    r"\s+([A-Z0-9]{1,8}(?:/[A-Z0-9]{1,8})?)",
    re.IGNORECASE,
)

PAREN_RADIO_CALL_RE = re.compile(
    r"\(\s*(OVER|OUT|ROGER|COPY|WILCO|GO AHEAD|SAY AGAIN|STAND BY|STANDBY)\s*([.!?])?\s*\)",
    re.IGNORECASE,
)
ANNOTATION_FRAGMENT_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-./ ]{0,24}\.?$")


def _normalize_embedded_timestamp(raw: str) -> str | None:
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


def _normalize_embedded_groups(groups: tuple[str, str, str, str]) -> str:
    def norm(token: str) -> str:
        token = token.replace("O", "0").replace("o", "0")
        token = token.replace("I", "1").replace("i", "1").replace("l", "1")
        token = token.replace("C", "0").replace("c", "0")
        token = token.replace("S", "5").replace("s", "5")
        token = token.replace("B", "8").replace("b", "8")
        if len(token) == 1:
            token = f"0{token}"
        return token

    parts = [norm(g) for g in groups]
    return " ".join(parts)


def split_embedded_timestamp_blocks(blocks: list[dict]) -> list[dict]:
    """
    Split blocks that contain embedded timestamps inside their text.
    """
    output: list[dict] = []
    for block in blocks:
        text = block.get("text")
        if not text:
            output.append(block)
            continue

        matches = list(EMBEDDED_TIMESTAMP_RE.finditer(text))
        # DISABLED: SECONDARY_EMBEDDED_RE is too aggressive and creates false positives
        # if not matches:
        #     matches = list(SECONDARY_EMBEDDED_RE.finditer(text))
        if not matches:
            output.append(block)
            continue

        prefix = text[:matches[0].start()].strip()
        if prefix:
            base_block = dict(block)
            base_block["text"] = prefix
            output.append(base_block)

        for idx, match in enumerate(matches):
            if match.re is SECONDARY_EMBEDDED_RE:
                ts = _normalize_embedded_groups(match.group(1, 2, 3, 4))
                speaker = match.group(5).rstrip(":")
                seg_start = match.end()
            else:
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
                    speaker = tokens[0].rstrip(":")
                    seg_text = " ".join(tokens[1:]).strip()
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
    Merge consecutive comm blocks that share the same timestamp.
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
                        # Same timestamp duplicates are usually OCR splits; keep the most complete variant.
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
    Remove immediately repeated word sequences inside a line.
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
    Merge duplicate comm blocks with the same timestamp within a small window.
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
            prev["text"] = new_text if len(new_text) > len(prev_text) else prev_text
        if not prev.get("timestamp_correction") and block.get("timestamp_correction"):
            prev["timestamp_correction"] = block.get("timestamp_correction")
    return merged


def clean_or_merge_continuations(blocks: list[dict]) -> list[dict]:
    """
    Merge non-leading continuation blocks into the previous block when possible.
    Drop exact duplicates. Preserve lines that contain embedded timestamps.
    """
    cleaned: list[dict] = []
    for idx, block in enumerate(blocks):
        if block.get("type") != "continuation":
            cleaned.append(block)
            continue

        text = clean_leading_footer_noise((block.get("text") or "").strip())
        if not text:
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
    Normalize parenthesized radio fillers: "(OVER)" -> "OVER", "(OUT.)" -> "OUT."
    """
    if not text:
        return text

    def repl(match: re.Match) -> str:
        phrase = match.group(1).upper()
        punct = match.group(2) or ""
        return f"{phrase}{punct}"

    normalized = PAREN_RADIO_CALL_RE.sub(repl, text)
    # Handle malformed OCR where opening/closing parenthesis is missing.
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
    Merge standalone annotation tags into the previous comm block or convert to comm.
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
    Merge short technical annotation fragments (e.g. "S-IVB.") into the previous text block.
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


def build_page_json(
    rows: list[dict],
    lines: list[str],
    page_num: int,
    page_offset: int = 0,
    valid_speakers: list[str] | None = None,
    text_replacements: dict[str, str] | None = None,
    mission_keywords: list[str] | None = None,
    valid_locations: list[str] | None = None,
    inline_annotation_terms: list[str] | None = None,
    initial_ts: str | None = None,
    previous_block_type: str | None = None,
) -> dict:
    """
    Build final page JSON from parsed rows with all corrections applied.
    """
    header_info = extract_header_metadata(lines, page_num, page_offset)
    blocks = []

    for row in rows:
        if row["type"] == "header":
            continue
        block_type = "continuation" if row["type"] == "text" else row["type"]
        block = {"type": block_type}

        if block_type == "comm":
            if row["timestamp"]:
                block["timestamp"] = row["timestamp"]
            if row["speaker"]:
                block["speaker"] = row["speaker"]
            if row["location"]:
                block["location"] = row["location"]
            if row.get("timestamp_suffix_hint"):
                block["timestamp_suffix_hint"] = row["timestamp_suffix_hint"]

        if row["text"]:
            text_value = row["text"]
            if (
                block_type == "comm"
                and (
                    HEADER_PAGE_ONLY_RE.match(text_value)
                    or HEADER_TAPE_ONLY_RE.match(text_value)
                    or HEADER_TAPE_SIMPLE_RE.match(text_value)
                    or HEADER_TAPE_PAGE_ONLY_RE.match(text_value)
                )
            ):
                continue
            block["text"] = text_value
            if block_type == "meta" and fuzzy_find(row["text"], END_OF_TAPE_KEYWORD):
                block["meta_type"] = "end_of_tape"
            if block_type == "footer" and row["text"].lstrip().startswith("***"):
                block["text"] = "*** Three asterisks denote clipping of words and phrases."
            if (
                block.get("text")
                and (
                    GOSS_NET_RE.match(block["text"])
                    or is_goss_net_noise(block["text"])
                    or is_not1_footer_noise(block["text"])
                )
                and block_type in ("continuation", "meta", "annotation")
            ):
                continue

        if block_type == "continuation" and blocks:
            if blocks[-1].get("text") and block.get("text"):
                blocks[-1]["text"] = (blocks[-1]["text"] + " " + block["text"]).strip()
            elif block.get("text"):
                blocks[-1]["text"] = block["text"]
            continue
        if (
            block_type == "meta"
            and block.get("text")
            and blocks
            and blocks[-1].get("type") == "meta"
            and not blocks[-1].get("meta_type")
            and not block.get("meta_type")
        ):
            blocks[-1]["text"] = (blocks[-1].get("text", "") + " " + block["text"]).strip()
            continue
        if blocks and block_type == "continuation" and blocks[-1]["type"] == "continuation":
            blocks[-1]["text"] = (blocks[-1]["text"] + " " + block["text"]).strip()
        else:
            blocks.append(block)

    # Split blocks that embed multiple timestamps in their text
    blocks = split_embedded_timestamp_blocks(blocks)
    # Merge consecutive comm blocks that share the same timestamp
    blocks = merge_duplicate_comm_timestamps(blocks)
    # Merge nearby duplicates within a small window
    blocks = merge_nearby_duplicate_timestamps(blocks)

    # Transform LUNAR REV blocks
    for block in blocks:
        if block.get("type") == "comm" and block.get("text"):
            text_value = block["text"].strip()
            match = LUNAR_REV_RE.match(text_value)
            if match:
                action = match.group(1).upper()
                rev_num = match.group(2)
                block["text"] = f"{action} LUNAR REV {rev_num}"
                ts = block.get("timestamp", "")
                if ts:
                    parts = ts.split()
                    if len(parts) == 4:
                        parts[-1] = "--"
                        block["timestamp"] = " ".join(parts)
                    elif len(parts) == 3:
                        block["timestamp"] = " ".join(parts + ["--"])
                block["type"] = "meta"
                block["meta_type"] = "lunar_rev"
                block.pop("speaker", None)
                block.pop("location", None)

    # Smart stitching of continuation blocks
    merged_blocks = []
    for block in blocks:
        if (
            block.get("type") == "continuation"
            and block.get("text")
            and merged_blocks
            and merged_blocks[-1].get("type") == "comm"
        ):
            text = block["text"]
            if text[:1] in ";,.)" or (text[:1].islower()):
                merged_blocks[-1]["text"] = (merged_blocks[-1].get("text", "") + " " + text).strip()
                continue
            if (
                block.get("type") == "comm"
                and not block.get("speaker")
                and block.get("text")
                and merged_blocks
                and merged_blocks[-1].get("type") == "comm"
            ):
                text = block["text"]
                if is_not1_footer_noise(text):
                    continue
                if text[:1] in ";,.)" or (text[:1].islower()):
                    merged_blocks[-1]["text"] = (merged_blocks[-1].get("text", "") + " " + text).strip()
                    continue
        if (
            block.get("type") in ("meta", "continuation", "annotation")
            and block.get("text")
            and merged_blocks
            and merged_blocks[-1].get("type") == "comm"
        ):
            text = block["text"]
            if text[:1] in ";,.)" or (text[:1].islower()):
                merged_blocks[-1]["text"] = (merged_blocks[-1].get("text", "") + " " + text).strip()
                continue
        merged_blocks.append(block)
    blocks = merged_blocks
    blocks = clean_or_merge_continuations(blocks)
    blocks = merge_inline_annotations(blocks, inline_annotation_terms)
    blocks = merge_fragment_annotations(blocks)

    # Canonicalize REST PERIOD pages
    rest_period_found = any(
        (b.get("text") and REST_PERIOD_RE.search(b.get("text", "")))
        for b in blocks
    )
    normalized_blocks: list[dict] = []
    for block in blocks:
        text_val = block.get("text", "")
        if text_val and REST_PERIOD_RE.search(text_val) and NO_COMM_RE.search(text_val):
            block["type"] = "meta"
            block["meta_type"] = "rest_period"
            block["text"] = "REST PERIOD - NO COMMUNICATIONS"
        elif text_val and REST_PERIOD_RE.search(text_val):
            block["type"] = "meta"
            block["meta_type"] = "rest_period"
            block["text"] = "REST PERIOD - NO COMMUNICATIONS"
        if text_val and is_transcription_header(text_val):
            if rest_period_found:
                continue
            block["type"] = "meta"
            block["meta_type"] = "transcript_header"
            block["text"] = "AIR-TO-GROUND VOICE TRANSCRIPTION"
        normalized_blocks.append(block)
    blocks = normalized_blocks

    if blocks and blocks[0]["type"] == "continuation" and previous_block_type in ("comm", "continuation"):
        blocks[0]["continuation_from_prev"] = True

    if rest_period_found or any(b.get("meta_type") == "rest_period" for b in blocks):
        header_info["page_type"] = "rest_period"

    # Final cleanup of footers and locations on merged text
    # Also extract tracking station annotations as separate blocks
    cleaned_blocks = []
    # Match uppercase station-like names only, to avoid capturing normal prose.
    annotation_pattern = re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z]{2,}){0,4})\s*\((REV|PASS)\s*(\d+)\)\s*")
    keyword_candidates = [kw.upper() for kw in (mission_keywords or [])]
    keyword_tokens = {
        kw: [tok for tok in kw.split() if tok]
        for kw in keyword_candidates
    }

    def station_variants(station: str) -> list[str]:
        normalized = re.sub(r"[^A-Z0-9 ]", "", station.upper())
        tokens = [tok for tok in normalized.split() if tok]
        if not tokens:
            return []

        variants: list[str] = []
        max_shift = min(2, len(tokens) - 1)
        for shift in range(max_shift + 1):
            variant_tokens = tokens[shift:]
            # Drop common connective prefix words that OCR often keeps in station snippets.
            while variant_tokens and variant_tokens[0] in {"AND", "THE", "AT", "IN", "ON", "OF"}:
                variant_tokens = variant_tokens[1:]
            if variant_tokens:
                variants.append(" ".join(variant_tokens))

        # De-duplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for var in variants:
            if var not in seen:
                seen.add(var)
                deduped.append(var)
        return deduped

    def match_station_name(station: str) -> str | None:
        variants = station_variants(station)
        if not variants:
            return None
        for variant in variants:
            if variant in keyword_candidates:
                return variant

        best_kw: str | None = None
        best_score = 0.0
        for variant in variants:
            var_tokens = variant.split()
            for kw in keyword_candidates:
                score = difflib.SequenceMatcher(None, variant, kw).ratio()
                kw_toks = keyword_tokens.get(kw, [])
                if var_tokens and kw_toks:
                    if var_tokens[-1] == kw_toks[-1]:
                        score += 0.04
                    overlap = len(set(var_tokens) & set(kw_toks))
                    score += 0.02 * overlap
                if score > best_score:
                    best_score = score
                    best_kw = kw

        return best_kw if best_score >= 0.64 else None

    for block in blocks:
        if block.get("text"):
            text = block["text"]
            if block.get("type") == "comm":
                text = remove_repeated_phrases(text)
                text = normalize_parenthesized_radio_calls(text)
            text = clean_trailing_footer(text)
            text = clean_leading_footer_noise(text)

            # Extract tracking station annotations: "STATIONNAME (REV N)" or "STATIONNAME (PASS N)"
            if block.get("type") == "comm":
                annotations: list[str] = []
                current_text = text
                while True:
                    annotation_match = annotation_pattern.search(current_text)
                    if not annotation_match:
                        break

                    station = annotation_match.group(1).upper().strip()
                    marker = annotation_match.group(2).upper()
                    number = annotation_match.group(3)

                    # If mission keywords are available, only extract known station-like terms.
                    matched_station = match_station_name(station)
                    if mission_keywords and not matched_station:
                        break
                    station_label = matched_station or station
                    annotations.append(f"{station_label} ({marker} {number})")
                    current_text = (
                        current_text[:annotation_match.start()] + current_text[annotation_match.end():]
                    ).strip()

                if annotations:
                    block["text"] = current_text

                    # Remove any residual location tag at the start: "(TRANQ) ..."
                    if valid_locations:
                        for loc in valid_locations:
                            block["text"] = re.sub(rf"^\({re.escape(loc)}\)\s*", "", block["text"], flags=re.IGNORECASE)

                    cleaned_blocks.append(block)
                    for annotation_text in annotations:
                        cleaned_blocks.append({"type": "annotation", "text": annotation_text})
                    continue

            # Remove any residual location tag at the start: "(TRANQ) ..."
            if valid_locations:
                for loc in valid_locations:
                    text = re.sub(rf"^\({re.escape(loc)}\)\s*", "", text, flags=re.IGNORECASE)
            block["text"] = text.strip()

        cleaned_blocks.append(block)

    blocks = cleaned_blocks

    # Apply correctors
    ts_corrector = TimestampCorrector(initial_ts)
    blocks = ts_corrector.process_blocks(blocks)

    if valid_locations:
        for block in blocks:
            loc = block.get("location")
            if loc:
                best_loc = difflib.get_close_matches(loc.upper(), valid_locations, n=1, cutoff=0.5)
                if best_loc:
                    block["location"] = best_loc[0]

    if valid_speakers:
        blocks = SpeakerCorrector(valid_speakers).process_blocks(blocks)

    # Text correction
    lexicon_path = Path("assets/lexicon/apollo11_lexicon.json")
    if lexicon_path.exists():
        blocks = TextCorrector(lexicon_path, text_replacements, mission_keywords).process_blocks(blocks)
        blocks = [
            b for b in blocks
            if not (
                b.get("text")
                and (
                    GOSS_NET_RE.match(str(b.get("text")))
                    or is_goss_net_noise(str(b.get("text")))
                    or (
                        b.get("type") in ("continuation", "meta", "annotation")
                        and is_not1_footer_noise(str(b.get("text")))
                    )
                )
            )
        ]

    return {"header": header_info, "blocks": blocks}
