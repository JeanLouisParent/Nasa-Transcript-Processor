"""
Orchestration of the post-processing pipeline.
"""

import re
from typing import Any
from pathlib import Path

from src.correctors.speaker_corrector import SpeakerCorrector
from src.correctors.text_corrector import TextCorrector
from src.correctors.location_corrector import LocationCorrector
from src.correctors.timestamp_corrector import TimestampCorrector
from src.utils.station_normalization import match_station_name
from src.ocr.parsing.cleaning import (
    split_embedded_timestamp_blocks,
    merge_duplicate_comm_timestamps,
    merge_nearby_duplicate_timestamps,
    clean_or_merge_continuations,
    merge_inline_annotations,
    merge_fragment_annotations,
    remove_repeated_phrases,
    normalize_parenthesized_radio_calls,
)
from src.ocr.parsing.utils import is_not1_footer_noise
from src.ocr.parsing.patterns import GOSS_NET_RE

class PostProcessor:
    """
    Orchestrates the cleanup and correction of parsed OCR communication blocks.
    
    This class applies structural cleaning (merging fragments, splitting embedded TS),
    followed by multi-stage corrections (timestamps, speakers, locations, and text).
    """
    def __init__(
        self,
        valid_speakers: list[str] | None = None,
        valid_locations: list[str] | None = None,
        mission_keywords: list[str] | None = None,
        text_replacements: dict[str, str] | None = None,
        speaker_ocr_fixes: dict[str, str] | None = None,
        invalid_location_annotations: list[str] | None = None,
        manual_speaker_corrections: dict[str, str] | None = None,
        lexicon_path: Path | None = None,
    ):
        self.valid_speakers = valid_speakers
        self.valid_locations = valid_locations
        self.mission_keywords = mission_keywords or []
        self.text_replacements = text_replacements or {}
        self.speaker_ocr_fixes = speaker_ocr_fixes or {}
        self.invalid_location_annotations = invalid_location_annotations or []
        self.manual_speaker_corrections = manual_speaker_corrections or {}
        self.lexicon_path = lexicon_path
        self.station_pattern = re.compile(r"\b([A-Z0-9 ]{3,40}?)\s*\((REV|PASS|RFV)\s*(\d+)\)\s*", re.IGNORECASE)

    def _clean_text(self, text: str) -> str:
        """Applies low-level text cleaning rules before spell-checking."""
        if not text:
            return ""
            
        # 1. Hardware/Location tag removal (e.g. (TRANQ), (COLUMBIA))
        if self.valid_locations:
            for loc in self.valid_locations:
                pattern = re.compile(rf"\(?\s*{re.escape(loc)}\s*\)?", re.IGNORECASE)
                text = pattern.sub(" ", text)
        
        # 2. Basic structural normalization
        text = remove_repeated_phrases(text)
        text = normalize_parenthesized_radio_calls(text)
        
        # 3. High-frequency OCR error fixes
        # Fix common I'm/I'd confusions
        text = re.sub(r"\bhim\b", "I'm", text, flags=re.IGNORECASE)
        if text.lower().startswith("had "):
            text = "I'd" + text[3:]
        
        # 4. Cleanup orphaned punctuation
        if text.count("(") == 1 and text.count(")") == 0:
            text = text.replace("(", "")
        elif text.count(")") == 1 and text.count("(") == 0:
            text = text.replace(")", "")
            
        # 5. Sentence casing
        text = text.strip()
        if len(text) > 10 and text[0].islower() and not text.startswith("..."):
            text = text[0].upper() + text[1:]
            
        return text.strip()

    def process_blocks(self, blocks: list[dict[str, Any]], initial_ts: str | None = None) -> list[dict[str, Any]]:
        """
        Executes the full post-processing pipeline on a list of blocks.
        """
        # 1. Structural cleaning
        blocks = split_embedded_timestamp_blocks(blocks)
        blocks = merge_duplicate_comm_timestamps(blocks)
        blocks = merge_nearby_duplicate_timestamps(blocks)
        blocks = clean_or_merge_continuations(blocks)
        blocks = merge_inline_annotations(blocks, self.invalid_location_annotations)
        blocks = merge_fragment_annotations(blocks)

        # 2. Internal text cleaning
        for block in blocks:
            if block.get("type") in ("comm", "annotation", "continuation") and block.get("text"):
                block["text"] = self._clean_text(block["text"])

        # 3. Timestamp correction
        ts_corrector = TimestampCorrector(initial_ts)
        blocks = ts_corrector.process_blocks(blocks)

        # 4. Manual corrections
        if self.manual_speaker_corrections:
            for block in blocks:
                if block.get("type") == "comm" and block.get("timestamp"):
                    ts = block["timestamp"]
                    if ts in self.manual_speaker_corrections:
                        block["speaker"] = self.manual_speaker_corrections[ts]

        # 5. Speaker correction
        if self.valid_speakers:
            # Pre-pass: Split corrupted speakers (e.g. "CC Apollo")
            valid_set = {s.upper() for s in self.valid_speakers}
            for block in blocks:
                if block.get("type") != "comm":
                    continue
                raw_spk = block.get("speaker", "").strip()
                if not raw_spk:
                    continue
                if " " in raw_spk:
                    parts = raw_spk.split(maxsplit=1)
                    first = parts[0].upper()
                    if first in valid_set or (first in self.speaker_ocr_fixes and self.speaker_ocr_fixes[first] in valid_set):
                        block["speaker"] = first
                        remaining = parts[1]
                        old_text = block.get("text", "")
                        block["text"] = (remaining + " " + old_text).strip()
            
            corrector = SpeakerCorrector(self.valid_speakers, self.speaker_ocr_fixes)
            blocks = corrector.process_blocks(blocks, self.mission_keywords)
            
            # Remove redundant speaker prefixes from text
            for block in blocks:
                if block.get("type") == "comm" and block.get("speaker") and block.get("text"):
                    spk = block["speaker"].upper()
                    txt = block["text"]
                    if txt.upper().startswith(f"{spk} "):
                        block["text"] = txt[len(spk)+1:].lstrip()
                    elif txt.upper().startswith(f"{spk}:"):
                        block["text"] = txt[len(spk)+1:].lstrip()

        # 6. Location correction
        if self.valid_locations:
            loc_corrector = LocationCorrector(self.valid_locations, self.invalid_location_annotations)
            blocks = loc_corrector.process_blocks(blocks)

        # 7. Final filtering and text correction
        if self.lexicon_path and self.lexicon_path.exists():
            text_corrector = TextCorrector(self.lexicon_path, self.text_replacements, self.mission_keywords)
            blocks = text_corrector.process_blocks(blocks)

        # 8. Station Annotation Extraction
        keyword_candidates = [kw.upper() for kw in (self.mission_keywords or [])]
        final_blocks = []
        for block in blocks:
            text = block.get("text", "")
            if not text:
                final_blocks.append(block)
                continue
            annotations = []
            current_text = text
            while True:
                match = self.station_pattern.search(current_text)
                if not match:
                    break
                station_raw = match.group(1).strip().upper()
                marker = match.group(2).upper().replace("RFV", "REV")
                number = match.group(3)
                matched_station = match_station_name(station_raw, keyword_candidates)
                if self.mission_keywords and not matched_station:
                    break
                station_label = matched_station or station_raw
                annotations.append(f"{station_label} ({marker} {number})")
                current_text = (current_text[:match.start()] + current_text[match.end():]).strip()
            if annotations:
                if not current_text and block.get("type") == "comm" and not block.get("timestamp"):
                    block["type"] = "annotation"
                    block["text"] = annotations[0]
                    block.pop("speaker", None)
                    block.pop("location", None)
                    final_blocks.append(block)
                    for ann in annotations[1:]:
                        final_blocks.append({"type": "annotation", "text": ann})
                else:
                    block["text"] = current_text
                    final_blocks.append(block)
                    for ann in annotations:
                        final_blocks.append({"type": "annotation", "text": ann})
            else:
                final_blocks.append(block)
        blocks = [b for b in final_blocks if b.get("text")]

        # 9. Block Reclassification
        for block in blocks:
            if block.get("type") == "comm":
                text = block.get("text", "").strip()
                if text.startswith("--") and "LUNAR REV" in text.upper():
                    block["type"] = "meta"
                    block.pop("speaker", None)
                    block.pop("location", None)
                elif text.startswith("(") and ("UNIDENTIFIABLE" in text.upper() or "UNINDENTIFIABLE" in text.upper()):
                    block["type"] = "annotation"
                    block.pop("speaker", None)
                    block.pop("location", None)
                elif not block.get("speaker"):
                    block["speaker"] = "SC"

        # 10. Filter noise
        blocks = [
            b for b in blocks 
            if not (b.get("type") == "comm" and b.get("speaker") == "SC" and len(b.get("text", "").strip()) < 3 and not any(c.isalnum() for c in b.get("text", "")))
        ]
        blocks = [
            b for b in blocks 
            if not (b.get("text") and (GOSS_NET_RE.match(str(b.get("text"))) or is_not1_footer_noise(str(b.get("text")))))
        ]

        # 11. Merge consecutive same-speaker blocks
        if not blocks:
            return blocks
        merged = []
        for i, block in enumerate(blocks):
            if (
                merged
                and block.get("type") == "comm"
                and merged[-1].get("type") == "comm"
                and block.get("speaker") == merged[-1].get("speaker")
                and block.get("text")
                and i > 0
                and not block.get("continuation_from_prev")
            ):
                prev = merged[-1]
                text = block["text"]
                prev_text = prev.get("text", "").strip()
                is_textual_cont = text[0].islower() or text.startswith("...") or text[0] in ",;.)"
                is_structural_cont = prev_text and prev_text[-1] not in ".?!"
                if is_textual_cont or is_structural_cont:
                    prev["text"] = (prev_text + " " + text).strip()
                    if not prev.get("timestamp_correction") and block.get("timestamp_correction"):
                        prev["timestamp_correction"] = block.get("timestamp_correction")
                    continue
            merged.append(block)
        return merged

    def prune_empty_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [b for b in blocks if b.get("text", "").strip()]
