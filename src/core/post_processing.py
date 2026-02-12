"""
Post-processing pipeline for OCR blocks.
"""
import re
from pathlib import Path
from typing import Any

from src.correctors.speaker_corrector import SpeakerCorrector
from src.correctors.text_corrector import TextCorrector
from src.correctors.location_corrector import LocationCorrector
from src.correctors.timestamp_corrector import TimestampCorrector
from src.ocr.parsing.cleaning import (
    split_embedded_timestamp_blocks,
    merge_duplicate_comm_timestamps,
    merge_nearby_duplicate_timestamps,
    clean_or_merge_continuations,
    merge_inline_annotations,
    merge_fragment_annotations,
    remove_repeated_phrases,
)
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
        """
        Initializes the PostProcessor with mission-specific configuration.

        Args:
            valid_speakers: Allowlist of speaker callsigns.
            valid_locations: Allowlist of location codes.
            mission_keywords: Terms used for speaker/location disambiguation.
            text_replacements: Global regex patterns for OCR error correction.
            speaker_ocr_fixes: Known OCR misreads for speakers.
            invalid_location_annotations: Terms incorrectly parsed as locations.
            manual_speaker_corrections: Timestamp-to-speaker mapping for hard fixes.
            lexicon_path: Path to the JSON vocabulary for spell checking.
        """
        self.valid_speakers = valid_speakers
        self.valid_locations = valid_locations
        self.mission_keywords = mission_keywords or []
        self.text_replacements = text_replacements or {}
        self.speaker_ocr_fixes = speaker_ocr_fixes or {}
        self.invalid_location_annotations = invalid_location_annotations or []
        self.manual_speaker_corrections = manual_speaker_corrections or {}
        self.lexicon_path = lexicon_path

    def process_blocks(self, blocks: list[dict[str, Any]], initial_ts: str | None = None) -> list[dict[str, Any]]:
        """
        Executes the full post-processing pipeline on a list of blocks.

        Stages:
        1. Structural cleaning (merging/splitting).
        2. Internal text cleaning.
        3. Chronological timestamp correction.
        4. Manual speaker overrides.
        5. Fuzzy speaker identification.
        6. Location normalization and tag removal.
        7. Final filtering and lexicon-based spell correction.

        Args:
            blocks: Raw communication blocks from the parser.
            initial_ts: The last known valid timestamp from the previous page.

        Returns:
            A cleaned and corrected list of blocks.
        """
        # 1. Structural cleaning
        blocks = split_embedded_timestamp_blocks(blocks)
        blocks = merge_duplicate_comm_timestamps(blocks)
        blocks = merge_nearby_duplicate_timestamps(blocks)
        blocks = clean_or_merge_continuations(blocks)
        blocks = merge_inline_annotations(blocks, self.invalid_location_annotations)
        blocks = merge_fragment_annotations(blocks)

        # 2. Text cleaning within blocks
        for block in blocks:
            if block.get("type") == "comm" and block.get("text"):
                block["text"] = remove_repeated_phrases(block["text"])

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
            corrector = SpeakerCorrector(self.valid_speakers, self.speaker_ocr_fixes)
            blocks = corrector.process_blocks(blocks, self.mission_keywords)
            
            # Remove redundant speaker prefixes from text (e.g. "CDR Roger" -> "Roger")
            for block in blocks:
                if block.get("type") == "comm" and block.get("speaker") and block.get("text"):
                    spk = block["speaker"].upper()
                    txt = block["text"]
                    # Check for "SPEAKER " or "SPEAKER:" at start
                    if txt.upper().startswith(f"{spk} "):
                        block["text"] = txt[len(spk)+1:].lstrip()
                    elif txt.upper().startswith(f"{spk}:"):
                        block["text"] = txt[len(spk)+1:].lstrip()

        # 6. Location correction
        if self.valid_locations:
            loc_corrector = LocationCorrector(self.valid_locations, self.invalid_location_annotations)
            blocks = loc_corrector.process_blocks(blocks)

            # Remove orphaned location tags from text
            for block in blocks:
                if block.get("type") == "comm" and block.get("text"):
                    text = block["text"]
                    for loc in self.valid_locations:
                         text = re.sub(rf"\(['\"\s]*{re.escape(loc)}['\"\s.]*\)\s*", " ", text, flags=re.IGNORECASE)
                    block["text"] = text.strip()

        # 7. Final filtering and text correction
        # Reclassify problematic blocks
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

        # Drop empty comm blocks
        blocks = [
            b for b in blocks
            if not (b.get("type") == "comm" and
                    not b.get("speaker") and
                    (len(b.get("text", "").strip()) < 10 or
                     b.get("text", "").strip() in (".", ". Over.", "Over.")))
        ]

        # Lexicon-based text correction
        if self.lexicon_path and self.lexicon_path.exists():
            text_corrector = TextCorrector(self.lexicon_path, self.text_replacements, self.mission_keywords)
            blocks = text_corrector.process_blocks(blocks)
            # Filter GOSS NET noise
            blocks = [b for b in blocks if not (b.get("text") and GOSS_NET_RE.match(str(b.get("text"))))]

        return blocks

    def prune_empty_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove comm blocks that have no text content (even if they have a speaker).
        Call this after all OCR passes and merges are complete.
        """
        return [
            b for b in blocks
            if not (b.get("type") == "comm" and not b.get("text", "").strip())
        ]
