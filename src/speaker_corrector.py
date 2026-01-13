"""
Speaker Correction Module.

Validates and corrects speaker names against a mission-specific allowlist.
Uses fuzzy matching to fix OCR errors (e.g. "CD R" -> "CDR").
"""

import difflib
from typing import Optional

class SpeakerCorrector:
    def __init__(self, valid_speakers: list[str]):
        """
        Initialize with a list of valid speakers.
        Args:
            valid_speakers: List of allowed speaker codes (e.g. ["CDR", "CC"])
        """
        self.valid_speakers = valid_speakers
        self.valid_speakers_set = set(valid_speakers)

    def correct_speaker(self, raw_speaker: str) -> str:
        """
        Correct a speaker code.
        Returns the closest match from valid_speakers or the original if no match found.
        """
        if not raw_speaker:
            return ""

        # normalization
        normalized = raw_speaker.upper().strip()
        
        # 1. Exact match
        if normalized in self.valid_speakers_set:
            return normalized

        # 2. Heuristic fixes
        # Remove parentheses if present (e.g. "(CDR)" -> "CDR")
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1]
            if normalized in self.valid_speakers_set:
                return normalized

        # 3. Fuzzy match
        # cutoff=0.6 allows for small typos (1 char diff in short strings)
        matches = difflib.get_close_matches(normalized, self.valid_speakers, n=1, cutoff=0.5)
        if matches:
            return matches[0]

        return raw_speaker # Return original if no close match

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Process blocks to correct speaker names.
        """
        if not self.valid_speakers:
            return blocks

        for block in blocks:
            if block.get("type") == "comm" and block.get("speaker"):
                block["speaker"] = self.correct_speaker(block["speaker"])
        
        return blocks
