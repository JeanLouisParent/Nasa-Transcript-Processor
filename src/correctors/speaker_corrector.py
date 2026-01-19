"""
Speaker Correction Module.

Validates and corrects speaker names against a mission-specific allowlist.
Uses fuzzy matching to fix OCR errors (e.g. "CD R" -> "CDR").
"""

import difflib


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
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "/")

        # 1. Exact match
        if normalized in self.valid_speakers_set:
            return normalized

        # 2. Heuristic fixes
        # Remove parentheses if present (e.g. "(CDR)" -> "CDR")
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1]
            if normalized in self.valid_speakers_set:
                return normalized
        # Map single-letter tokens to doubled codes if present (e.g. "C" -> "CC")
        if len(normalized) == 1:
            doubled = normalized * 2
            if doubled in self.valid_speakers_set:
                return doubled
        # Common OCR slips for CMP
        if normalized in {"CT", "CTP"} and "CMP" in self.valid_speakers_set:
            return "CMP"

        # 3. Fuzzy match
        # Prefer same-length call signs when OCR returns 3 chars.
        if len(normalized) == 3:
            candidates = [s for s in self.valid_speakers if len(s) == 3]
        else:
            candidates = self.valid_speakers
        # cutoff=0.6 allows for small typos (1 char diff in short strings)
        matches = difflib.get_close_matches(normalized, candidates, n=1, cutoff=0.5)
        if matches:
            return matches[0]

        return raw_speaker # Return original if no close match

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Process blocks to correct speaker names.
        Also attempts to extract speaker from text if speaker field is empty.
        """
        if not self.valid_speakers:
            return blocks

        for block in blocks:
            if block.get("type") != "comm":
                continue

            # 1. Try to recover speaker from text if empty
            if not block.get("speaker") and block.get("text"):
                # Split first word
                tokens = block["text"].split()
                if tokens:
                    # Try two-token speaker (e.g. "SWIM 1", "PRESIDENT NIXON")
                    if len(tokens) >= 2:
                        candidate = f"{tokens[0]} {tokens[1]}".upper()
                        if candidate in self.valid_speakers_set:
                            block["speaker"] = candidate
                            block["text"] = " ".join(tokens[2:]).strip()
                            continue

                    first_word = tokens[0]
                    # Check if first word looks like a speaker (fuzzy match)
                    # We use a stricter cutoff here to avoid extracting random words
                    corrected = self.correct_speaker(first_word)

                    # If correct_speaker found a match in valid_speakers (and it's not just returning original)
                    if corrected in self.valid_speakers_set:
                        block["speaker"] = corrected
                        block["text"] = " ".join(tokens[1:]).strip()

            # 2. Correct existing speaker field
            if block.get("speaker"):
                block["speaker"] = self.correct_speaker(block["speaker"])

        return blocks
