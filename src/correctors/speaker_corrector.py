"""
Speaker Correction Module.

Validates and corrects speaker names against a mission-specific allowlist.
Uses fuzzy matching to fix OCR errors (e.g. "CD R" -> "CDR").
"""

import difflib


class SpeakerCorrector:
    def __init__(self, valid_speakers: list[str], ocr_fixes: dict[str, str] | None = None):
        """
        Initialize with a list of valid speakers.
        Args:
            valid_speakers: List of allowed speaker codes (e.g. ["CDR", "CC"])
            ocr_fixes: Dictionary of common OCR errors to corrections (e.g. {"CT": "CMP"})
        """
        self.valid_speakers = valid_speakers
        self.valid_speakers_set = set(valid_speakers)
        self.ocr_fixes = ocr_fixes or {}

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
        # Common OCR slips
        if normalized in self.ocr_fixes and self.ocr_fixes[normalized] in self.valid_speakers_set:
            return self.ocr_fixes[normalized]

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
                text = block["text"]
                tokens = text.split()

                if tokens:
                    # Search for speaker in first 4 tokens (skipping likely non-speaker prefixes)
                    for i in range(min(4, len(tokens))):
                        candidate_token = tokens[i]
                        token_alnum = "".join(ch for ch in candidate_token if ch.isalnum())
                        looks_like_quoted_single_speaker = (
                            len(token_alnum) == 1 and token_alnum.isalpha()
                        )

                        # Skip tokens that are clearly not speakers
                        # (timestamps fragments, pure numbers, single chars with quotes/colons)
                        if ((len(candidate_token) <= 2 and any(c in candidate_token for c in "':0123456789")
                             and not looks_like_quoted_single_speaker)) or \
                           candidate_token.isdigit() or \
                           all(c in "0123456789: '-" for c in candidate_token):
                            continue

                        # Try two-token speaker (e.g. "SWIM 1", "PRESIDENT NIXON")
                        if i + 1 < len(tokens):
                            candidate = f"{tokens[i]} {tokens[i+1]}".upper()
                            if candidate in self.valid_speakers_set:
                                block["speaker"] = candidate
                                block["text"] = " ".join(tokens[:i] + tokens[i+2:]).strip()
                                break

                        # Try single token speaker
                        corrected = self.correct_speaker(candidate_token)
                        if corrected in self.valid_speakers_set:
                            block["speaker"] = corrected
                            block["text"] = " ".join(tokens[:i] + tokens[i+1:]).strip()
                            break

            # 2. Correct existing speaker field
            if block.get("speaker"):
                block["speaker"] = self.correct_speaker(block["speaker"])

        return blocks
