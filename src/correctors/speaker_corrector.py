"""
Speaker Correction Module.

Validates and corrects speaker names against a mission-specific allowlist.
Uses fuzzy matching to fix OCR errors (e.g. "CD R" -> "CDR").
"""

import difflib


class SpeakerCorrector:
    """
    Validates and recovers mission speaker identifiers (e.g., CDR, CC, LMP).
    
    Includes advanced heuristics for detecting OCR garbage, extracting speakers
    accidentally merged into text, and inferring speakers from dialogue flow.
    """
    def __init__(self, valid_speakers: list[str], ocr_fixes: dict[str, str] | None = None):
        """
        Initializes the corrector with mission callsigns and common OCR errors.

        Args:
            valid_speakers: Allowlist of uppercase speaker codes.
            ocr_fixes: Dictionary mapping common OCR artifacts to valid codes.
        """
        self.valid_speakers = valid_speakers
        self.valid_speakers_set = set(valid_speakers)
        self.ocr_fixes = ocr_fixes or {}

    def is_garbage_speaker(self, speaker: str, mission_keywords: list[str] | None = None) -> bool:
        """
        Detects if a parsed speaker code is likely corrupted OCR noise.

        Evaluates complexity, punctuation anomalies, and overlap with mission keywords.

        Args:
            speaker: The raw speaker string to check.
            mission_keywords: Terms to exclude (e.g. TLI should not be a speaker).

        Returns:
            True if the speaker should be treated as noise.
        """
        if not speaker or not speaker.strip():
            return False  # Empty is not garbage, it's just missing

        mission_keywords = mission_keywords or []

        # 1. Mission keyword detection (events/locations, not speakers)
        clean = speaker.upper().strip()
        # Remove parens/punctuation for keyword check
        for char in '()?\'"!;:':
            clean = clean.replace(char, '')
        clean = clean.strip()

        if mission_keywords and clean in [kw.upper() for kw in mission_keywords]:
            return True

        # 2. Multiple parentheses (text fragment absorbed)
        if speaker.count('(') >= 2 or speaker.count(')') >= 2:
            return True

        # 3. Punctuation marks (not in valid speakers)
        if any(c in speaker for c in '?!;:'):
            return True

        # 4. Unclosed quotes (OCR artifact like "(V'")
        quote_count = speaker.count("'") + speaker.count('"')
        if quote_count % 2 == 1:
            return True

        # 5. Repeated token pattern: "(TR) (TR)"
        tokens = speaker.split()
        if len(tokens) >= 2 and tokens[0] == tokens[1] and tokens[0].startswith('('):
            return True

        return False

    def correct_speaker(self, raw_speaker: str) -> str:
        """
        Normalizes and matches a speaker string against the allowlist.

        Args:
            raw_speaker: The raw speaker string from the parser.

        Returns:
            A valid callsign, or the original string if no match is found.
        """
        if not raw_speaker:
            return ""

        # normalization
        normalized = raw_speaker.upper().strip()
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "/")

        # Reject pure-digit speakers (e.g. "08" from OCR)
        if normalized.isdigit():
            return ""

        # 1. Exact match
        if normalized in self.valid_speakers_set:
            return normalized

        # 1b. Try extracting first valid token from multi-word speaker
        # Handles cases like "CC END CF TAPE", "LMP REFSSMAT?"
        if " " in normalized or len(normalized) > 6:
            tokens = raw_speaker.upper().split()
            for token in tokens:
                clean_token = "".join(ch for ch in token if ch.isalnum() or ch == "/")
                if clean_token and clean_token in self.valid_speakers_set:
                    return clean_token
                if clean_token and clean_token in self.ocr_fixes and self.ocr_fixes[clean_token] in self.valid_speakers_set:
                    return self.ocr_fixes[clean_token]

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

    def _extract_speaker_from_text(self, text: str, mission_keywords: list[str] | None = None) -> str | None:
        """
        Attempts to find a valid speaker code within the first few words of dialogue.

        Args:
            text: Dialogue line to search.
            mission_keywords: Terms used to filter false positives.

        Returns:
            Identified speaker or None.
        """
        if not text:
            return None

        tokens = text.split()
        mission_keywords = mission_keywords or []

        # Search first 4 tokens
        for i in range(min(4, len(tokens))):
            candidate_token = tokens[i]

            # Skip timestamp-like, pure digits
            if candidate_token.isdigit() or all(c in "0123456789: '-" for c in candidate_token):
                continue

            # Try two-token speaker (e.g., "SWIM 1")
            if i + 1 < len(tokens):
                two_token = f"{tokens[i]} {tokens[i+1]}".upper()
                if two_token in self.valid_speakers_set:
                    return two_token

            # Try single token via correction
            corrected = self.correct_speaker(candidate_token)
            if corrected and corrected in self.valid_speakers_set:
                # Make sure it's not a mission keyword
                if not (mission_keywords and corrected.upper() in [kw.upper() for kw in mission_keywords]):
                    return corrected

        return None

    def _infer_from_context(self, prev_comm: dict | None, last_crew: str | None) -> str | None:
        """
        Predicts the next speaker based on dialogue alternation (e.g. CC -> crew).

        Args:
            prev_comm: The block dictionary preceding the current one.
            last_crew: The last identified crew member callsign.

        Returns:
            Inferred speaker code or None.
        """
        if not prev_comm:
            return None

        prev_speaker = prev_comm.get("speaker", "")
        if not prev_speaker:
            return None

        CREW_SPEAKERS = {"CDR", "CMP", "LMP"}

        # Pattern: CC → crew alternation
        if prev_speaker == "CC":
            if last_crew:
                return last_crew
            return None

        # Pattern: crew → CC alternation
        if prev_speaker in CREW_SPEAKERS:
            return "CC"

        return None

    def _recover_speaker(
        self,
        text: str,
        prev_comm: dict | None,
        last_crew: str | None,
        mission_keywords: list[str] | None = None
    ) -> str | None:
        """
        Combines recovery strategies to identify a missing or corrupted speaker.

        Args:
            text: Dialogue text.
            prev_comm: Context block.
            last_crew: Context history.
            mission_keywords: Identification filters.

        Returns:
            Recovered speaker code or None.
        """
        # Strategy 1: Extract from text
        speaker = self._extract_speaker_from_text(text, mission_keywords)
        if speaker:
            return speaker

        # Strategy 2: Infer from context
        speaker = self._infer_from_context(prev_comm, last_crew)
        if speaker:
            return speaker

        return None

    def process_blocks(self, blocks: list[dict], mission_keywords: list[str] | None = None) -> list[dict]:
        """
        Main entry point for batch speaker correction.

        Args:
            blocks: List of communication block dictionaries.
            mission_keywords: Terms used for identification and filtering.

        Returns:
            The modified blocks with validated and recovered speakers.
        """
        if not self.valid_speakers:
            return blocks

        mission_keywords = mission_keywords or []
        prev_comm = None
        last_crew = None
        CREW_SPEAKERS = {"CDR", "CMP", "LMP"}

        for block in blocks:
            if block.get("type") != "comm":
                continue

            speaker = block.get("speaker", "").strip()
            text = block.get("text", "").strip()

            # 1. Detect garbage or missing speaker
            is_garbage = self.is_garbage_speaker(speaker, mission_keywords) if speaker else False
            is_missing = not speaker

            if is_garbage or is_missing:
                # Attempt recovery
                recovered = self._recover_speaker(text, prev_comm, last_crew, mission_keywords)
                if recovered:
                    block["speaker"] = recovered
                    # Remove speaker from text if it was extracted
                    if text.upper().startswith(recovered):
                        remaining = text[len(recovered):].lstrip()
                        if remaining:
                            block["text"] = remaining
                elif is_garbage:
                    # Clear garbage speaker if recovery failed
                    block["speaker"] = ""

            # 2. Correct existing valid speaker
            elif speaker:
                block["speaker"] = self.correct_speaker(speaker)

            # 3. Track context for next iteration
            if block.get("speaker"):
                prev_comm = block
                if block["speaker"] in CREW_SPEAKERS:
                    last_crew = block["speaker"]

        return blocks
