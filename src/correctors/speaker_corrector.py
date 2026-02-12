"""
Speaker identification and normalization logic.
"""

import re
import difflib
from loguru import logger


class SpeakerCorrector:
    """
    Handles validation and correction of speaker callsigns using an allowlist
    and mission-specific OCR error mappings.
    """

    def __init__(self, valid_speakers: list[str], ocr_fixes: dict[str, str] | None = None):
        self.valid_speakers = valid_speakers
        self.valid_speakers_set = {s.upper() for s in valid_speakers}
        self.ocr_fixes = {
            k.upper(): v.upper() if v else ""
            for k, v in (ocr_fixes or {}).items()
        }

    def is_garbage_speaker(self, speaker: str, mission_keywords: list[str] | None = None) -> bool:
        if not speaker:
            return True
        s = speaker.upper().strip()
        if s in self.valid_speakers_set:
            return False
        if s in self.ocr_fixes:
            return False
        if s.isdigit() or not any(c.isalnum() for c in s):
            return True
        if mission_keywords:
            for kw in mission_keywords:
                if s == kw.upper():
                    return True
        return False

    def correct_speaker(self, raw_speaker: str) -> str:
        if not raw_speaker:
            return ""
        normalized = raw_speaker.upper().strip()
        normalized = normalized.replace("\u00a0", " ").replace("\u200b", "").replace("\u2011", "-")
        
        if normalized in self.valid_speakers_set:
            return normalized
        if normalized in self.ocr_fixes:
            fix = self.ocr_fixes[normalized]
            return fix if (fix in self.valid_speakers_set or not fix) else ""

        tokens = normalized.split()
        if tokens:
            first = "".join(ch for ch in tokens[0] if ch.isalnum() or ch == "/")
            if first in self.valid_speakers_set:
                return first
            if first in self.ocr_fixes:
                fix = self.ocr_fixes[first]
                if not fix: return ""
                if fix in self.valid_speakers_set: return fix

        clean_norm = "".join(ch for ch in normalized if ch.isalnum() or ch == "/")
        if not clean_norm:
            return ""

        if clean_norm in self.ocr_fixes:
            fix = self.ocr_fixes[clean_norm]
            if not fix: return ""
            if fix in self.valid_speakers_set: return fix

        if clean_norm.isdigit() and len(clean_norm) <= 2:
            return ""

        if clean_norm in self.valid_speakers_set:
            return clean_norm

        candidates = self.valid_speakers
        if len(clean_norm) == 3:
            candidates = [s for s in self.valid_speakers if len(s) == 3]
            mapped = clean_norm.replace("F", "R").replace("I", "R").replace("L", "R").replace("K", "R").replace("Y", "R")
            if mapped in self.valid_speakers_set:
                return mapped
            mapped = clean_norm.replace("H", "M").replace("N", "M")
            if mapped in self.valid_speakers_set:
                return mapped

        matches = difflib.get_close_matches(clean_norm, candidates, n=1, cutoff=0.6)
        if matches:
            return matches[0]

        for valid in self.valid_speakers:
            if len(valid) >= 2 and clean_norm.startswith(valid):
                return valid

        return ""

    def _extract_speaker_from_text(self, text: str, mission_keywords: list[str] | None = None) -> str | None:
        if not text:
            return None
        tokens = text.split()
        mission_keywords = mission_keywords or []
        for i in range(min(4, len(tokens))):
            candidate_token = tokens[i]
            if candidate_token.isdigit() or all(c in "0123456789: '-" for c in candidate_token):
                continue
            if i + 1 < len(tokens):
                two_token = f"{tokens[i]} {tokens[i+1]}".upper()
                if two_token in self.valid_speakers_set:
                    return two_token
            corrected = self.correct_speaker(candidate_token)
            if corrected and corrected in self.valid_speakers_set:
                if not (mission_keywords and corrected.upper() in [kw.upper() for kw in mission_keywords]):
                    return corrected
        return None

    def _infer_from_context(self, prev_comm: dict | None, last_crew: str | None) -> str | None:
        if not prev_comm:
            return None
        prev_speaker = prev_comm.get("speaker", "")
        if not prev_speaker:
            return None
        CREW_SPEAKERS = {"CDR", "CMP", "LMP"}
        if prev_speaker == "CC":
            return last_crew if last_crew else None
        if prev_speaker in CREW_SPEAKERS:
            return "CC"
        return None

    def _recover_speaker(self, text: str, prev_comm: dict | None, last_crew: str | None, mission_keywords: list[str] | None = None) -> str | None:
        speaker = self._extract_speaker_from_text(text, mission_keywords)
        if speaker:
            return speaker
        return self._infer_from_context(prev_comm, last_crew)

    def process_blocks(self, blocks: list[dict], mission_keywords: list[str] | None = None) -> list[dict]:
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
            is_garbage = self.is_garbage_speaker(speaker, mission_keywords) if speaker else False
            is_missing = not speaker
            if is_garbage or is_missing:
                recovered = self._recover_speaker(text, prev_comm, last_crew, mission_keywords)
                if recovered:
                    block["speaker"] = recovered
                    if text.upper().startswith(recovered):
                        remaining = text[len(recovered):].lstrip()
                        if remaining:
                            block["text"] = remaining
                elif is_garbage:
                    block["speaker"] = ""
            elif speaker:
                block["speaker"] = self.correct_speaker(speaker)
            if block.get("speaker"):
                prev_comm = block
                if block["speaker"] in CREW_SPEAKERS:
                    last_crew = block["speaker"]
        return blocks
