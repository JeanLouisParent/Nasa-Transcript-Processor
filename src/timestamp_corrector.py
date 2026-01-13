"""
Timestamp Correction Module.

Normalizes and validates timestamps in transcript blocks.
Format expected: DD HH MM SS (Day, Hour, Minute, Second).
Enforces logical constraints (0-23h, 0-59m, 0-59s) and chronological order.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Regex for loose timestamp detection (allows common OCR errors like O instead of 0, ' instead of digit)
TS_CHARS_CLASS = r"[\dOI'I\)\(\]\[]"
TIMESTAMP_PATTERN = re.compile(rf"^\s*({TS_CHARS_CLASS}{{1,2}})[\s:;]+({TS_CHARS_CLASS}{{1,2}})[\s:;]+({TS_CHARS_CLASS}{{1,2}})[\s:;]+({TS_CHARS_CLASS}{{1,2}})\s*$", re.IGNORECASE)

@dataclass
class Timecode:
    day: int
    hour: int
    minute: int
    second: int

    def to_seconds(self) -> int:
        return self.day * 86400 + self.hour * 3600 + self.minute * 60 + self.second

    def __str__(self) -> str:
        return f"{self.day:02d} {self.hour:02d} {self.minute:02d} {self.second:02d}"

    @staticmethod
    def from_string(text: str) -> Optional['Timecode']:
        """Parse strict DD HH MM SS string."""
        try:
            parts = [int(p) for p in text.split()]
            if len(parts) != 4:
                return None
            return Timecode(*parts)
        except ValueError:
            return None


class TimestampCorrector:
    def __init__(self):
        self.last_valid_tc: Timecode | None = None

    def correct_string(self, text: str) -> str | None:
        """
        Attempt to fix a malformed timestamp string.
        Returns normalized 'DD HH MM SS' or None if unrecoverable.
        """
        if not text:
            return None

        # 1. Normalize separators and chars
        # Replace common OCR errors
        normalized = text.upper()
        # O/Q -> 0
        normalized = normalized.replace("O", "0").replace("Q", "0")
        # I/L/]/[ -> 1
        normalized = normalized.replace("I", "1").replace("L", "1").replace("]", "1").replace("[", "1")
        # S -> 5, B -> 8
        normalized = normalized.replace("S", "5").replace("B", "8")
        # ) / ( -> 0 (very common for 0)
        normalized = normalized.replace(")", "0").replace("(", "0")
        # ' -> can be anything, but let's assume it doesn't add a digit
        normalized = normalized.replace("'", "")
        
        # Handle "xx xx xx --" case (dashes for missing seconds)
        normalized = normalized.replace("--", "00").replace("-", "0")

        match = TIMESTAMP_PATTERN.match(normalized)
        if not match:
            return None

        try:
            # Re-clean each group to handle cases like "1)" -> "10" after replacements
            groups = []
            for g in match.groups():
                # If group is ")", it became "0" already. If it was "1)", it's "10".
                # Ensure we have digits
                clean_g = "".join(c for c in g if c.isdigit())
                if not clean_g:
                    clean_g = "0"
                groups.append(int(clean_g))
            
            d, h, m, s = groups
        except ValueError:
            return None

        # 2. Logical Validation & Correction
        
        # Seconds correction
        if s >= 60:
            # Try to fix "85" -> "05" if OCR read 0 as 8? Or just clamp?
            # Or maybe "5" became "55"?
            # Simplest heuristic: modulo 60 if close, or clamp?
            # Actually, "91" minutes is impossible. 
            pass # Keep logic simple for now

        # Validate ranges
        if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
            # Attempt fuzzy fix based on last_valid_tc if available?
            # For now, invalidate if strictly impossible
            return None

        return f"{d:02d} {h:02d} {m:02d} {s:02d}"

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Process a list of blocks to correct timestamps.
        Maintains chronological order context.
        """
        corrected_blocks = []
        self.last_valid_tc = None # Reset for page (or should we carry over from prev page?)
        # For now, reset per page because we process pages independently.
        
        for block in blocks:
            if block.get("type") != "comm" or not block.get("timestamp"):
                corrected_blocks.append(block)
                continue

            raw_ts = block["timestamp"]
            corrected_ts = self.correct_string(raw_ts)

            if corrected_ts:
                # Check consistency with previous (monotonicity)
                current_tc = Timecode.from_string(corrected_ts)
                
                if self.last_valid_tc and current_tc:
                    # If current is WAY earlier than previous (e.g. prev=Day 4, curr=Day 0), it's likely an OCR error on the Day digit
                    # Example: "04 12 00 00" followed by "01 12 00 10" -> "01" is likely "04"
                    
                    diff = current_tc.to_seconds() - self.last_valid_tc.to_seconds()
                    
                    if diff < -60: # Tolerance of 1 minute backward (maybe out of order lines?)
                        # Try to fix the Day digit first
                        if current_tc.day != self.last_valid_tc.day:
                            fixed_tc = Timecode(self.last_valid_tc.day, current_tc.hour, current_tc.minute, current_tc.second)
                            if fixed_tc.to_seconds() >= self.last_valid_tc.to_seconds():
                                corrected_ts = str(fixed_tc)
                                current_tc = fixed_tc
                
                self.last_valid_tc = current_tc
                block["timestamp"] = corrected_ts
            else:
                # Could not correct -> mark as suspect or leave as is?
                # For now, leave raw but maybe add a flag?
                pass

            corrected_blocks.append(block)

        return corrected_blocks
