"""
Timestamp Corrector Module.

Handles normalization, correction, and chronological sequencing of NASA timecodes.
Ensures that timestamps are strictly monotonic (always increasing).
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Timecode:
    day: int
    hour: int
    minute: int
    second: int

    def to_seconds(self) -> int:
        return self.day * 86400 + self.hour * 3600 + self.minute * 60 + self.second

    @staticmethod
    def from_seconds(total_seconds: int) -> 'Timecode':
        d, rem = divmod(total_seconds, 86400)
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        return Timecode(d, h, m, s)

    def __str__(self) -> str:
        return f"{self.day:02d} {self.hour:02d} {self.minute:02d} {self.second:02d}"

    def __lt__(self, other):
        return self.to_seconds() < other.to_seconds()

    def __le__(self, other):
        return self.to_seconds() <= other.to_seconds()


class TimestampCorrector:
    """
    Corrects and sequences timestamps to ensure chronological integrity.
    """

    def __init__(self, initial_ts: Optional[str] = None):
        self.last_valid_ts: Optional[Timecode] = None
        self.ts_pattern = re.compile(r"(\d{1,2})[\s:]+(\d{1,2})[\s:]+(\d{1,2})[\s:]+(\d{1,2})")
        
        if initial_ts:
            self.last_valid_ts = self.parse(initial_ts)

    def normalize_noise(self, text: str) -> str:
        """Replace common OCR artifacts in timestamps."""
        if not text:
            return ""
        
        # Replace common misreads
        norm = text.upper()
        norm = norm.replace("O", "0").replace("Q", "0")
        norm = norm.replace("I", "1").replace("L", "1").replace("]", "1").replace("[", "1")
        norm = norm.replace("S", "5").replace("B", "8")
        norm = norm.replace(")", "0").replace("(", "0")
        norm = norm.replace("'", "")
        
        # Handle dashes
        norm = norm.replace("--", "00").replace("-", "0")
        return norm

    def parse(self, text: str) -> Optional[Timecode]:
        """Attempt to parse a timecode string."""
        clean = self.normalize_noise(text)
        match = self.ts_pattern.search(clean)
        if not match:
            # Try to handle missing seconds
            parts = re.findall(r"\d+", clean)
            if len(parts) == 3:
                parts.append("00")
            
            if len(parts) == 4:
                try:
                    return Timecode(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
                except ValueError:
                    return None
            return None
        
        try:
            return Timecode(
                int(match.group(1)), 
                int(match.group(2)), 
                int(match.group(3)), 
                int(match.group(4))
            )
        except ValueError:
            return None

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Process a list of blocks to ensure strictly increasing timestamps.
        """
        for block in blocks:
            if block.get("type") != "comm":
                continue

            ts_str = block.get("timestamp", "")
            current_ts = self.parse(ts_str)

            if current_ts:
                # If we have a previous TS, ensure monotonicity
                if self.last_valid_ts:
                    # If current is less than or equal to previous, it's an error or duplicate
                    if current_ts <= self.last_valid_ts:
                        # Invent: Increment previous by 1 second
                        new_seconds = self.last_valid_ts.to_seconds() + 1
                        current_ts = Timecode.from_seconds(new_seconds)
                        block["timestamp_correction"] = "inferred_monotonic"
                    
                    # Check for impossible jumps (> 12 hours)
                    elif (current_ts.to_seconds() - self.last_valid_ts.to_seconds()) > 43200:
                        new_seconds = self.last_valid_ts.to_seconds() + 1
                        current_ts = Timecode.from_seconds(new_seconds)
                        block["timestamp_correction"] = "corrected_jump"

                block["timestamp"] = str(current_ts)
                self.last_valid_ts = current_ts
            else:
                # If TS is missing or unparseable but it's a COMM block, infer from last
                if self.last_valid_ts:
                    new_seconds = self.last_valid_ts.to_seconds() + 1
                    inferred_ts = Timecode.from_seconds(new_seconds)
                    block["timestamp"] = str(inferred_ts)
                    block["timestamp_correction"] = "inferred_missing"
                    self.last_valid_ts = inferred_ts

        return blocks