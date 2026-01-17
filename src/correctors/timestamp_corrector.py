"""
Timestamp Corrector Module.

Handles normalization, correction, and chronological sequencing of NASA timecodes.
Prefers monotonic ordering while tolerating small backward OCR slips.
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
    Corrects and sequences timestamps while preserving likely-valid OCR reads.
    """

    def __init__(self, initial_ts: Optional[str] = None, backward_tolerance_s: int = 300):
        self.last_valid_ts: Optional[Timecode] = None
        self.ts_pattern = re.compile(r"(\d{1,2})[\s:]+(\d{1,2})[\s:]+(\d{1,2})[\s:]+(\d{1,2})")
        self.backward_tolerance_s = backward_tolerance_s
        
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
            suffix_hint = block.get("timestamp_suffix_hint")

            if current_ts and suffix_hint and self.last_valid_ts:
                try:
                    hint_digit = int(suffix_hint)
                except ValueError:
                    hint_digit = None
                if hint_digit is not None:
                    if (
                        current_ts.day == self.last_valid_ts.day
                        and current_ts.hour == self.last_valid_ts.hour
                        and current_ts.minute == self.last_valid_ts.minute
                        and current_ts.second == 0
                    ):
                        tens = self.last_valid_ts.second // 10
                        candidate_second = tens * 10 + hint_digit
                        if 0 <= candidate_second <= 59:
                            candidate_ts = Timecode(
                                current_ts.day,
                                current_ts.hour,
                                current_ts.minute,
                                candidate_second,
                            )
                            delta = candidate_ts.to_seconds() - self.last_valid_ts.to_seconds()
                            if 0 < delta <= 20:
                                current_ts = candidate_ts
                                block["timestamp_correction"] = "inferred_suffix"

            if current_ts:
                # If we have a previous TS, ensure monotonicity
                if self.last_valid_ts:
                    delta = current_ts.to_seconds() - self.last_valid_ts.to_seconds()
                    # If current is less than or equal to previous, it's likely an OCR error
                    if delta <= 0:
                        if (
                            current_ts.day == self.last_valid_ts.day
                            and current_ts.hour == self.last_valid_ts.hour
                            and current_ts.minute == self.last_valid_ts.minute
                            and current_ts.second < 20
                            and self.last_valid_ts.second < 20
                        ):
                            candidate_second = current_ts.second + 40
                            if candidate_second <= 59:
                                candidate_ts = Timecode(
                                    current_ts.day,
                                    current_ts.hour,
                                    current_ts.minute,
                                    candidate_second,
                                )
                                candidate_delta = candidate_ts.to_seconds() - self.last_valid_ts.to_seconds()
                                if 0 < candidate_delta <= 60:
                                    current_ts = candidate_ts
                                    block["timestamp_correction"] = "inferred_tens"
                                    self.last_valid_ts = current_ts
                                    block["timestamp"] = str(current_ts)
                                    continue
                        if abs(delta) > self.backward_tolerance_s:
                            # Invent: Increment previous by 1 second
                            new_seconds = self.last_valid_ts.to_seconds() + 1
                            current_ts = Timecode.from_seconds(new_seconds)
                            block["timestamp_correction"] = "inferred_monotonic"
                            self.last_valid_ts = current_ts
                        else:
                            # Keep raw timestamp but don't move the global cursor backwards
                            block["timestamp_correction"] = "out_of_order"
                    # Check for impossible jumps (> 12 hours) within the same day
                    elif delta > 43200 and current_ts.day == self.last_valid_ts.day:
                        new_seconds = self.last_valid_ts.to_seconds() + 1
                        current_ts = Timecode.from_seconds(new_seconds)
                        block["timestamp_correction"] = "corrected_jump"
                        self.last_valid_ts = current_ts
                    else:
                        self.last_valid_ts = current_ts

                block["timestamp"] = str(current_ts)
                if not self.last_valid_ts:
                    self.last_valid_ts = current_ts
            else:
                # If TS is missing or unparseable but it's a COMM block, infer from last
                if self.last_valid_ts:
                    new_seconds = self.last_valid_ts.to_seconds() + 1
                    inferred_ts = Timecode.from_seconds(new_seconds)
                    block["timestamp"] = str(inferred_ts)
                    block["timestamp_correction"] = "inferred_missing"
                    self.last_valid_ts = inferred_ts
            block.pop("timestamp_suffix_hint", None)

        return blocks
