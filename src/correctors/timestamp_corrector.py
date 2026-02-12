"""
Timestamp Corrector Module.

Handles normalization, correction, and chronological sequencing of NASA timecodes.
Prefers monotonic ordering while tolerating small backward OCR slips.
"""

import re
from dataclasses import dataclass


@dataclass
class Timecode:
    """
    Representation of a NASA mission timecode (GET - Ground Elapsed Time).
    """
    day: int
    hour: int
    minute: int
    second: int

    def to_seconds(self) -> int:
        """Converts the timecode to total elapsed seconds."""
        return self.day * 86400 + self.hour * 3600 + self.minute * 60 + self.second

    @staticmethod
    def from_seconds(total_seconds: int) -> 'Timecode':
        """Creates a Timecode object from total elapsed seconds."""
        d, rem = divmod(total_seconds, 86400)
        h, rem = divmod(rem, 3600)
        m, s = divmod(rem, 60)
        return Timecode(d, h, m, s)

    def __str__(self) -> str:
        """Returns the space-separated DD HH MM SS representation."""
        return f"{self.day:02d} {self.hour:02d} {self.minute:02d} {self.second:02d}"

    def __lt__(self, other):
        return self.to_seconds() < other.to_seconds()

    def __le__(self, other):
        return self.to_seconds() <= other.to_seconds()


class TimestampCorrector:
    """
    Ensures chronological integrity of timestamps across communication blocks.
    
    Detects OCR noise, fixes common character misreads, and enforces monotonic
    sequencing while tolerating small backward slips.
    """

    def __init__(self, initial_ts: str | None = None, backward_tolerance_s: int = 300):
        """
        Initializes the corrector with an optional starting timecode.

        Args:
            initial_ts: DD HH MM SS string to use as the baseline.
            backward_tolerance_s: Threshold for allowing small time regressions.
        """
        self.last_valid_ts: Timecode | None = None
        self.ts_pattern = re.compile(r"(\d{1,2})[\s:]+(\d{1,2})[\s:]+(\d{1,2})[\s:]+(\d{1,2})")
        self.backward_tolerance_s = backward_tolerance_s
        self.reset_threshold_s = 3600
        self.seen_any_ts = False
        self.stable_run_len = 3
        self.stable_run_window = 6

        if initial_ts:
            self.last_valid_ts = self.parse(initial_ts)

    def normalize_noise(self, text: str) -> str:
        """
        Cleans OCR artifacts from raw timestamp strings.

        Args:
            text: Raw timestamp text.

        Returns:
            String containing only digits and spaces.
        """
        if not text:
            return ""

        # Replace common misreads
        norm = text.upper()
        norm = norm.replace("O", "0").replace("Q", "0")
        norm = norm.replace("I", "1").replace("L", "1").replace("]", "1").replace("[", "1")
        norm = norm.replace("S", "5").replace("B", "8")
        norm = norm.replace(")", "0").replace("(", "0")
        norm = norm.replace("'", "")
        norm = norm.replace("°", "0")
        # Remove OCR noise characters that appear in timestamps
        norm = norm.replace(":", "").replace("?", "").replace(".", "").replace("/", "")

        # Handle dashes
        norm = norm.replace("--", "00").replace("-", "0")
        return norm

    def parse(self, text: str) -> Timecode | None:
        """
        Attempts to convert a raw string into a Timecode object.

        Args:
            text: Raw timestamp string.

        Returns:
            Timecode object or None if parsing fails.
        """
        clean = self.normalize_noise(text)
        match = self.ts_pattern.search(clean)
        if not match:
            # Try to handle missing seconds
            parts = re.findall(r"\d+", clean)
            if len(parts) == 3:
                parts.append("00")

            if len(parts) == 4:
                try:
                    day = int(parts[0])
                    hour = int(parts[1])
                    minute = int(parts[2])
                    second = int(parts[3])

                    # Apply day correction even in fallback parsing
                    if day == 94:
                        day = 4
                    elif day == 55:
                        day = 5
                    elif day > 10:
                        day = day % 10

                    # Reject impossible time fields
                    if hour > 23 or minute > 59 or second > 59:
                        return None

                    return Timecode(day, hour, minute, second)
                except ValueError:
                    return None
            return None

        try:
            day = int(match.group(1))
            hour = int(match.group(2))
            minute = int(match.group(3))
            second = int(match.group(4))

            # Correct common OCR errors for day field
            # Apollo missions lasted ~8-12 days max, so day > 10 is likely an OCR error
            if day == 94:  # Common: "04" misread as "94" (0→9)
                day = 4
            elif day == 55:  # Common: "05" misread as "55" (0→5)
                day = 5
            elif day > 10:  # Any other impossible day value
                # Try to infer: if first digit is wrong, use second digit
                day = day % 10

            # Reject impossible time fields (PAD data / countdowns parsed as timestamps)
            if hour > 23 or minute > 59 or second > 59:
                return None

            return Timecode(day, hour, minute, second)
        except ValueError:
            return None

    def _hour_snap(self, current_ts: Timecode, last_ts: Timecode) -> Timecode | None:
        """
        Heuristically repairs OCR hour misreads by snapping to the last known hour.

        Args:
            current_ts: The problematic current timecode.
            last_ts: The baseline timecode.

        Returns:
            A corrected Timecode object or None if no plausible correction is found.
        """
        if current_ts.day != last_ts.day:
            return None

        candidates: list[tuple[int, Timecode]] = []
        for delta_h in (0, -1, 1):
            hour = last_ts.hour + delta_h
            if 0 <= hour <= 23:
                candidate = Timecode(current_ts.day, hour, current_ts.minute, current_ts.second)
                delta = candidate.to_seconds() - last_ts.to_seconds()
                if 0 < delta <= 600:
                    candidates.append((delta, candidate))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Corrects timestamps across a batch of blocks to ensure chronological order.

        Args:
            blocks: List of communication block dictionaries.

        Returns:
            Modified blocks with corrected and sequential timestamps.
        """
        parsed_ts: list[Timecode | None] = []
        for block in blocks:
            if block.get("type") != "comm":
                parsed_ts.append(None)
                continue
            parsed_ts.append(self.parse(block.get("timestamp", "")))

        def has_stable_run(start_idx: int) -> bool:
            candidates: list[Timecode] = []
            for j in range(start_idx, min(len(blocks), start_idx + self.stable_run_window)):
                ts = parsed_ts[j]
                if ts:
                    candidates.append(ts)
                if len(candidates) >= self.stable_run_len:
                    break
            if len(candidates) < self.stable_run_len:
                return False
            for a, b in zip(candidates, candidates[1:]):
                if b.to_seconds() <= a.to_seconds():
                    return False
                if b.to_seconds() - a.to_seconds() > 600:
                    return False
            return True

        for idx, block in enumerate(blocks):
            if block.get("type") != "comm":
                continue

            current_ts = parsed_ts[idx]
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
                if self.last_valid_ts and not self.seen_any_ts:
                    delta = current_ts.to_seconds() - self.last_valid_ts.to_seconds()
                    if delta < -self.reset_threshold_s:
                        self.last_valid_ts = current_ts
                        block["timestamp"] = str(current_ts)
                        block["timestamp_correction"] = "sequence_reset"
                        self.seen_any_ts = True
                        block.pop("timestamp_suffix_hint", None)
                        continue
                if self.last_valid_ts:
                    delta = current_ts.to_seconds() - self.last_valid_ts.to_seconds()
                    if delta < -self.reset_threshold_s and has_stable_run(idx):
                        self.last_valid_ts = current_ts
                        block["timestamp"] = str(current_ts)
                        block["timestamp_correction"] = "sequence_reset"
                        self.seen_any_ts = True
                        block.pop("timestamp_suffix_hint", None)
                        continue
                # If we have a previous TS, ensure monotonicity
                if self.last_valid_ts:
                    delta = current_ts.to_seconds() - self.last_valid_ts.to_seconds()
                    if delta > 3600:
                        snapped = self._hour_snap(current_ts, self.last_valid_ts)
                        if snapped:
                            current_ts = snapped
                            block["timestamp_correction"] = "corrected_hour"
                            self.last_valid_ts = current_ts
                            block["timestamp"] = str(current_ts)
                            continue
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
                        snapped = None
                        if abs(delta) > self.backward_tolerance_s:
                            snapped = self._hour_snap(current_ts, self.last_valid_ts)
                        if snapped:
                            current_ts = snapped
                            block["timestamp_correction"] = "corrected_hour"
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
                self.seen_any_ts = True
            else:
                # If TS is missing or unparseable but it's a COMM block, infer from last
                if self.last_valid_ts:
                    new_seconds = self.last_valid_ts.to_seconds() + 1
                    inferred_ts = Timecode.from_seconds(new_seconds)
                    block["timestamp"] = str(inferred_ts)
                    block["timestamp_correction"] = "inferred_missing"
                    self.last_valid_ts = inferred_ts
                    self.seen_any_ts = True
            block.pop("timestamp_suffix_hint", None)

        return blocks
