"""
Location Correction Module.

Validates and corrects location fields against a mission-specific allowlist.
Removes invalid annotations like "(Laughing)" that were incorrectly parsed as locations.
"""

import difflib


class LocationCorrector:
    def __init__(self, valid_locations: list[str], invalid_annotations: list[str] | None = None):
        """
        Initialize with a list of valid locations.
        Args:
            valid_locations: List of allowed location codes (e.g. ["EAGLE", "COLUMBIA", "TRANQ", "EVA"])
            invalid_annotations: List of annotations to filter out (e.g. ["LAUGHING", "LAUGHTER"])
        """
        self.valid_locations = valid_locations
        self.valid_locations_set = set(valid_locations)
        self.invalid_annotations = set(invalid_annotations or [])

    def correct_location(self, raw_location: str) -> str:
        """
        Correct a location field.
        Returns the closest match from valid_locations or empty string if no valid match.
        """
        if not raw_location:
            return ""

        # Normalization
        normalized = raw_location.upper().strip()
        # Remove punctuation
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == " ")

        # 1. Exact match
        if normalized in self.valid_locations_set:
            return normalized

        # 2. Filter known invalid annotations
        # These are annotations that should not be in location field
        if normalized in self.invalid_annotations:
            return ""

        # 3. Check if it looks like a timestamp (wrongly placed)
        # e.g. "04 08 00 41"
        if any(ch.isdigit() for ch in normalized):
            # If it's mostly digits, it's probably not a location
            digit_count = sum(1 for ch in normalized if ch.isdigit())
            if digit_count >= len(normalized) * 0.5:
                return ""

        # 4. Fuzzy match against valid locations
        # This helps fix OCR errors like "EAGIE" -> "EAGLE"
        matches = difflib.get_close_matches(normalized, self.valid_locations, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        # 5. If no match and it's a single character, likely an OCR error
        if len(normalized) <= 2:
            return ""

        # Return empty string if no valid match found
        # This removes invalid locations from the output
        return ""

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Process blocks to correct location fields.
        Removes invalid locations and fixes OCR errors.
        """
        if not self.valid_locations:
            return blocks

        for block in blocks:
            if block.get("type") != "comm":
                continue

            # Correct location field
            if block.get("location"):
                corrected = self.correct_location(block["location"])
                if corrected:
                    block["location"] = corrected
                else:
                    # Remove invalid location
                    block.pop("location", None)

        return blocks
