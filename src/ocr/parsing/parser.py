"""
Transcript Parser Class.

Replaces the monolithic state machine function with a structured class.
"""

import re
import difflib
from typing import Any

from .patterns import (
    TIMESTAMP_STRICT_RE,
    TIMESTAMP_PREFIX_RE,
    SPEAKER_TOKEN_RE,
    SPEAKER_LINE_RE,
    SPEAKER_PAREN_RE,
    LOCATION_PAREN_RE,
    HEADER_PAGE_RE,
    HEADER_TAPE_RE,
    HEADER_PAGE_ONLY_RE,
    HEADER_TAPE_ONLY_RE,
    HEADER_TAPE_PAGE_ONLY_RE,
    END_OF_TAPE_KEYWORD,
    TRANSITION_KEYWORDS,
)
from .utils import fuzzy_find
from .preprocessor import preprocess_lines


class TranscriptParser:
    """
    State machine implementation for converting OCR text lines into structured objects.
    
    Handles multi-column layout detection, speaker identification, and 
    technical metadata (headers, footers, annotations).
    """
    def __init__(
        self,
        page_num: int,
        mission_keywords: list[str] | None = None,
        valid_speakers: list[str] | None = None
    ):
        """
        Initializes the parser for a specific page.

        Args:
            page_num: Zero-indexed page number from the source document.
            mission_keywords: Terms used to prioritize technical identifications.
            valid_speakers: List of expected speaker callsigns.
        """
        self.page_num = page_num
        self.mission_keywords = mission_keywords
        self.valid_speakers = valid_speakers

        # Output rows
        self.rows: list[dict[str, Any]] = []
        self.line_index = 0

        # Internal state
        self.pending_ts = ""
        self.pending_speaker = ""
        self.pending_location = ""
        self.pending_text: list[str] = []
        self.pending_force_comm = False
        self.pending_ts_hint: str | None = None

        # Timestamp run detection
        self.timestamp_only_run = 0
        self.timestamp_run_start_idx: int | None = None
        self.timestamp_list_mode = False
        self.timestamp_list_start_idx: int | None = None
        self.timestamp_list_row_idx: int | None = None

        # Context tracking
        self.prev_comm_like = False
        self.saw_comm_or_ts = False
        self.first_ts_idx: int | None = None

    def detect_and_remove_footer(self, lines: list[dict]) -> tuple[list[dict], bool]:
        """
        Identifies and strips standardized footer text from preprocessed lines.

        Args:
            lines: List of preprocessed line objects.

        Returns:
            Tuple of (cleaned_lines, has_footer_flag).
        """
        CANONICAL_FOOTER_TEXT = "THREE ASTERISKS DENOTE CLIPPING OF WORDS AND PHRASES"
        FUZZY_THRESHOLD = 0.75

        has_footer = False
        indices_to_remove = []

        for idx, entry in enumerate(lines):
            text = entry["text"]
            upper = text.upper().strip()
            normalized = re.sub(r"[^A-Z ]", "", upper)
            similarity = difflib.SequenceMatcher(None, normalized, CANONICAL_FOOTER_TEXT).ratio()

            if similarity >= FUZZY_THRESHOLD:
                has_footer = True
                indices_to_remove.append(idx)
                if idx > 0:
                    prev_text = lines[idx - 1]["text"].strip()
                    if re.match(r"^\*{3,}$", prev_text):
                        indices_to_remove.append(idx - 1)

        cleaned_lines = [entry for i, entry in enumerate(lines) if i not in indices_to_remove]
        return cleaned_lines, has_footer

    def header_keyword_match(self, line: str) -> bool:
        """
        Checks if a line contains mission-standard header keywords.

        Args:
            line: Text string to evaluate.

        Returns:
            True if the line likely represents a transcription header.
        """
        upper = line.upper()
        if re.search(r"\bGOSS\b", upper):
            return True
        if re.search(r"\bNET\b", upper):
            return True
        if re.search(r"\bTAPE\b", upper):
            return True
        if re.search(r"\bPAGE\b", upper):
            return True
        if re.search(r"\bAPOLLO\b", upper):
            return True
        if "AIR-TO-GROUND" in upper or "AIR TO GROUND" in upper:
            return True
        return False

    def transition_keyword_match(self, line: str) -> bool:
        """
        Checks if a line contains markers for mission phase transitions.

        Args:
            line: Text string to evaluate.

        Returns:
            True if the line indicates an AOS/LOS or similar transition.
        """
        upper = line.upper()
        for kw in TRANSITION_KEYWORDS:
            if len(kw) <= 3:
                if re.search(rf"\b{re.escape(kw)}\b", upper):
                    return True
            else:
                if kw in upper:
                    return True
        return False

    def is_technical_data_not_timestamp(self, line: str) -> bool:
        """
        Differentiates between valid timestamps and technical parameters (e.g. TIG).

        Args:
            line: Text string to evaluate.

        Returns:
            True if the line starts with technical data that mimics a timecode.
        """
        # TIG (Time of Ignition) at START of line followed by numbers
        if re.match(r'^\s*TIG\s+\d+\s+\d{2}\s+\d{2}\s+\d{2}\s+\d{2}', line, re.IGNORECASE):
            return True
        return False

    def normalize_speaker(self, speaker: str) -> str:
        """
        Validates and fixes common OCR errors in speaker codes.

        Args:
            speaker: Raw speaker string.

        Returns:
            Corrected speaker identifier, or the original if no match is found.
        """
        if not speaker or not self.valid_speakers:
            return speaker

        # NOUN, VERB followed by numbers are DSKY commands, not speakers
        if re.match(r'^(NOUN|VERB)\s*\d', speaker.strip(), re.IGNORECASE):
            return ""

        speaker_cleaned = speaker.strip().upper().rstrip("?")
        valid_speaker_set = {s.upper() for s in self.valid_speakers}

        if speaker_cleaned in valid_speaker_set:
            return speaker_cleaned

        matches = difflib.get_close_matches(speaker_cleaned, [s.upper() for s in self.valid_speakers], n=1, cutoff=0.65)
        if matches:
            return matches[0]

        return speaker

    def take_speaker_tokens(self, tokens: list[str]) -> tuple[str, list[str]]:
        """
        Consumes tokens from a list to identify a valid speaker name.

        Args:
            tokens: List of text words to evaluate.

        Returns:
            Tuple of (identified_speaker_string, remaining_tokens_list).
        """
        speaker_tokens = []
        valid_speaker_set = {s.upper() for s in (self.valid_speakers or [])}

        while tokens and len(speaker_tokens) < 3:
            if valid_speaker_set:
                max_n = min(3, len(tokens))
                found_multi = False
                for n in range(max_n, 1, -1):
                    candidate = " ".join(tokens[:n]).upper().strip()
                    if candidate in valid_speaker_set:
                        return candidate, tokens[n:]
                if found_multi:
                    break

            token = tokens[0]
            if len(token) == 1:
                break
            if SPEAKER_TOKEN_RE.match(token):
                cleaned = token.rstrip("?")
                if self.valid_speakers:
                    cleaned_upper = cleaned.upper()
                    is_exact_match = cleaned_upper in valid_speaker_set
                    matches = difflib.get_close_matches(cleaned_upper, self.valid_speakers, n=1, cutoff=0.7) if not is_exact_match else [cleaned_upper]

                    if not matches:
                        prefix = f"{cleaned_upper} "
                        if any(v.startswith(prefix) for v in valid_speaker_set):
                            break
                        break
                    speaker_tokens.append(matches[0])
                    tokens.pop(0)
                    continue
                speaker_tokens.append(cleaned)
                tokens.pop(0)
                continue
            break
        return " ".join(speaker_tokens), tokens

    def flush_pending(self) -> None:
        """
        Finalizes the currently accumulated communication data into a row object.
        """
        if not self.pending_ts and not self.pending_speaker and not self.pending_text:
            return

        self.line_index += 1
        row = {
            "page": self.page_num + 1,
            "line": self.line_index,
            "type": "comm" if (self.pending_ts or self.pending_force_comm) else "text",
            "timestamp": self.pending_ts,
            "speaker": self.pending_speaker,
            "location": self.pending_location,
            "text": " ".join(self.pending_text).strip(),
        }
        if self.pending_ts_hint:
            row["timestamp_suffix_hint"] = self.pending_ts_hint

        self.rows.append(row)

        # Reset state
        self.pending_ts = ""
        self.pending_speaker = ""
        self.pending_location = ""
        self.pending_text = []
        self.pending_force_comm = False
        self.pending_ts_hint = None

    def parse(self, text: str) -> tuple[list[dict], bool]:
        """
        Converts raw OCR text into structured rows.

        Args:
            text: Multi-line string from OCR engine.

        Returns:
            Tuple of (parsed_rows_list, has_footer_flag).
        """
        lines = preprocess_lines(text, self.mission_keywords)
        if not lines:
            return [], False

        lines, has_footer = self.detect_and_remove_footer(lines)

        self.first_ts_idx = next(
            (
                i
                for i, entry in enumerate(lines)
                if TIMESTAMP_STRICT_RE.match(entry["text"]) or TIMESTAMP_PREFIX_RE.match(entry["text"])
            ),
            None,
        )

        for idx, entry in enumerate(lines):
            self._process_line(idx, entry, lines)

        self.flush_pending()
        self._post_process_timestamp_list(lines)

        return self.rows, has_footer

    def _process_line(self, idx: int, entry: dict, all_lines: list[dict]):
        """
        Core logic for processing a single OCR line and updating parser state.

        Delegates to specialized handlers for different line types.
        """
        line = entry["text"]
        forced_type = entry["forced"]
        upper = line.upper()
        has_lower = any(c.islower() for c in line)
        has_timestamp = (TIMESTAMP_PREFIX_RE.match(line) or TIMESTAMP_STRICT_RE.match(line)) and not self.is_technical_data_not_timestamp(line)

        # Adjust forced type based on context
        forced_type = self._adjust_forced_type(forced_type, line, upper, has_lower, has_timestamp, idx)

        # Early exit for header-only lines
        if self._is_header_only(line) and not has_timestamp:
            return

        # Handle location-only lines
        if self._handle_location_only(line):
            return

        # Handle standalone speaker lines
        if self._handle_standalone_speaker(line):
            return

        # Handle END OF TAPE markers
        if forced_type is None and fuzzy_find(line, END_OF_TAPE_KEYWORD):
            self._append_meta_row(line)
            self.prev_comm_like = False
            return

        # Handle forced metadata types (header, footer, annotation, meta)
        if forced_type in ("header", "footer", "annotation", "meta"):
            self._handle_metadata_line(forced_type, line, upper, has_lower)
            return

        if forced_type == "comm":
            self.pending_force_comm = True
            self.saw_comm_or_ts = True

        # Handle standalone timestamp
        if self._handle_standalone_timestamp(line):
            return

        # Handle timestamp with prefix (e.g., "00 01 23 45 CC Roger")
        if self._handle_prefix_timestamp(line):
            return

        # Flush pending if in timestamp list mode
        if self.timestamp_list_mode and self.pending_ts and not self.pending_speaker and not self.pending_text:
            self.flush_pending()
        if not self.timestamp_list_mode and self.timestamp_only_run:
            self.timestamp_only_run = 0
            self.timestamp_run_start_idx = None

        # Skip standalone speaker/location lines if no pending timestamp
        if (
            not self.pending_ts
            and not self.timestamp_list_mode
            and (SPEAKER_LINE_RE.match(line) or LOCATION_PAREN_RE.match(line))
        ):
            return

        # Handle timestamp list mode data assignment
        if self._handle_timestamp_list_data(line, forced_type):
            return

        # Detect header/annotation without forced type
        if self._handle_implicit_metadata(line, upper, has_lower, idx, forced_type):
            return

        # Accumulate text for pending communication
        self._accumulate_pending_text(line)

    def _adjust_forced_type(self, forced_type: str | None, line: str, upper: str, has_lower: bool, has_timestamp: bool, idx: int) -> str | None:
        """Adjusts the forced type based on context and line characteristics."""
        # In timestamp list mode, only keep specific forced types
        if self.timestamp_list_mode and forced_type in ("header", "footer", "annotation", "meta"):
            if not (
                (forced_type == "footer" and line.lstrip().startswith("***"))
                or (forced_type == "meta" and fuzzy_find(line, END_OF_TAPE_KEYWORD))
                or (forced_type == "header" and self._is_header_only(line))
            ):
                forced_type = None

        # Timestamp overrides metadata type
        if forced_type in ("header", "footer", "annotation", "meta") and has_timestamp:
            forced_type = "comm"

        # Meta type refinement
        if forced_type == "meta":
            if idx <= (self.first_ts_idx or -1) and (HEADER_PAGE_RE.search(upper) or HEADER_TAPE_RE.search(upper)):
                forced_type = "header"
            elif self.pending_ts or self.prev_comm_like or has_lower:
                forced_type = "comm"

        # Short uppercase lines after comm are likely comm continuations
        if (
            forced_type is None
            and self.prev_comm_like
            and not has_lower
            and len(line.strip()) <= 8
            and not line.lstrip().startswith("***")
        ):
            forced_type = "comm"

        # Annotation refinement
        if forced_type == "annotation":
            if TIMESTAMP_PREFIX_RE.match(line) or TIMESTAMP_STRICT_RE.match(line):
                forced_type = "comm"
            elif "(REV" not in upper and "(RFV" not in upper:
                if has_lower:
                    forced_type = "comm"
                elif len(line.strip()) <= 4:
                    forced_type = "comm"

        # Header with REV/RFV is actually annotation
        if forced_type == "header" and ("(REV" in upper or "(RFV" in upper):
            forced_type = "annotation"

        # Footer refinement
        if forced_type == "footer":
            if fuzzy_find(line, END_OF_TAPE_KEYWORD):
                forced_type = "meta"
            elif not line.lstrip().startswith("***"):
                forced_type = "comm"

        return forced_type

    def _is_header_only(self, line: str) -> bool:
        """Checks if line is header-only (PAGE X, TAPE Y patterns)."""
        return bool(
            HEADER_PAGE_ONLY_RE.match(line)
            or HEADER_TAPE_ONLY_RE.match(line)
            or HEADER_TAPE_PAGE_ONLY_RE.match(line)
        )

    def _handle_location_only(self, line: str) -> bool:
        """Handles lines that contain only a location tag like (CAP COMM)."""
        location_match = LOCATION_PAREN_RE.match(line)
        if not location_match:
            return False

        location_value = location_match.group(1).strip()
        if self.pending_ts:
            self.pending_location = location_value
            return True

        if self.rows and self.rows[-1].get("type") == "comm" and not self.rows[-1].get("location"):
            self.rows[-1]["location"] = location_value
            return True

        return False

    def _handle_standalone_speaker(self, line: str) -> bool:
        """Handles standalone speaker lines that attach to previous comm."""
        if SPEAKER_LINE_RE.match(line) and not self.pending_ts and not self.timestamp_list_mode:
            if self.rows and self.rows[-1].get("type") == "comm" and not self.rows[-1].get("speaker"):
                self.rows[-1]["speaker"] = self.normalize_speaker(line)
                return True
        return False

    def _handle_metadata_line(self, forced_type: str, line: str, upper: str, has_lower: bool):
        """Appends header, footer, annotation, or meta rows."""
        self.flush_pending()
        self.line_index += 1
        self.rows.append(
            {
                "page": self.page_num + 1,
                "line": self.line_index,
                "type": "meta" if forced_type == "meta" else forced_type,
                "timestamp": "",
                "speaker": "",
                "location": "",
                "text": line,
            }
        )
        self.prev_comm_like = forced_type == "annotation" and has_lower

    def _append_meta_row(self, line: str):
        """Appends a generic meta row (e.g., END OF TAPE)."""
        self.flush_pending()
        self.line_index += 1
        self.rows.append(
            {
                "page": self.page_num + 1,
                "line": self.line_index,
                "type": "meta",
                "timestamp": "",
                "speaker": "",
                "location": "",
                "text": line,
            }
        )

    def _handle_standalone_timestamp(self, line: str) -> bool:
        """Handles lines containing only a timestamp."""
        if not TIMESTAMP_STRICT_RE.match(line) or self.is_technical_data_not_timestamp(line):
            return False

        self.flush_pending()
        if self.timestamp_only_run == 0:
            self.timestamp_run_start_idx = len(self.rows)
        self.pending_ts = line
        self.timestamp_only_run += 1

        # Detect timestamp list mode after 5 consecutive timestamps
        if self.timestamp_only_run >= 5:
            self.timestamp_list_mode = True
            if self.timestamp_list_start_idx is None:
                self.timestamp_list_start_idx = self.timestamp_run_start_idx
                self.timestamp_list_row_idx = self.timestamp_list_start_idx

        self.prev_comm_like = True
        self.saw_comm_or_ts = True
        return True

    def _handle_prefix_timestamp(self, line: str) -> bool:
        """Handles lines starting with timestamp followed by speaker/text."""
        if self.is_technical_data_not_timestamp(line):
            return False

        prefix_match = TIMESTAMP_PREFIX_RE.match(line)
        if not prefix_match:
            return False

        self.flush_pending()
        if self.timestamp_only_run == 0:
            self.timestamp_run_start_idx = len(self.rows)

        self.pending_ts = prefix_match.group(1)
        self.timestamp_only_run += 1

        # Detect timestamp list mode
        if self.timestamp_only_run >= 5:
            self.timestamp_list_mode = True
            if self.timestamp_list_start_idx is None:
                self.timestamp_list_start_idx = self.timestamp_run_start_idx
                self.timestamp_list_row_idx = self.timestamp_list_start_idx

        # Parse remainder (speaker, location, text)
        remainder = line[len(self.pending_ts):].strip()

        # Handle +X suffix for seconds (e.g., "00 01 23 +4" -> "00 01 23 44")
        if remainder.startswith("+") and len(self.pending_ts.split()) == 3:
            plus_match = re.match(r"^\+(\d)\b", remainder)
            if plus_match:
                self.pending_ts = f"{self.pending_ts} 4{plus_match.group(1)}"
                remainder = remainder[plus_match.end():].strip()

        if remainder:
            tokens = remainder.split()

            # Extract timestamp suffix hint (e.g., "12" before speaker)
            if (
                len(tokens) >= 2
                and tokens[0].isdigit()
                and len(tokens[0]) <= 2
                and SPEAKER_TOKEN_RE.match(tokens[1])
            ):
                self.pending_ts_hint = tokens.pop(0)

            # Extract speaker
            speaker, tokens = self.take_speaker_tokens(tokens)
            if speaker:
                self.pending_speaker = speaker

                # Extract location from speaker (e.g., "CC (CAPCOM)")
                loc_match = re.search(r"\(([^)]+)\)", self.pending_speaker)
                if loc_match:
                    self.pending_location = loc_match.group(1)
                    self.pending_speaker = self.pending_speaker[:loc_match.start()].strip()
                elif tokens and re.match(r"^\([^)]+\)$", tokens[0]):
                    self.pending_location = tokens.pop(0).strip("()")

            # Remaining tokens are text
            if tokens:
                self.pending_text.append(" ".join(tokens))

        self.prev_comm_like = True
        self.saw_comm_or_ts = True
        return True

    def _handle_timestamp_list_data(self, line: str, forced_type: str | None) -> bool:
        """Assigns speaker/location/text to existing timestamp rows in list mode."""
        if not self.timestamp_list_mode or self.pending_ts or forced_type is not None:
            return False

        if self.timestamp_list_row_idx is None:
            return False

        # Ensure row index is valid
        if self.timestamp_list_row_idx >= len(self.rows):
            self.timestamp_list_row_idx = len(self.rows) - 1 if self.rows else None

        if self.timestamp_list_row_idx is None or self.timestamp_list_row_idx < 0:
            return False

        row = self.rows[self.timestamp_list_row_idx]

        # Assign speaker
        if SPEAKER_LINE_RE.match(line):
            if row.get("speaker"):
                self._advance_timestamp_list_row()
                row = self.rows[self.timestamp_list_row_idx]
            row["speaker"] = self.normalize_speaker(line)
        # Assign location
        elif LOCATION_PAREN_RE.match(line):
            if row.get("location"):
                self._advance_timestamp_list_row()
                row = self.rows[self.timestamp_list_row_idx]
            row["location"] = line.strip("()").strip()
        # Assign text
        else:
            if row.get("text") and (row.get("speaker") or row.get("location")):
                self._advance_timestamp_list_row()
                row = self.rows[self.timestamp_list_row_idx]
            row["text"] = (row.get("text", "") + " " + line).strip()

        return True

    def _advance_timestamp_list_row(self):
        """Moves to next row in timestamp list mode."""
        self.timestamp_list_row_idx += 1
        if self.timestamp_list_row_idx >= len(self.rows):
            self.timestamp_list_row_idx = len(self.rows) - 1

    def _handle_implicit_metadata(self, line: str, upper: str, has_lower: bool, idx: int, forced_type: str | None) -> bool:
        """Detects and handles header/annotation lines without forced type."""
        if forced_type is not None or self.pending_ts:
            return False

        is_header = (
            self.first_ts_idx is not None
            and idx <= self.first_ts_idx
            and self.header_keyword_match(line)
        )
        is_annotation = (
            "(REV" in upper
            or "(RFV" in upper
            or (
                self.mission_keywords
                and not has_lower
                and len(line.strip()) <= 30
                and any(fuzzy_find(line, kw) for kw in self.mission_keywords)
            )
        )
        is_end_of_tape = fuzzy_find(line, END_OF_TAPE_KEYWORD)
        is_transition = self.transition_keyword_match(line)

        if not (is_header or is_annotation or is_end_of_tape or is_transition):
            return False

        self.flush_pending()
        self.line_index += 1
        self.rows.append(
            {
                "page": self.page_num + 1,
                "line": self.line_index,
                "type": "meta" if (is_end_of_tape or is_transition) else ("header" if is_header else "annotation"),
                "timestamp": "",
                "speaker": "",
                "location": "",
                "text": line,
            }
        )
        return True

    def _accumulate_pending_text(self, line: str):
        """Accumulates text for the current pending communication."""
        # Handle comm lines before first timestamp
        if not self.saw_comm_or_ts and not TIMESTAMP_PREFIX_RE.match(line) and not TIMESTAMP_STRICT_RE.match(line):
            self.pending_text.append(line)
            return

        if self.pending_ts:
            # Check for location tag at start
            loc_match = re.match(r"^\s*\(([A-Z0-9\s]+)\)\s*", line)
            if loc_match:
                self.pending_location = loc_match.group(1).strip()
                line = line[loc_match.end():].strip()
                if not line:
                    return

            # Check if line is a speaker
            if SPEAKER_LINE_RE.match(line) or SPEAKER_PAREN_RE.match(line):
                new_speaker = self.normalize_speaker(line) if not self.pending_speaker else line
                self.pending_speaker = f"{self.pending_speaker} {new_speaker}".strip() if self.pending_speaker else new_speaker
                return

            self.pending_text.append(line)
            self.prev_comm_like = True
            self.saw_comm_or_ts = True
            return

        self.pending_text.append(line)
        self.prev_comm_like = self.prev_comm_like or self.pending_force_comm

    def _post_process_timestamp_list(self, all_lines: list[dict]):
        """
        Cleans up and matches speakers/text for batches of timestamps found in isolation.
        """
        if not self.timestamp_list_mode:
            return

        ts_row_indices = [i for i, row in enumerate(self.rows) if row.get("timestamp")]
        start_idx = self.timestamp_list_start_idx
        if start_idx is None and ts_row_indices:
            start_idx = ts_row_indices[0]
        if start_idx is not None and ts_row_indices:
            has_speaker = any(self.rows[i].get("speaker") for i in ts_row_indices)
            if not has_speaker:
                ts_line_indices = [
                    i
                    for i, entry in enumerate(all_lines)
                    if TIMESTAMP_STRICT_RE.match(entry["text"]) or TIMESTAMP_PREFIX_RE.match(entry["text"])
                ]
                if ts_line_indices:
                    row_idx = start_idx
                    for i in range(start_idx, len(self.rows)):
                        self.rows[i]["speaker"] = ""
                        self.rows[i]["location"] = ""
                        self.rows[i]["text"] = ""
                    for entry in all_lines[ts_line_indices[-1] + 1 :]:
                        if row_idx >= len(self.rows):
                            break
                        line = entry["text"]
                        if (
                            HEADER_PAGE_ONLY_RE.match(line)
                            or HEADER_TAPE_ONLY_RE.match(line)
                            or HEADER_TAPE_PAGE_ONLY_RE.match(line)
                        ):
                            continue
                        row = self.rows[row_idx]
                        if SPEAKER_LINE_RE.match(line):
                            if row.get("speaker"):
                                row_idx += 1
                                if row_idx >= len(self.rows):
                                    break
                                row = self.rows[row_idx]
                            row["speaker"] = self.normalize_speaker(line)
                        elif LOCATION_PAREN_RE.match(line):
                            if row.get("location"):
                                row_idx += 1
                                if row_idx >= len(self.rows):
                                    break
                                row = self.rows[row_idx]
                            row["location"] = line.strip("()").strip()
                        else:
                            if row.get("text") and (row.get("speaker") or row.get("location")):
                                row_idx += 1
                                if row_idx >= len(self.rows):
                                    break
                                row = self.rows[row_idx]
                            row["text"] = (row.get("text", "") + " " + line).strip()
                    while self.rows and not (
                        self.rows[-1].get("timestamp")
                        or self.rows[-1].get("speaker")
                        or self.rows[-1].get("location")
                        or self.rows[-1].get("text")
                    ):
                        self.rows.pop()
