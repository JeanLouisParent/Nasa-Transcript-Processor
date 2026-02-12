"""
Regex patterns and constants for transcript parsing.
"""

import re

# Character set for timestamps (digits + common OCR errors)
# Includes: digits, O/I (0/1 errors), ' (apostrophe), () (parentheses), : (colon), ? (question mark), . (dot), - (dash)
TS_CHARS = r"[\dOI'():?\.-]"

# Timestamp patterns
TIMESTAMP_STRICT_RE = re.compile(rf"^{TS_CHARS}{{2}}(?:\s+{TS_CHARS}{{2}}){{2,3}}$")
TIMESTAMP_PREFIX_RE = re.compile(rf"^({TS_CHARS}{{2}}(?:\s+{TS_CHARS}{{2}}){{2,3}})\b")
TIMESTAMP_EMBEDDED_RE = re.compile(rf"\s+({TS_CHARS}{{2}}(?:\s+{TS_CHARS}{{2}}){{2,3}})\b")

# Revision markers
REV_EMBEDDED_RE = re.compile(r"(\([RE][FV]\s+\d+\))", re.IGNORECASE)

# Speaker patterns (accept lowercase for OCR errors like "LMp" -> "LMP")
SPEAKER_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{1,8}\??(?:/[A-Za-z0-9]{1,8})?\??$")
SPEAKER_LINE_RE = re.compile(
    r"^[A-Za-z0-9]{1,8}\??(?:/[A-Za-z0-9]{1,8})?\??"
    r"(?:\s+[A-Za-z0-9]{1,8}\??(?:/[A-Za-z0-9]{1,8})?\??){0,2}"
    r"(?:\s*\([A-Za-z0-9]+\))?$"
)
SPEAKER_PAREN_RE = re.compile(r"^\(([A-Za-z0-9]+)\)$")

# Location pattern
LOCATION_PAREN_RE = re.compile(r"^\s*\(([A-Z0-9\s]+)\)\s*$")

# Header patterns
HEADER_PAGE_RE = re.compile(r"\b(?:PAGE|PLAY|LAY)\s*(\d{1,4})\b", re.IGNORECASE)
HEADER_TAPE_RE = re.compile(r"\bTAPE\s*([0-9]{1,2}\s*/\s*[0-9]{1,2})\b", re.IGNORECASE)
HEADER_PAGE_ONLY_RE = re.compile(r"^\s*(?:PAGE|PLAY|LAY)\s*\d{1,4}\s*$", re.IGNORECASE)
HEADER_TAPE_ONLY_RE = re.compile(r"^\s*TAPE\s*\d{1,2}\s*/\s*\d{1,2}\s*$", re.IGNORECASE)
HEADER_TAPE_SIMPLE_RE = re.compile(r"^\s*TAPE\s*\d{1,4}\s*$", re.IGNORECASE)
HEADER_TAPE_PAGE_ONLY_RE = re.compile(
    r"^\s*(?:TAPE\s*\d{1,2}\s*/\s*\d{1,2}\s+(?:PAGE|PLAY|LAY)\s*\d{1,4}"
    r"|(?:PAGE|PLAY|LAY)\s*\d{1,4}\s+TAPE\s*\d{1,2}\s*/\s*\d{1,2})\s*$",
    re.IGNORECASE,
)

# Keywords
HEADER_KEYWORDS = ("GOSS", "NET", "TAPE", "PAGE", "APOLLO", "AIR-TO-GROUND")
END_OF_TAPE_KEYWORD = "END OF TAPE"
TRANSITION_KEYWORDS = (
    "REST PERIOD",
    "NO COMMUNICATIONS",
    "NO COMMUNICATION",
    "LOSS OF SIGNAL",
    "AOS",
    "LOS",
    "SILENCE",
)

# Special patterns
LINE_TAG_RE = re.compile(r"^\[(HEADER|FOOTER|ANNOTATION|COMM|META)\]\s*", re.IGNORECASE)
LUNAR_REV_RE = re.compile(r"^--\s*(BEGIN|END)\s+LUNAR\s+REV\s+(\d+)\b", re.IGNORECASE)
REST_PERIOD_RE = re.compile(r"\bREST\s+PERIOD\b", re.IGNORECASE)
NO_COMM_RE = re.compile(r"\bNO\s+COMMUNICATIONS?\b", re.IGNORECASE)
GOSS_NET_RE = re.compile(r"^\(?\s*GOSS\s+NET\s+1\s*\)?$", re.IGNORECASE)

PAREN_RADIO_CALL_RE = re.compile(
    r"\(\s*(OVER|OUT|ROGER|COPY|WILCO|GO AHEAD|SAY AGAIN|STAND BY|STANDBY)\s*([.!?])?\s*\)",
    re.IGNORECASE,
)
ANNOTATION_FRAGMENT_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-./ ]{0,24}\.?$")
