"""
Global constants used across the transcript processing pipeline.
"""

# Speaker categories
CREW_SPEAKERS = frozenset({"CDR", "CMP", "LMP"})
"""Crew member speakers (Commander, Command Module Pilot, Lunar Module Pilot)"""

GROUND_SPEAKERS = frozenset({"CC", "CT", "MCC", "PAO"})
"""Ground control speakers (Capsule Communicator, Control, Mission Control, Public Affairs)"""

SHIP_SPEAKERS = frozenset({"SC", "MS", "HORNET", "SWIM 1", "SWIM 2", "RECOVERY"})
"""Ship and recovery team speakers"""

SPECIAL_SPEAKERS = frozenset({"MSFN", "PRESIDENT NIXON"})
"""Special/external speakers"""

# All valid speakers combined
ALL_VALID_SPEAKERS = CREW_SPEAKERS | GROUND_SPEAKERS | SHIP_SPEAKERS | SPECIAL_SPEAKERS

# Technical keywords that look like speakers but aren't
TECHNICAL_KEYWORDS = frozenset({"NOUN", "VERB", "TIG", "LOI", "TEI", "TPI", "CDH", "PDI"})
"""DSKY commands and maneuver codes that should not be treated as speakers"""

# Timestamp constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MAX_SECONDS = 59
"""Maximum valid seconds in a timestamp (0-59)"""
MAX_MINUTES = 59
"""Maximum valid minutes in a timestamp (0-59)"""
MAX_HOURS = 23
"""Maximum valid hours in a timestamp (0-23)"""

# Timestamp correction constants
TIMESTAMP_JUMP_THRESHOLD = 40
"""Seconds to add when detecting a potential timestamp gap (used in timestamp inference)"""

# Text cleaning
MIN_VALID_TEXT_LENGTH = 2
"""Minimum length for a valid comm text (shorter = likely noise)"""

MAX_NOISE_PUNCTUATION = 3
"""Maximum number of punctuation-only characters before considering it noise"""

MIN_FOOTER_LENGTH = 15
"""Minimum character length for a valid footer"""

TIMESTAMP_SEARCH_START = 12
"""Character position to start searching for embedded timestamps (skips initial timestamp)"""

MAX_WORD_COMPARISON_LENGTH = 12
"""Maximum number of words to compare when detecting repeated phrases"""

# Image processing constants
MAX_PIXEL_VALUE = 255
"""Maximum pixel value for 8-bit images"""

POINTS_PER_INCH = 72
"""Standard PDF/PostScript points per inch conversion factor"""

IMAGE_ROTATION_ANGLE_90 = 90
"""Standard 90-degree rotation angle"""

IMAGE_ROTATION_ANGLE_45 = 45
"""45-degree angle threshold for rotation correction"""

MAX_SPOT_SIZE = 15
"""Maximum size (in pixels) for noise spots to remove from images"""

# Pattern matching
MAX_STATION_NAME_LENGTH = 40
"""Maximum length for station/location name patterns"""

MAX_TEXT_DISPLAY_LENGTH = 90
"""Maximum length for text display in logs/UI"""
