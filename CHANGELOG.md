# Changelog

All notable changes to the NASA Transcript Processing Pipeline.

## [Unreleased]

### Added

#### New Commands

- **`reparse` command**: Reparse pages from stored OCR text without re-running OCR (1-2 minutes vs 3 hours)
  - Reads cached OCR text from `output/<stem>/pages/*/ocr/*.txt`
  - Re-runs parsing pipeline with current configuration
  - Applies all correctors with updated settings
  - Enables fast iteration on config/code changes

- **`postprocess` command**: Post-process existing per-page JSON without re-running OCR
  - Re-runs block processing (merging, continuations, annotations)
  - Fastest iteration for block-level logic changes (~30 seconds)

#### New Corrector: LocationCorrector

- **Module**: `src/correctors/location_corrector.py`
- Validates location fields against mission-specific allowlists
- Filters invalid annotations (LAUGHING, GARBLE, PAUSE, etc.)
- Filters timestamps misidentified as locations
- Fuzzy matching with 70% threshold
- Removes single-character OCR noise

#### Advanced Timestamp Features

- **Day Correction**: Automatically corrects impossible day values
  - `94` → `04` (0 misread as 9)
  - `55` → `05` (0 misread as 5)
  - Any day > 10 → `day % 10`

- **Hour Snapping**: Repairs OCR misreads in hour field
  - Tests adjacent hours (last_hour ± {0, 1})
  - Selects candidate with smallest positive delta (0-600s)
  - Flags correction as `corrected_hour`

- **Sequence Reset Detection**: Distinguishes intentional backward jumps from errors
  - Looks ahead for "stable run" (3+ monotonic timestamps)
  - Accepts large backward jumps if followed by stable sequence
  - Enables handling of replayed audio or alternate timelines

- **Extended Noise Normalization**: Handles more OCR artifacts
  - Added: `°` (degree) → `0`, `/` (slash) → removed
  - Added: `.` (dot), `:` (colon), `?` (question) → removed
  - Example: `"04 12 11 :6"` → `"04 12 11 06"`

- **Timestamp Preprocessing**: Pre-normalizes timestamps before parsing
  - `TIMESTAMP_ANY_SEP_RE` pattern in `preprocessor.py`
  - Matches 4 groups with any separators
  - Normalizes and pads to `DD HH MM SS` format
  - Enables recognition of heavily corrupted timestamps

#### Configuration Enhancements

- **Speaker OCR Fixes**: Global and mission-specific
  ```toml
  [pipeline_defaults.correctors.speaker_ocr_fixes]
  "CT" = "CMP"
  "CMF" = "CMP"

  [mission.11.speaker_ocr_fixes]
  "TLI" = "CC"
  ```

- **Manual Speaker Corrections**: Fix specific blocks by exact timestamp
  ```toml
  [mission.11.manual_speaker_corrections]
  "00 05 41 36" = "CMP"  # "(V'" OCR error
  ```

- **Invalid Location Annotations**: Configurable filter list
  ```toml
  [pipeline_defaults.correctors.invalid_location_annotations]
  annotations = ["LAUGHING", "LAUGHTER", "GARBLE", "PAUSE"]
  ```

- **Hyphenated Technical Terms**: Preserve exact format in lexicon
  - `S-IVB`, `S-IV`, `DELTA-V`, `AIR-TO-GROUND`, etc.
  - Prevents spell-checker from breaking hyphenated words

#### Block Processing Improvements

- **`clean_or_merge_continuations()`**: Intelligently merges continuation blocks
  - Preserves blocks with embedded timestamps
  - Detects and removes exact duplicates
  - Merges text into previous block with proper spacing

- **`merge_inline_annotations()`**: Merges standalone annotation tags
  - Identifies invalid location annotations
  - Appends to previous comm block's text
  - Converts orphans to comm blocks

- **Embedded Timestamp Detection**: Improved pattern matching
  - `EMBEDDED_TIMESTAMP_RE` with flexible separators
  - Handles 1-2 digit fields (not just 2)
  - Extended character set: includes C, S, B for OCR noise

#### Export Improvements

- **Metadata Cleaning**: Removes debug fields from merged export
  - `timestamp_correction` → removed
  - `timestamp_warning` → removed
  - `timestamp_suffix_hint` → removed
  - `_column_fill` → removed

- **Page Validation**: Skips invalid pages (page_num ≤ 0)

### Changed

#### Text Correction Order

- **CRITICAL**: Text replacements now applied AFTER spell-checking
  - Previously: clean_noise → **replacements** → spell-check → **replacements again**
  - Now: clean_noise → spell-check → **replacements once**
  - Prevents spell-checker from re-corrupting fixed patterns
  - Example: "S-IVB" stays "S-IVB" instead of becoming "Is-IVB"

#### Speaker Correction Improvements

- **Extended Search Window**: Searches first 4 tokens instead of just 1st
  - Handles cases like `"'44 CC Roger..."` (timestamp artifact before speaker)
  - Skips likely non-speaker prefixes (timestamps, numbers, punctuation)

- **Two-token Speakers**: Supports multi-word speakers
  - Example: "SWIM 1", "PRESIDENT NIXON"

#### Timestamp Patterns

- **Extended `TS_CHARS`**: `[\dOI'():?\.-]`
  - Added: `.` (dot), `-` (dash) for more OCR variations
  - Enables matching of timestamps like `"04.12.11-06"`

### Fixed

- **Embedded Timestamp Splitting**: Disabled overly aggressive `SECONDARY_EMBEDDED_RE`
  - Was matching false positives in normal text (e.g., "You're GO" with nearby numbers)
  - Caused catastrophic block fragmentation (sentences split into single words)
  - Now only uses conservative `EMBEDDED_TIMESTAMP_RE` pattern

- **Speaker Extraction**: Fixed OCR artifacts being identified as speakers
  - Examples: "minute", "Charlie", "look", "converge" no longer treated as speakers
  - Manual corrections and fuzzy matching handle edge cases

- **Day Distribution**: Corrected impossible day values in timestamps
  - Before: 41% of timestamps had day "94" (impossible for 8-day mission)
  - After: 0% impossible days, distribution matches mission timeline
  - Fixes propagate to HTML comparison (days 24, 55, 95 corrected)

### Documentation

- **New**: [QUICKSTART.md](QUICKSTART.md) — Quick start guide with examples
- **Updated**: [README.md](README.md) — Added fast iteration workflows
- **Updated**: [CONFIGURATION.md](docs/CONFIGURATION.md) — Documented all new config options
- **Updated**: [PIPELINE.md](docs/PIPELINE.md) — Added reparse/postprocess workflows
- **Updated**: [POST_PROCESSING.md](docs/POST_PROCESSING.md) — Documented LocationCorrector, advanced timestamp features
- **Updated**: [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Added LocationCorrector to module diagram

## [Previous Versions]

_History before structured changelog not available_

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
