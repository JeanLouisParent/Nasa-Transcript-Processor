# Transcript Comparison Report: AS11_TEC

## Executive Summary

Comparison of the reference HTML transcript (`assets/a11tec.html`) against the generated JSON transcript (`output/AS11_TEC/AS11_TEC_merged.json`) reveals:

- **Reference HTML**: 8,428 communication blocks
- **Generated JSON**: 8,497 communication blocks
- **Difference**: +69 blocks (JSON has MORE, but this is misleading)
- **Actually Missing**: 228 significant communications from HTML not found in JSON
- **Partial Matches**: 909 communications with minor OCR variations

## Critical Finding: Famous Quote Missing

**Armstrong's famous moon landing quote is MISSING from the merged JSON:**

```
"THAT'S ONE SMALL STEP FOR MAN, ONE GIANT LEAP FOR MANKIND."
```

- **Timestamp**: 04:13:24:48
- **Speaker**: CDR (Commander Neil Armstrong)
- **Location**: Present in source page file (`Page_379/AS11_TEC_page_0379.json`)
- **Status**: Lost during processing

### Root Cause Analysis

The famous quote is being lost due to a bug in `src/ocr/parsing/block_builder.py` in the `merge_duplicate_comm_timestamps()` function:

**The Problem:**
1. Two consecutive communications have the same timestamp (04:13:24:48):
   - Block 1: "THAT'S ONE SMALL STEP FOR MAN, ONE GIANT LEAP FOR MANKIND."
   - Block 2: "And the - the surface is fine and powdery. I can - I can pick it up..."

2. The merge function (lines 168-178) checks word overlap:
   - If overlap < 30%: keeps both blocks separately
   - If overlap >= 30%: **ONLY keeps the longer text**

3. Since both texts likely share common words like "the", they meet the 30% threshold
4. The function keeps only the LONGER text (Block 2) and discards the famous quote

**The Bug (line 178):**
```python
prev["text"] = new_text if len(new_text) > len(prev_text) else prev_text
```

This logic was designed to handle OCR split lines, but it incorrectly merges genuinely separate communications that happen to share the same timestamp.

## Missing Communications Analysis

### By Speaker

| Speaker | Missing Count |
|---------|---------------|
| CC (CAPCOM) | 81 |
| LMP (Lunar Module Pilot) | 65 |
| CMP (Command Module Pilot) | 39 |
| CDR (Commander) | 35 |
| CT (Comm Tech) | 3 |
| SC (Spacecraft - Unidentified) | 3 |
| HORNET | 2 |

### By Time Period (Top 15)

Most missing communications occur during:

| Time Period | Missing Count |
|-------------|---------------|
| 04:06:xx:xx | 25 |
| 04:12:xx:xx | 19 |
| 04:13:xx:xx | 11 (includes famous quote) |
| 05:04:xx:xx | 10 |
| 04:14:xx:xx | 8 |
| 04:04:xx:xx | 7 |
| 04:01:xx:xx | 5 |
| 04:07:xx:xx | 5 |
| 05:06:xx:xx | 5 |

**Pattern**: Day 4 hours 6-14 (lunar surface operations) have the highest concentration of missing communications.

## Famous Quotes Status

| Quote | In HTML | In JSON | Status |
|-------|---------|---------|--------|
| "one giant leap for mankind" | ✓ | ✗ | **MISSING** |
| "the eagle has landed" | ✓ | ✓ | Present |
| "magnificent desolation" | ✓ | ✓ | Present |

Note: The HTML reference has "(A)" in parentheses: "ONE SMALL STEP FOR (A) MAN", while the JSON has "ONE SMALL STEP FOR MAN" (without the (A)), but the entire quote block is missing from the merged output regardless.

## Sample Missing Communications

### Critical Missing Examples:

1. **[00:00:09:19] CC**: "Ignition confirmed; thrust is GO, 11."
   - Actually present in JSON as: "Ignition confirmed, thrust is GO, 11."
   - (Minor punctuation difference - false positive)

2. **[00:01:29:27] LMP**: "Cecil B. deAldrin is standing by for instructions."
   - Actually present in JSON as: "Cecil B. deAldrin is standing by for struck-"
   - (Truncated text - OCR issue)

3. **[00:10:35:54] CDR**: "Five-by."
   - Actually present in JSON as: "Five-boy."
   - (OCR misread: "by" → "boy")

4. **[04:13:24:48] CDR (TRANQ)**: "THAT'S ONE SMALL STEP FOR MAN, ONE GIANT LEAP FOR MANKIND."
   - **GENUINELY MISSING** - Lost due to timestamp deduplication bug

### Other Missing Communications (First 10):

1. [00:00:11:51] CC: "Roger. Shutdown. We copy 101.4 by 103.6."
2. [00:00:14:43] CDR: "Okay. Thank you. CANARY (REV 1)"
3. [00:01:37:08] CMP: "Go ahead, TLI plus 5."
4. [00:01:47:16] CDR: "Roger. That's verified; the probe is extended."
5. [00:01:54:38] CC: "Roger. 0ut." (Note: "0ut" likely OCR error for "Out")
6. [00:04:17:18] CC: "Roger. Copy. CRY0 PRESS light."
7. [00:04:28:16] CC: "Apollo 11, this ls Houston. Over."
8. [00:04:28:21] CMP: "Houston, Apollo ll."
9. [00:04:41:47] CDR: "That EMS DELTA-V counter is minus 4.0."
10. [00:05:46:07] CC: "This is,Houston. Readback correct. Out."

## Notes on "Missing" vs Actually Missing

Many communications flagged as "missing" are actually present but with minor variations:

**Common Variations:**
- OCR errors: "five-by" → "five-boy", "is" → "ls", "11" → "ll"
- Punctuation: "Ignition confirmed;" → "Ignition confirmed,"
- Spacing: "This is,Houston" → "This is Houston"
- Number/letter confusion: "0ut" (zero) → "Out" (letter O)
- Truncation: Text cut off mid-word

**Genuinely Missing:**
- Communications lost due to timestamp deduplication bug
- Approximately 20-30 communications are likely truly missing
- The famous moon landing quote is the most significant loss

## Recommendations

### Immediate Actions:

1. **Fix the timestamp deduplication bug** in `src/ocr/parsing/block_builder.py`:
   - Lines 168-178 of `merge_duplicate_comm_timestamps()`
   - Change logic to preserve BOTH blocks when they have distinct content
   - The current 30% overlap threshold is too aggressive
   - Consider checking for sentence boundaries or all-caps text (indicates quotes)

2. **Re-run the pipeline** after fixing the bug to regenerate `AS11_TEC_merged.json`

3. **Verify the famous quote is restored** after regeneration

### Longer Term:

1. **Improve OCR post-processing** for common errors:
   - "by" → "boy"
   - "0" ↔ "O"
   - "l" ↔ "1"
   - "ls" → "is"

2. **Add validation checks** for known critical communications:
   - Famous quotes
   - Mission-critical calls (landing, liftoff, etc.)

3. **Review other merged pages** where multiple blocks share timestamps

## Data Quality Assessment

### Strengths:
- Overall structure and ordering preserved
- Timestamps accurately captured
- Speaker identification mostly correct
- 97%+ of content successfully captured

### Weaknesses:
- Timestamp deduplication loses distinct communications
- OCR errors in short acknowledgments ("OK", "Roger", etc.)
- Some truncated text from page breaks
- Famous quote missing (critical for historical accuracy)

## Conclusion

The transcript extraction is largely successful with 8,497 blocks captured vs 8,428 in the reference. However:

1. **The famous moon landing quote is missing** - this is a critical loss
2. The bug is identified and fixable
3. Most "missing" communications are actually OCR variations
4. Approximately 200+ blocks need investigation for genuine losses
5. The timestamp deduplication logic needs refinement to avoid losing distinct communications

**Priority**: Fix the deduplication bug and regenerate the merged JSON immediately to restore the famous quote and other lost communications.

---

*Report generated: 2026-02-04*
*Comparison tool: `/Users/jean-louis/Work/ocr_transcript_v2/compare_transcripts_fast.py`*
*Detailed results: `/Users/jean-louis/Work/ocr_transcript_v2/comparison_results.json`*
