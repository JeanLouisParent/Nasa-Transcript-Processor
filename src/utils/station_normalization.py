"""
Station name normalization utilities.

Provides functions to normalize tracking station names and match them against
mission keywords, handling OCR variations and word order differences.
"""

import re


def station_variants(station: str) -> list[str]:
    """
    Generate normalized variants of a station name by progressively shifting tokens.

    Handles cases like:
    - "GRAND BAHAMA ISLANDS" -> ["GRAND BAHAMA ISLANDS", "BAHAMA ISLANDS", "ISLANDS"]
    - "THE GRAND TURK" -> ["GRAND TURK", "TURK"]

    Args:
        station: Raw station name (e.g., "GRAND BAHAMA ISLANDS")

    Returns:
        List of variant strings, from longest to shortest
    """
    normalized = re.sub(r"[^A-Z0-9 ]", "", station.upper())
    tokens = [tok for tok in normalized.split() if tok]
    if not tokens:
        return []

    variants: list[str] = []
    max_shift = min(2, len(tokens) - 1)
    for shift in range(max_shift + 1):
        variant_tokens = tokens[shift:]
        # Strip common stop words from the beginning
        while variant_tokens and variant_tokens[0] in {"AND", "THE", "AT", "IN", "ON", "OF"}:
            variant_tokens = variant_tokens[1:]
        if variant_tokens:
            variants.append(" ".join(variant_tokens))

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            deduped.append(variant)
    return deduped


def match_station_name(station: str, mission_keywords: list[str]) -> str | None:
    """
    Match a station name against mission keywords using fuzzy logic.

    Scoring logic:
    - Exact match: instant return
    - Partial match: score based on token overlap and suffix matching
    - Returns best match if score >= 0.64

    Args:
        station: Raw station name to match
        mission_keywords: List of known station names

    Returns:
        Best matching keyword, or None if no good match found
    """
    if not mission_keywords:
        return None

    variants = station_variants(station)
    if not variants:
        return None

    # Quick exact match check
    for variant in variants:
        if variant in mission_keywords:
            return variant

    # Fuzzy scoring
    best_kw: str | None = None
    best_score = 0.0

    for variant in variants:
        var_tokens = variant.split()
        for kw in mission_keywords:
            kw_tokens = kw.split()
            score = 0.0

            # Bonus if last token matches (e.g., "ISLANDS" in both)
            if var_tokens and kw_tokens and var_tokens[-1] == kw_tokens[-1]:
                score += 0.04

            # Score based on token overlap
            overlap = len(set(var_tokens) & set(kw_tokens))
            score += 0.02 * overlap

            # Bonus for token count similarity
            if len(var_tokens) == len(kw_tokens):
                score += 0.1
            elif abs(len(var_tokens) - len(kw_tokens)) <= 1:
                score += 0.05

            # Bonus for prefix match
            if kw_tokens and var_tokens and kw_tokens[0] == var_tokens[0]:
                score += 0.03

            if score > best_score:
                best_score = score
                best_kw = kw

    # Return best match if score is above threshold
    return best_kw if best_score >= 0.64 else None
