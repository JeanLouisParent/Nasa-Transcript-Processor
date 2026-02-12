"""
Text Correction Module.

Corrects OCR text errors using a domain-specific lexicon (Ground Truth).
Features:
- Hyphenation repair (word- break -> wordbreak)
- Noise removal
- Spelling correction using Levenshtein distance and word frequency
- Contextual correction using bigrams
"""

import difflib
import json
import re
from collections import Counter
from pathlib import Path

from src.utils.station_normalization import match_station_name

# Regex for word tokenization (keeps apostrophes inside words like "don't")
WORD_RE = re.compile(r"\b[\w']+\b")
# Regex for technical uppercase hyphen tokens (e.g., DELTA-P, S-IVB, LOI-2)
TECHNICAL_HYPHEN_RE = re.compile(r"^[A-Z0-9]{2,}-[A-Z0-9]{1,4}$")
# Regex for hyphenated words
HYPHENATED_WORD_RE = re.compile(r"\b[\w']+(?:-[\w']+)+\b")

class TextCorrector:
    """
    Applies domain-specific spelling and structural corrections to OCR text.
    
    Uses a ground-truth lexicon, bigram frequencies, and regex patterns to
    fix common NASA transcription errors (telemetry, callsigns, technical terms).
    """
    def __init__(self, lexicon_path: Path | None = None, replacements: dict[str, str] | None = None, mission_keywords: list[str] | None = None):
        """
        Initializes the corrector with a lexicon and mission-specific rules.

        Args:
            lexicon_path: Path to the JSON vocabulary file.
            replacements: Custom mission-specific regex replacement mapping.
            mission_keywords: Protected terms that should bypass fuzzy correction.
        """
        self.vocab = set()
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.mission_keywords_phrases: list[str] = []
        self._mission_keyword_tokens: dict[str, list[str]] = {}

        # Default common NASA OCR fixes (generic)
        self.replacements = {
            r"\(0\b": "GO",
            r"\bG0\b": "GO",
            # Fix Apollo Programs: TOO, P0O, T52 -> P00, P52
            r"\b[PT][0O](\d)\b": r"P0\1",
            r"\b[PT](\d{2})\b": r"P\1",
            r"\b[PT]OO\b": "P00"
        }
        # Merge with mission-specific replacements
        if replacements:
            self.replacements.update(replacements)

        if lexicon_path and lexicon_path.exists():
            self._load_lexicon(lexicon_path)

        # Add mission keywords to vocab with a high frequency to protect them
        if mission_keywords:
            self.mission_keywords_phrases = [kw.upper() for kw in mission_keywords if kw]
            self._mission_keyword_tokens = {
                kw: [tok for tok in kw.split() if tok]
                for kw in self.mission_keywords_phrases
            }
            for kw_phrase in mission_keywords:
                for word in kw_phrase.split():
                    w_lower = word.lower()
                    self.vocab.add(w_lower)
                    self.word_freq[w_lower] = max(self.word_freq.get(w_lower, 0), 100)

    def _match_mission_keyword_phrase(self, station: str) -> str | None:
        """
        Matches a raw string against protected mission phrases (e.g. tracking stations).

        Args:
            station: Raw text segment to evaluate.

        Returns:
            The canonical phrase if matched, otherwise None.
        """
        return match_station_name(station, self.mission_keywords_phrases)

    def _load_lexicon(self, path: Path):
        """
        Parses the JSON ground-truth lexicon to populate vocabulary and frequencies.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"Warning: Lexicon file not found: {path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in lexicon file {path}: {e}")
            return
        except Exception as e:
            print(f"Error reading lexicon file {path}: {e}")
            return

        # Load vocabulary and frequencies
        if "top_words" in data:
            self.word_freq.update(data["top_words"])

        if "alphabetical_vocabulary" in data:
            self.vocab.update(data["alphabetical_vocabulary"])
            # Add top_words to vocab if not already there (should be)
            self.vocab.update(self.word_freq.keys())

        if "common_bigrams" in data:
            self.bigram_freq.update(data["common_bigrams"])

    @staticmethod
    def _canonicalize_mode_token(token: str) -> str | None:
        """
        Heuristically fixes OCR misreads of AGC program modes (e.g., P00, P52).

        Args:
            token: Raw alphanumeric token.

        Returns:
            Corrected mode string (e.g. 'P00') or None if not a mode token.
        """
        if not token:
            return None
        raw = re.sub(r"[^A-Z0-9]", "", token.upper())
        if len(raw) < 2:
            return None
        if raw[0] != "P":
            return None

        char_map = {
            "O": "0",
            "Q": "0",
            "D": "0",
            "U": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "B": "8",
        }

        tail = []
        for ch in raw[1:]:
            tail.append(char_map.get(ch, ch))
            if len(tail) == 2:
                break
        if len(tail) < 2:
            tail.append("0")
        if len(tail) < 2:
            return None

        # Conservative: accept only two-digit tail after normalization.
        if not all(c.isdigit() for c in tail[:2]):
            return None
        return f"P{tail[0]}{tail[1]}"

    def normalize_structured_ocr_patterns(self, text: str) -> str:
        """
        Applies corrections to technical and telemetry-style dialogue segments.

        Fixes AGC modes, coordinates, mission-specific station callouts, 
        axes (X/Y/Z), and hardware statuses.

        Args:
            text: Dialogue line to normalize.

        Returns:
            Text with structured artifacts corrected.
        """
        if not text:
            return text

        def mode_repl(match: re.Match) -> str:
            tok = match.group(1)
            canon = self._canonicalize_mode_token(tok)
            return canon if canon else tok

        def mode_token(token: str) -> str:
            canon = self._canonicalize_mode_token(token)
            return canon if canon else token

        # Program mode tokens near ACCEPT/DATA contexts.
        text = re.sub(
            r"\b(P[A-Z0-9]{1,4})\b(?=\s+(?:AND|IN)\s+(?:ACCEPT|DATA)\b)",
            mode_repl,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(IN|AND)\s+(P[A-Z0-9]{1,4})\b(?=\s+(?:ACCEPT|DATA)\b)",
            lambda m: f"{m.group(1)} {mode_token(m.group(2))}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(HOUSTON,\s+)(P[A-Z0-9]{1,4})\b",
            lambda m: f"{m.group(1)}{mode_token(m.group(2))}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(P[A-Z0-9]{1,4})\b(?=,\s*HOUSTON\b)",
            mode_repl,
            text,
            flags=re.IGNORECASE,
        )

        # Navigation-like numeric groups: optional "(" before a 3-part tuple
        # where first field is often zero-dropped by OCR. Remove stray "(".
        def coord_repl(match: re.Match) -> str:
            a, b, c = match.group(1), match.group(2), match.group(3)
            if len(a) < 3:
                a = a.zfill(3)
            return f"{a} {b} {c}"

        text = re.sub(r"\((\d{1,3})\s+(\d{2})\s+(\d{4})(?=[,\s])", coord_repl, text)

        # DELTA-P readback marker is commonly OCR'd as DELTA-UP/DELTA-CUP.
        text = re.sub(r"\bDELTA-(?:C?UP)\b", "DELTA-P", text, flags=re.IGNORECASE)
        # LM/CM is sometimes OCR'd as IM/CM in this context.
        text = re.sub(r"\bIM/CM\b", "LM/CM", text, flags=re.IGNORECASE)

        # H/S confusion at word start before hyphenated technical terms (S-band, S-IV, S-IVB, etc.)
        text = re.sub(r"\b(?:His|Is|H)-(?=band|IV|IVB|VC|TP)\b", "S-", text, flags=re.IGNORECASE)

        # Normalize mission station annotations using configured mission keywords.
        # Example OCR: "AND GRAHAM ISLANDS (REV 1)" -> "GRAND BAHAMA ISLANDS (REV 1)".
        def station_repl(match: re.Match) -> str:
            raw_station = match.group(1).strip()
            marker = match.group(2).upper()
            number = match.group(3)
            matched = self._match_mission_keyword_phrase(raw_station)
            if not matched:
                return match.group(0)
            return f"{matched} ({marker} {number})"

        text = re.sub(
            r"\b([A-Z][A-Z0-9 ]{2,40}?)\s*\((REV|PASS)\s*(\d+)\)",
            station_repl,
            text,
            flags=re.IGNORECASE,
        )

        # Apollo-specific domain patterns
        # Direction axes: "minus Two" or "minus-Two" → "minus-X", etc.
        text = re.sub(r'\b(minus|plus)[-\s]+Two\b', r'\1-X', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(minus|plus)[-\s]+(one|l)\b', r'\1-Y', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(minus|plus)[-\s]+Zero\b', r'\1-Z', text, flags=re.IGNORECASE)

        # Quadrant indicators: "equal Bravo" → "quad Bravo"
        text = re.sub(r'\bequal\s+([A-Z][a-z]*)\b', r'quad \1', text, flags=re.IGNORECASE)

        # Spurious single letters in phrases: "a little e weak" → "a little weak"
        text = re.sub(r'\blittle\s+[a-z]\s+weak\b', 'little weak', text, flags=re.IGNORECASE)

        # Context-specific OCR letter confusion
        # "tie" → "the" in specific contexts (background, foreground, etc.)
        text = re.sub(r'\b(in|from)\s+tie\s+(background|foreground)\b', r'\1 the \2', text, flags=re.IGNORECASE)

        # "or" → "on" in question/reference contexts
        text = re.sub(r'\bquestion\s+or\s+(this|that|the)\b', r'question on \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\breference\s+or\s+(this|that|the)\b', r'reference on \1', text, flags=re.IGNORECASE)

        # Radio operations: PRESS light (not PASS light)
        # Only apply when context suggests it's a button/light
        if re.search(r'\b(turn|hit|depress|press|activate)\b', text, re.IGNORECASE):
            text = re.sub(r'\bPASS\s+light\b', 'PRESS light', text, flags=re.IGNORECASE)

        return text

    def clean_noise(self, text: str) -> str:
        """
        Removes non-essential OCR noise and fixes basic formatting issues.

        Args:
            text: Dialogue text.

        Returns:
            Cleaned text with artifacts removed and hyphenation repaired.
        """
        if not text:
            return ""

        # Replace non-ASCII OCR artifacts
        text = text.replace("\u00B7", ".")  # middle dot → period
        text = text.replace("\u00B0", "")   # degree symbol → remove (OCR error)

        # Fix hyphenation: "commu- nication" -> "communication"
        text = re.sub(r"(\w+)-\s+([a-z]+)", r"\1\2", text)

        # Merge single-letter token with following hyphenated word (e.g., "c t-off" -> "ct-off")
        text = re.sub(r"\b([A-Za-z])\s+([A-Za-z]+-[A-Za-z]+)\b", r"\1\2", text)

        # Normalize truncated decimals only in numeric contexts (e.g., "minus 4." -> "minus 4.0")
        text = re.sub(r"\b(plus|minus)\s+(\d+)\.(?=\s|$|[)\],;:])", r"\1 \2.0", text, flags=re.IGNORECASE)

        # Remove stray colons inside words (e.g., "Apollo:o" -> "Apolloo")
        text = re.sub(r"(?<=[A-Za-z]):(?=[A-Za-z])", "", text)

        # Fix single-letter tokens with parens before common directives (e.g., "G() at" -> "GO at")
        text = re.sub(r"\b([A-Z])\(\)\s+(at|for|from)\b", r"\1O \2", text)

        # Strip parentheses inside short all-caps tokens (e.g., "G()" -> "G", "G()O" -> "GO")
        def strip_paren_token(m: re.Match) -> str:
            token = m.group(0)
            letters = re.sub(r"[^A-Z]", "", token)
            return letters or token

        text = re.sub(r"\b[A-Z][A-Z()]{0,5}\b", strip_paren_token, text)

        # Note: mission-specific replacements moved to end of correct_text()
        # to apply AFTER spell checking

        # Remove prohibited chars inside words (OCR artifacts like '|', '~')
        text = re.sub(r"[|~_]", "", text)

        # Remove orphaned apostrophes (not part of contractions)
        # Keep apostrophes in: don't, it's, we're, etc.
        text = re.sub(r"\b'\s+", " ", text)  # Remove leading orphaned apostrophe with space after
        text = re.sub(r"\s+'\b", " ", text)  # Remove trailing orphaned apostrophe with space before
        text = re.sub(r"\s+'\s+", " ", text)  # Remove standalone apostrophe between spaces

        # Remove orphaned opening parentheses without closing
        # Match: "( word" but not "(word" or "(word)"
        text = re.sub(r"\(\s+(?=[A-Za-z])", "", text)  # Remove "( word" -> "word"

        # Remove orphaned closing parentheses without opening
        text = re.sub(r"(?<=[A-Za-z])\s+\)", "", text)  # Remove "word )" -> "word"
        text = re.sub(r"(?<=\s)\)\s+", " ", text)  # Remove " ) " -> " "

        return text

    def detect_embedded_apostrophe(self, word: str) -> tuple[bool, str]:
        """
        Identifies apostrophes that are likely OCR artifacts in non-contraction words.

        Args:
            word: Single token to check.

        Returns:
            Tuple of (is_error_flag, word_without_apostrophe).
        """
        if "'" not in word:
            return False, word

        # Known contractions (common ones)
        known_contractions = {
            "don't", "it's", "we're", "he's", "she's", "that's",
            "there's", "they're", "isn't", "aren't", "haven't",
            "hadn't", "hasn't", "won't", "wouldn't", "can't",
            "couldn't", "shouldn't", "i've", "we've", "you've",
            "i'll", "we'll", "you'll", "he'll", "they'll"
        }

        if word.lower() in known_contractions:
            return False, word

        # Apostrophe at position -2 (e.g., "don't") is likely a real contraction
        apostrophe_pos = word.find("'")
        if apostrophe_pos == len(word) - 2:
            return False, word

        # Pattern: letter + apostrophe + lowercase letter(s) in middle of word
        # This is likely OCR error
        if re.search(r"\w+'[a-z]+", word) and apostrophe_pos > 0:
            candidate = word.replace("'", "")
            return True, candidate

        return False, word

    def suggest_correction(
        self,
        word: str,
        prev_word: str | None = None,
        allow_short: bool = False,
        allow_known_short: bool = False
    ) -> str | None:
        """
        Finds the best lexicon match for a word using frequency and context.

        Args:
            word: Raw token to correct.
            prev_word: Contextual predecessor for bigram evaluation.
            allow_short: Whether to fuzzy match words < 3 characters.
            allow_known_short: Whether to allow matching if the word is already in vocab but short.

        Returns:
            Suggested corrected string or None if no high-confidence match is found.
        """
        word_lower = word.lower()

        # 1. Detect embedded apostrophe (OCR error)
        is_embedded, word_without_apos = self.detect_embedded_apostrophe(word)
        if is_embedded:
            # Try to correct the apostrophe-free version
            corrected = self.suggest_correction(word_without_apos, prev_word, allow_short, allow_known_short)
            if corrected:
                return corrected
            # If that didn't work, check if apostrophe-free version is in vocab
            if word_without_apos.lower() in self.vocab:
                # Restore case
                if word.istitle():
                    return word_without_apos.title()
                elif word.isupper():
                    return word_without_apos.upper()
                return word_without_apos

        # 2. Known word?
        if word_lower in self.vocab or word_lower.isdigit():
            if not (allow_known_short and len(word) < 3):
                return None

        # 2b. Known contraction?
        known_contractions = {
            "i'm", "i'll", "i've", "i'd",
            "you're", "you'll", "you've", "you'd",
            "he's", "he'll", "he'd",
            "she's", "she'll", "she'd",
            "it's", "it'll",
            "we're", "we'll", "we've", "we'd",
            "they're", "they'll", "they've", "they'd",
            "that's", "there's", "here's",
            "don't", "can't", "won't", "shouldn't", "wouldn't", "couldn't", "wasn't", "weren't", "isn't", "aren't", "haven't", "hasn't", "hadn't"
        }
        if word_lower in known_contractions:
            return None

        # 3. Short word noise?
        if len(word) < 3 and not allow_short:
            return None

        # 4. Get candidates from lexicon
        # Use a lower cutoff for short words to ensure common words aren't missed
        cutoff = 0.5 if len(word_lower) <= 3 else 0.6
        candidates = difflib.get_close_matches(word_lower, self.vocab, n=20, cutoff=cutoff)

        # 5. Rank candidates
        # Score = (Similarity Ratio * 10000) + Frequency + Bigram Bonus
        best_candidate = None
        best_score = -1

        for cand in candidates:
            if allow_known_short and len(word_lower) < 3 and cand == word_lower:
                continue
            # Avoid shortening very short words
            if len(word_lower) <= 3 and len(cand) < len(word_lower):
                continue

            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, word_lower, cand).ratio()

            # Weighted score: ratio is dominant.
            # Add a penalty for length difference to favor MASTER over WASTE for MASTEP
            len_diff = abs(len(word_lower) - len(cand))
            score = (ratio * 10000) - (len_diff * 500) + self.word_freq.get(cand, 0)

            # Context bonus (amplified for better disambiguation)
            if prev_word:
                bigram = f"{prev_word.lower()} {cand}"
                if bigram in self.bigram_freq:
                    # Context match adds significant boost for context-aware correction
                    score += self.bigram_freq[bigram] * 100

            if score > best_score:
                best_score = score
                best_candidate = cand

        if not best_candidate:
            return None

        # Restore case
        if word.istitle():
            return best_candidate.title()
        elif word.isupper():
            return best_candidate.upper()

        return best_candidate

    def _apply_generic_ocr_fixes(self, text: str) -> str:
        """
        Applies mission-agnostic OCR error corrections (e.g. '0MNI' -> 'OMNI').
        
        Args:
            text: Dialogue string.
            
        Returns:
            Text with common character confusion fixed.
        """
        if not text:
            return text
            
        # Common OCR swaps: 0 -> O inside specific words
        text = re.sub(r"\b0MNI\b", "OMNI", text)
        text = re.sub(r"\bPYR0\b", "PYRO", text)
        text = re.sub(r"\bC0MM\b", "COMM", text)
        text = re.sub(r"\b0ver\b", "Over", text, flags=re.IGNORECASE)
        text = re.sub(r"\b0ut\b", "Out", text, flags=re.IGNORECASE)
        
        # Digit/letter confusion: ll, I1, il → 11
        text = re.sub(r"\bll\b", "11", text)
        text = re.sub(r"\bI1\b", "11", text)
        text = re.sub(r"\bil\b", "11", text)
        # Recover apostrophe after digit fix: word'11 → word'll
        text = re.sub(r"(\w+)'11\b", r"\1'll", text)
        
        # Common OCR vowel confusion in compound words
        text = re.sub(r"\bfive-boy\b", "five-by", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfive-try\b", "five-by", text, flags=re.IGNORECASE)
        # Common OCR letter confusion in technical/radio terms
        text = re.sub(r"\bPFESS\b", "PRESS", text, flags=re.IGNORECASE)
        text = re.sub(r"\bPASS\s+light\b", "PRESS light", text, flags=re.IGNORECASE)
        text = re.sub(r"\bMASTEP\b", "MASTER", text, flags=re.IGNORECASE)
        
        # Typo from OCR
        text = text.replace("Unindentifiable", "Unidentifiable")
        
        return text

    def correct_text(self, text: str) -> str:
        """
        Full orchestration of the text correction pipeline.

        Applies structural cleaning, token-by-token spell checking with context,
        and mission-specific replacements.

        Args:
            text: Dialogue string.

        Returns:
            Corrected dialogue string.
        """
        # 1. Initial cleaning and generic fixes
        text = self.clean_noise(text)
        text = self._apply_generic_ocr_fixes(text)

        # 2. Tokenize while keeping delimiters to reconstruct string
        # Split by whitespace but keep delimiters in the list
        # Actually, simpler to split by words and reconstruct
        tokens = re.split(r"(\s+)", text)
        corrected_tokens = []

        prev_word = None

        for token in tokens:
            # Check if token contains a word (not just space/punctuation)
            # We strip punctuation for the lookup, but keep it for reconstruction
            word_match = re.search(r"\w+", token)

            if word_match:
                # Extract pure word part
                # Note: this simple logic assumes one word per token split by space
                # If token is "Houston." -> word is "Houston"
                # We need to be careful not to lose the "."

                # Let's use the regex to isolate the word inside the token
                def repl(m, token=token):
                    nonlocal prev_word
                    w = m.group(0)
                    allow_short = len(w) < 3 and "-" in token
                    corr = self.suggest_correction(w, prev_word, allow_short=allow_short)
                    final = corr if corr else w
                    prev_word = final  # Update context
                    return final

                def fix_hyphenated(m):
                    nonlocal prev_word
                    word = m.group(0)
                    # Preserve technical uppercase hyphen tokens (e.g., DELTA-P, S-IVB, LOI-2).
                    if TECHNICAL_HYPHEN_RE.match(word):
                        return word
                    parts = word.split("-")
                    if any(re.search(r"\d", part) for part in parts):
                        return word
                    corrected_parts = []
                    for part in parts:
                        corr = self.suggest_correction(
                            part,
                            prev_word,
                            allow_short=(len(part) < 3),
                            allow_known_short=True
                        )
                        final = corr if corr else part
                        if part.istitle():
                            final = final.title()
                        elif part.isupper():
                            final = final.upper()
                        corrected_parts.append(final)
                        prev_word = final
                    return "-".join(corrected_parts)

                token = HYPHENATED_WORD_RE.sub(fix_hyphenated, token)

                # Use [\w']+ to keep apostrophes with words (e.g., "don't", "re'ceiling")
                new_token = re.sub(r"[\w']+", repl, token)
                corrected_tokens.append(new_token)
            else:
                # Just whitespace/punctuation
                corrected_tokens.append(token)

        corrected = "".join(corrected_tokens)

        # Enforce canonical header line after token correction.
        if len(corrected) <= 90:
            upper = corrected.upper().replace("0", "O")
            if "VOICE" in upper and "TRANS" in upper:
                target = "AIR-TO-GROUND VOICE TRANSCRIPTION"
                norm_text = re.sub(r"[^A-Z]", "", upper)
                norm_target = re.sub(r"[^A-Z]", "", target)
                if difflib.SequenceMatcher(None, norm_text, norm_target).ratio() >= 0.6:
                    return target
            if "1" in upper and len(upper) <= 25:
                norm_text = re.sub(r"[^A-Z0-9]", "", upper)
                if difflib.SequenceMatcher(None, norm_text, "GOSSNET1").ratio() >= 0.6:
                    return "(GOSS NET 1)"

        # Apply mission-specific replacements AFTER spell checking
        # This ensures patterns like "Is-IVB" → "S-IVB" are applied after
        # spell checking has finished modifying the text
        for pattern, replacement in self.replacements.items():
            corrected = re.sub(pattern, replacement, corrected)

        corrected = self.normalize_structured_ocr_patterns(corrected)

        return corrected

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Main entry point for batch text correction.

        Args:
            blocks: List of block dictionaries containing 'text' fields.

        Returns:
            The modified list of blocks with corrected text.
        """
        if not self.vocab:
            return blocks

        for block in blocks:
            # Apply to 'text' fields (comm, continuation, annotation)
            if block.get("type") == "footer":
                continue
            if "text" in block and block["text"]:
                block["text"] = self.correct_text(block["text"])

        return blocks
