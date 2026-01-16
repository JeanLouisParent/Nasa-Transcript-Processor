"""
Text Correction Module.

Corrects OCR text errors using a domain-specific lexicon (Ground Truth).
Features:
- Hyphenation repair (word- break -> wordbreak)
- Noise removal
- Spelling correction using Levenshtein distance and word frequency
- Contextual correction using bigrams
"""

import json
import re
import difflib
from pathlib import Path
from collections import Counter

# Regex for word tokenization (keeps apostrophes inside words like "don't")
WORD_RE = re.compile(r"\b[\w']+\b")

class TextCorrector:
    def __init__(self, lexicon_path: Path = None, replacements: dict[str, str] = None, mission_keywords: list[str] = None):
        """
        Initialize text corrector with a lexicon and custom replacements.
        Args:
            lexicon_path: Path to the lexicon JSON file.
            replacements: Dictionary of regex patterns to replacement strings.
        """
        self.vocab = set()
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        
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
            for kw_phrase in mission_keywords:
                for word in kw_phrase.split():
                    w_lower = word.lower()
                    self.vocab.add(w_lower)
                    self.word_freq[w_lower] = max(self.word_freq.get(w_lower, 0), 100)

    def _load_lexicon(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            
            # Load vocabulary and frequencies
            if "top_words" in data:
                self.word_freq.update(data["top_words"])
            
            if "alphabetical_vocabulary" in data:
                self.vocab.update(data["alphabetical_vocabulary"])
                # Add top_words to vocab if not already there (should be)
                self.vocab.update(self.word_freq.keys())

            if "common_bigrams" in data:
                self.bigram_freq.update(data["common_bigrams"])
                
        except Exception as e:
            print(f"Error loading lexicon: {e}")

    def clean_noise(self, text: str) -> str:
        """Remove common OCR noise and fix hyphenation."""
        if not text:
            return ""

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

        # Apply mission-specific replacements
        for pattern, replacement in self.replacements.items():
            text = re.sub(pattern, replacement, text)

        # Remove prohibited chars inside words (OCR artifacts like '|', '~')
        text = re.sub(r"[|~_]", "", text)
        
        return text

    def suggest_correction(
        self,
        word: str,
        prev_word: str = None,
        allow_short: bool = False,
        allow_known_short: bool = False
    ) -> str | None:
        """
        Suggest a correction for a word using frequency and context (bigrams).
        """
        word_lower = word.lower()
        
        # 1. Known word?
        if word_lower in self.vocab or word_lower.isdigit():
            if not (allow_known_short and len(word) < 3):
                return None
            
        # 2. Short word noise?
        if len(word) < 3 and not allow_short:
            return None

        # 3. Get candidates from lexicon
        # Use a lower cutoff for short words to ensure common words aren't missed
        cutoff = 0.5 if len(word_lower) <= 3 else 0.6
        candidates = difflib.get_close_matches(word_lower, self.vocab, n=20, cutoff=cutoff)
        
        # 4. Rank candidates
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
            
            # Context bonus (keep it low so ratio remains dominant)
            if prev_word:
                bigram = f"{prev_word.lower()} {cand}"
                if bigram in self.bigram_freq:
                    # Context match adds a small boost
                    score += self.bigram_freq[bigram] * 10 
            
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

    def correct_text(self, text: str) -> str:
        """
        Apply full correction pipeline to a text string with context awareness.
        """
        # 1. Cleaning
        text = self.clean_noise(text)
        
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
                def repl(m):
                    nonlocal prev_word
                    w = m.group(0)
                    allow_short = len(w) < 3 and "-" in token
                    corr = self.suggest_correction(w, prev_word, allow_short=allow_short)
                    final = corr if corr else w
                    prev_word = final # Update context
                    return final

                def fix_hyphenated(m):
                    nonlocal prev_word
                    word = m.group(0)
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

                token = re.sub(r"\b[\w']+(?:-[\w']+)+\b", fix_hyphenated, token)

                new_token = re.sub(r"\w+", repl, token)
                corrected_tokens.append(new_token)
            else:
                # Just whitespace/punctuation
                corrected_tokens.append(token)
        
        return "".join(corrected_tokens)

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Process blocks to correct text content.
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
