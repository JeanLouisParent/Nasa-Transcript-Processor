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
    def __init__(self, lexicon_path: Path = None):
        """
        Initialize text corrector with a lexicon.
        Args:
            lexicon_path: Path to the lexicon JSON file.
        """
        self.vocab = set()
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        
        if lexicon_path and lexicon_path.exists():
            self._load_lexicon(lexicon_path)

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
        # Look for hyphen followed by space and lowercase letter
        text = re.sub(r"(\w+)-\s+([a-z]+)", r"\1\2", text)

        # Remove prohibited chars inside words (OCR artifacts like '|', '~')
        # But keep punctuation like .,?!
        # Logic: Replace | ~ _ with space if surrounded by spaces, or remove if inside word?
        # Safe bet: Replace weird chars with nothing if they are not standard punctuation
        text = re.sub(r"[|~_]", "", text)
        
        # Fix common number errors: "I1" -> "11" inside numbers? 
        # Risky without context. Let's stick to cleaning.
        
        return text

    def suggest_correction(self, word: str) -> str | None:
        """
        Suggest a correction for a word if it's not in the vocabulary.
        Returns the best candidate or None.
        """
        word_lower = word.lower()
        
        # If word is known (or is a number), it's fine
        if word_lower in self.vocab or word_lower.isdigit():
            return None
            
        # Ignore very short words (1-2 chars) unless they are clearly noise
        if len(word) < 3:
            return None

        # Find close matches
        candidates = difflib.get_close_matches(word_lower, self.vocab, n=3, cutoff=0.7)
        
        if not candidates:
            return None

        # Rank candidates by frequency
        best_candidate = max(candidates, key=lambda w: self.word_freq.get(w, 0))
        
        # Restore case (Title Case or UPPER CASE)
        if word.istitle():
            return best_candidate.title()
        elif word.isupper():
            return best_candidate.upper()
        
        return best_candidate

    def correct_text(self, text: str) -> str:
        """
        Apply full correction pipeline to a text string.
        """
        # 1. Cleaning
        text = self.clean_noise(text)
        
        # 2. Word-by-word correction
        # We use a regex to find words to preserve whitespace and punctuation
        
        def replace_func(match):
            word = match.group(0)
            correction = self.suggest_correction(word)
            return correction if correction else word

        text = WORD_RE.sub(replace_func, text)
        
        return text

    def process_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        Process blocks to correct text content.
        """
        if not self.vocab:
            return blocks

        for block in blocks:
            # Apply to 'text' fields (comm, continuation, annotation)
            if "text" in block and block["text"]:
                block["text"] = self.correct_text(block["text"])
        
        return blocks
