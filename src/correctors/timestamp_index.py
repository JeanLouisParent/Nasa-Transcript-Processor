"""
Global Timestamp Index.

Maintains a mapping of timestamps across all pages of a mission
to ensure chronological integrity and support cross-page corrections.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List


class GlobalTimestampIndex:
    """
    Registry of all timestamps found in a mission, organized by page.
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path
        # Structure: { page_num: [timestamp_strings] }
        self.data: Dict[int, List[str]] = {}

    def add_timestamps(self, page_num: int, timestamps: List[str]):
        """Register all timestamps found on a specific page."""
        self.data[page_num] = timestamps

    def get_last_timestamp_before(self, page_num: int) -> Optional[str]:
        """
        Look back through previous pages to find the last valid timestamp.
        """
        # Search backwards from page_num - 1
        for p in range(page_num - 1, -1, -1):
            if p in self.data and self.data[p]:
                return self.data[p][-1] # Last timestamp of that page
        return None

    def save(self):
        """Save index to disk."""
        if not self.index_path:
            return
        
        # Ensure parent directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_data = {str(k): v for k, v in self.data.items()}
        self.index_path.write_text(json.dumps(serializable_data, indent=2), encoding="utf-8")

    @staticmethod
    def load(index_path: Path) -> 'GlobalTimestampIndex':
        """Load index from disk."""
        index = GlobalTimestampIndex(index_path)
        if index_path.exists():
            try:
                raw_data = json.loads(index_path.read_text(encoding="utf-8"))
                index.data = {int(k): v for k, v in raw_data.items()}
            except Exception:
                pass
        return index
