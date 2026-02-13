"""
Global Timestamp Index.

Maintains a mapping of timestamps across all pages of a mission
to ensure chronological integrity and support cross-page corrections.
"""

import json
from pathlib import Path


class GlobalTimestampIndex:
    """
    Persistent registry of all timecodes identified across mission pages.
    
    Organizes timestamps by page number to support chronological validation
    and cross-page context for missing or corrupted timecodes.
    """

    def __init__(self, index_path: Path | None = None):
        """
        Initializes the index.

        Args:
            index_path: Optional filesystem path for JSON persistence.
        """
        self.index_path = index_path
        # Structure: { page_num: [timestamp_strings] }
        self.data: dict[int, list[str]] = {}

    def add_timestamps(self, page_num: int, timestamps: list[str]):
        """
        Registers a list of timestamps found on a specific page.

        Args:
            page_num: Zero-indexed page identifier.
            timestamps: List of DD HH MM SS strings.
        """
        self.data[page_num] = timestamps

    def get_last_timestamp_before(self, page_num: int) -> str | None:
        """
        Retrieves the chronologically latest timestamp prior to the specified page.

        Args:
            page_num: Current page index.

        Returns:
            The most recent timecode string, or None if no prior timestamps exist.
        """
        # Search backwards from page_num - 1
        for p in range(page_num - 1, -1, -1):
            if p in self.data and self.data[p]:
                return self.data[p][-1] # Last timestamp of that page
        return None

    def save(self) -> None:
        """Persists the current index state to disk as JSON."""
        if not self.index_path:
            return

        # Ensure parent directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        serializable_data = {str(k): v for k, v in self.data.items()}
        self.index_path.write_text(json.dumps(serializable_data, indent=2), encoding="utf-8")

    @staticmethod
    def load(index_path: Path) -> 'GlobalTimestampIndex':
        """
        Creates an index instance by loading data from a JSON file.

        Args:
            index_path: Path to the index file.

        Returns:
            A GlobalTimestampIndex populated with stored data.
        """
        index = GlobalTimestampIndex(index_path)
        if index_path.exists():
            try:
                raw_data = json.loads(index_path.read_text(encoding="utf-8"))
                index.data = {int(k): v for k, v in raw_data.items()}
            except Exception:
                pass
        return index
