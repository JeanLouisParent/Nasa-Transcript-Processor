"""
Block filtering and manipulation utilities.

Common helpers for working with transcript block lists.
"""

def filter_by_type(blocks: list[dict], block_type: str) -> list[dict]:
    """
    Filters blocks by type.

    Args:
        blocks: List of block dictionaries
        block_type: Type to filter ("comm", "footer", "annotation", "meta", etc.)

    Returns:
        Filtered list containing only blocks of specified type
    """
    return [b for b in blocks if b.get("type") == block_type]


def filter_comm_blocks(blocks: list[dict]) -> list[dict]:
    """
    Filters to only communication blocks.

    Args:
        blocks: List of block dictionaries

    Returns:
        List of comm blocks
    """
    return filter_by_type(blocks, "comm")


def filter_with_timestamp(blocks: list[dict]) -> list[dict]:
    """
    Filters to blocks that have a timestamp.

    Args:
        blocks: List of block dictionaries

    Returns:
        List of blocks with non-empty timestamps
    """
    return [b for b in blocks if b.get("timestamp")]


def filter_with_speaker(blocks: list[dict]) -> list[dict]:
    """
    Filters to blocks that have a speaker.

    Args:
        blocks: List of block dictionaries

    Returns:
        List of blocks with non-empty speakers
    """
    return [b for b in blocks if b.get("speaker")]


def filter_comm_with_timestamp(blocks: list[dict]) -> list[dict]:
    """
    Filters to comm blocks that have timestamps.

    Args:
        blocks: List of block dictionaries

    Returns:
        List of comm blocks with timestamps
    """
    return [b for b in blocks if b.get("type") == "comm" and b.get("timestamp")]


def has_block_type(blocks: list[dict], block_type: str) -> bool:
    """
    Checks if any block of given type exists.

    Args:
        blocks: List of block dictionaries
        block_type: Type to check for

    Returns:
        True if at least one block of type exists
    """
    return any(b.get("type") == block_type for b in blocks)


def count_by_type(blocks: list[dict], block_type: str) -> int:
    """
    Counts blocks of given type.

    Args:
        blocks: List of block dictionaries
        block_type: Type to count

    Returns:
        Number of blocks of specified type
    """
    return sum(1 for b in blocks if b.get("type") == block_type)


def get_first_of_type(blocks: list[dict], block_type: str) -> dict | None:
    """
    Gets first block of given type.

    Args:
        blocks: List of block dictionaries
        block_type: Type to find

    Returns:
        First block of type, or None if not found
    """
    for block in blocks:
        if block.get("type") == block_type:
            return block
    return None


def get_last_of_type(blocks: list[dict], block_type: str) -> dict | None:
    """
    Gets last block of given type.

    Args:
        blocks: List of block dictionaries
        block_type: Type to find

    Returns:
        Last block of type, or None if not found
    """
    for block in reversed(blocks):
        if block.get("type") == block_type:
            return block
    return None
