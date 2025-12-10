"""General utility functions."""
import json
import re
from typing import Any, Dict, List, Optional


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text that may contain markdown or other content."""
    text = text.strip()

    # Remove markdown code blocks
    if "```" in text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()

    # Try to find JSON object in the text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_bool(value: Any) -> bool:
    """Parse a value to boolean, handling string representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value)


def extract_number_from_text(text: str, max_value: int) -> Optional[int]:
    """Extract a number from text, typically for selecting from a list.

    Returns 0-indexed value, or None if no valid number found.
    """
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        # Take the last number mentioned (usually the final answer)
        idx = int(numbers[-1]) - 1  # Convert to 0-indexed
        if 0 <= idx < max_value:
            return idx
    return None


def find_items_in_set(
    items: List[str],
    valid_set: List[str],
) -> tuple[List[str], List[str]]:
    """Check which items are in valid_set.

    Returns (found, missing) tuple.
    """
    item_set = set(items)
    found = [v for v in valid_set if v in item_set]
    missing = [v for v in valid_set if v not in item_set]
    return found, missing
