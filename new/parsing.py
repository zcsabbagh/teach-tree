"""Parsing utilities for extracting assessment data from LLM responses."""
import re
from typing import List, Dict, Any, Optional

from .utils import extract_json_from_text, parse_bool


def parse_estimates_from_json(
    data: Dict[str, Any],
    valid_topics: List[str],
) -> List[Dict[str, Any]]:
    """Parse knowledge estimates from a JSON structure.

    Returns list of dicts with: topic, known, confidence, reasoning
    """
    estimates = []

    # Handle both {"estimates": [...]} and direct array formats
    estimate_list = data.get("estimates", data if isinstance(data, list) else [])

    for e in estimate_list:
        topic = e.get("topic", e.get("name", ""))
        known = parse_bool(e.get("known", e.get("knows", False)))
        # Handle None values (when LLM returns null)
        raw_confidence = e.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else 0.5
        reasoning = e.get("reasoning", e.get("reason", "")) or ""

        if topic and topic in valid_topics:
            estimates.append({
                "topic": topic,
                "known": known,
                "confidence": confidence,
                "reasoning": reasoning,
            })

    return estimates


def regex_extract_topic_assessment(
    text: str,
    topic: str,
) -> Optional[Dict[str, Any]]:
    """Extract assessment for a single topic using regex patterns.

    Returns dict with: topic, known, confidence, reasoning
    Or None if no pattern matches.
    """
    text_lower = text.lower()
    topic_escaped = re.escape(topic.lower())

    known = None
    confidence = 0.5
    reasoning = "Extracted via pattern matching"

    # Pattern 1: "topic": true/false or topic: true/false
    pattern1 = rf'["\']?{topic_escaped}["\']?\s*[:\-]\s*(true|false|yes|no)'
    match1 = re.search(pattern1, text_lower, re.IGNORECASE)
    if match1:
        known = match1.group(1).lower() in ("true", "yes")
        confidence = 0.7

    # Pattern 2: topic ... knows/doesn't know
    if known is None:
        pattern2 = rf'{topic_escaped}[^.]*?(knows|doesn\'t know|does not know|understand|confused)'
        match2 = re.search(pattern2, text_lower, re.IGNORECASE)
        if match2:
            word = match2.group(1).lower()
            known = word in ("knows", "understand")
            confidence = 0.6

    # Pattern 3: ✓ or ✗ near topic name
    if known is None:
        pattern3 = rf'{topic_escaped}[^|\n]*?(✓|✗|correct|incorrect|wrong)'
        match3 = re.search(pattern3, text_lower, re.IGNORECASE)
        if match3:
            symbol = match3.group(1)
            known = symbol in ("✓", "correct")
            confidence = 0.6

    # Pattern 4: confidence percentage
    conf_pattern = rf'{topic_escaped}[^%]*?(\d+)\s*%'
    conf_match = re.search(conf_pattern, text_lower, re.IGNORECASE)
    if conf_match:
        confidence = int(conf_match.group(1)) / 100

    if known is not None:
        return {
            "topic": topic,
            "known": known,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    return None


def regex_extract_estimates(
    text: str,
    topics: List[str],
) -> List[Dict[str, Any]]:
    """Extract estimates for multiple topics using regex patterns."""
    estimates = []
    for topic in topics:
        estimate = regex_extract_topic_assessment(text, topic)
        if estimate:
            estimates.append(estimate)
    return estimates


def build_llm_extraction_prompt(
    raw_response: str,
    topics: List[str],
) -> str:
    """Build a prompt for LLM-based extraction of assessment data."""
    topics_json = ", ".join(f'"{t}"' for t in topics)

    return f"""Extract the knowledge assessment from this teacher's response.

TEACHER'S RESPONSE:
{raw_response}

TOPICS TO EXTRACT: {", ".join(topics)}

For EACH topic, determine:
1. known: Does the teacher think the student KNOWS this topic? (true/false)
   - Only true if there's clear evidence of correct understanding
   - false if: no evidence, errors, confusion, or wrong answers
2. confidence: How certain is the assessment? (0.0-1.0)
   - 0.5 = no evidence or uncertain
   - 0.8+ = clear evidence
3. reasoning: Brief explanation of the assessment

Respond with ONLY valid JSON, no other text:
{{"estimates": [
  {{"topic": "topic_name", "known": true/false, "confidence": 0.0-1.0, "reasoning": "explanation"}}
]}}

You MUST include ALL topics: [{topics_json}]"""


def parse_assessment_response(
    response_text: str,
    valid_topics: List[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Parse an assessment response, returning estimates and missing topics.

    Returns:
        (estimates, missing_topics) tuple
    """
    estimates = []

    # Try to extract JSON
    data = extract_json_from_text(response_text)
    if data:
        estimates = parse_estimates_from_json(data, valid_topics)

    # Check which topics are covered
    covered = {e["topic"] for e in estimates}
    missing = [t for t in valid_topics if t not in covered]

    return estimates, missing


def fill_missing_topics(
    estimates: List[Dict[str, Any]],
    all_topics: List[str],
    default_reasoning: str = "No evidence in conversation",
) -> List[Dict[str, Any]]:
    """Ensure all topics have an estimate, filling in defaults for missing ones."""
    covered = {e["topic"] for e in estimates}
    result = list(estimates)

    for topic in all_topics:
        if topic not in covered:
            result.append({
                "topic": topic,
                "known": False,
                "confidence": 0.5,
                "reasoning": default_reasoning,
            })

    return result
