"""
Query classification system for categorizing user queries by type.

Provides QueryType enum and QUERY_CLASSIFIERS with regex patterns for:
- UNSAFE: Harmful or illegal content (blocked before LLM)
- VAGUE: Intentionally broad questions or direct topic names
- META: Questions about assistant capabilities/identity
- DOCUMENT: Questions about knowledge base contents
- REGULAR: Normal Q&A queries
"""

import os
import re
from enum import Enum

import yaml


class QueryType(Enum):
    """Query classification types with priority-based handling."""

    UNSAFE = "unsafe"  # Harmful or illegal content (highest priority)
    VAGUE = "vague"  # Intentionally broad questions or direct topic names
    META = "meta"  # Questions about assistant capabilities/identity
    DOCUMENT = "document"  # Questions about knowledge base contents
    REGULAR = "regular"  # Normal Q&A queries (lowest priority)


def _extract_topics_from_constraint(constraint: str) -> list[str]:
    """Extract ALL topic keywords from constraint, handling nested parentheses correctly."""
    topics = []
    if "vague but legitimate questions" not in constraint:
        return topics

    # Extract the section about topics
    topic_section = re.search(
        r"I can answer questions about ([^.!?]+)", constraint, re.IGNORECASE
    )

    if not topic_section:
        return topics

    topic_text = topic_section.group(1)

    # Parse topics more carefully, respecting parenthetical content
    # Pattern: "topic (subtopic1, subtopic2), topic2 (subtopic3), ..."

    current_topic = ""
    in_parens = 0
    topics_found = []

    for char in topic_text:
        if char == "(":
            in_parens += 1
            current_topic += char
        elif char == ")":
            in_parens -= 1
            current_topic += char
        elif char == "," and in_parens == 0:
            # End of this topic, split it
            topic_with_subtopics = current_topic.strip()
            if topic_with_subtopics:
                topics_found.append(topic_with_subtopics)
            current_topic = ""
        else:
            current_topic += char

    # Don't forget the last topic
    if current_topic.strip():
        topics_found.append(current_topic.strip())

    # Now extract main topics and subtopics
    for item in topics_found:
        # Remove parenthetical content but keep the main topic
        main_topic = re.sub(r"\s*\([^)]*\)", "", item).strip()
        if main_topic and len(main_topic) > 2:
            # Split compound topics on "and"
            parts = [p.strip() for p in main_topic.split(" and ")]
            topics.extend([p for p in parts if p and len(p) > 2])

        # Extract subtopics from parentheses
        paren_content = re.findall(r"\(([^)]+)\)", item)
        for paren_text in paren_content:
            subtopics = [t.strip() for t in paren_text.split(",")]
            topics.extend([t for t in subtopics if t and len(t) > 2])

    return list(set(topics))  # Remove duplicates


def _load_topic_keywords():
    """Load topic keywords from prompt-config.yaml for VAGUE pattern matching."""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "prompt-config.yaml"
        )
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract topic keywords from the vague questions constraint
        topics = []
        for prompt_config in config.values():
            if "output_constraints" not in prompt_config:
                continue
            for constraint in prompt_config["output_constraints"]:
                extracted = _extract_topics_from_constraint(constraint)
                topics.extend(extracted)

        return list(set(topics))  # Remove duplicates
    except Exception:  # pylint: disable=broad-exception-caught
        # If loading fails, use default topics
        return [
            "artificial intelligence",
            "quantum",
            "consciousness",
            "cryptography",
            "contemporary art",
            "ancient civilizations",
            "psychology",
            "philosophy",
            "linguistics",
        ]


# Load topic keywords for dynamic VAGUE pattern
_TOPIC_KEYWORDS = _load_topic_keywords()

# Regex patterns for query classification (priority order: unsafe > vague > meta > document > regular)
QUERY_CLASSIFIERS = {
    "unsafe": {
        "pattern": re.compile(
            r"\b(illegal|crime|violence|harm|exploit|abuse|hate|"
            r"discriminat|drug|weapon|terrorist|attack|unethical)\b",
            re.IGNORECASE,
        ),
        "description": "Harmful or illegal content - blocked before LLM",
    },
    "vague": {
        "pattern": re.compile(
            # Action phrases that indicate broad topic requests
            r"(tell me about|explain|describe|what.*about|inform me about|brief|overview|summary|introduction to|"
            r"guide to|"
            # All dynamically loaded topics from config (no hardcoding needed!)
            + (
                "|".join(re.escape(kw) for kw in _TOPIC_KEYWORDS)
                if _TOPIC_KEYWORDS
                else ""
            )
            + r")",
            re.IGNORECASE,
        ),
        "description": "Intentionally broad topic requests or direct topic names - bypass similarity threshold",
    },
    "meta": {
        "pattern": re.compile(
            r"\b(who are you|what are you|tell me about yourself|introduce yourself|"
            r"capabilities|limitations|purpose|how are you built|how do you work"
            r"|what are your limitations|how can you help|what is your purpose)\b",
            re.IGNORECASE,
        ),
        "description": "Questions about assistant identity/capabilities",
    },
    "document": {
        "pattern": re.compile(
            r"\b(what topics|what do you know|what can you|what documents|"
            r"what information|what subjects|what's your knowledge base|"
            r"do you have access to|can you access)\b",
            re.IGNORECASE,
        ),
        "description": "Questions about knowledge base contents",
    },
}
