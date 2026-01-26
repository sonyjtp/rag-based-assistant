#!/usr/bin/env python3
"""Test vague query auto-prefixing functionality."""

import sys

from src.rag_assistant import RAGAssistant

sys.path.insert(0, "src")


# pylint: disable=protected-access
def main():
    """Test the vague query auto-prefixing."""
    print("=" * 70)
    print("TESTING VAGUE QUERY AUTO-PREFIXING")
    print("=" * 70)

    assistant = RAGAssistant()

    test_queries = [
        # Vague queries (should be prefixed)
        "psychology",
        "human-machine interaction",
        "contemporary art",
        "quantum cryptography",
        "history",
        # Complete queries (should NOT be prefixed)
        "Tell me about psychology",
        "What is human-machine interaction?",
        "Explain contemporary art",
        "Why is quantum cryptography important?",
        "History of ancient civilizations",
    ]

    print("\nQuery classification with auto-prefixing:\n")
    for query in test_queries:
        is_vague = assistant.is_vague_incomplete(query)
        prefixed = f"Tell me about {query}" if is_vague else query
        qtype = assistant._classify_query(prefixed)
        status = "🔄 PREFIXED" if is_vague else "✅ COMPLETE"
        print(f"{status:15} {query:40} → {qtype.value}")


if __name__ == "__main__":
    main()
