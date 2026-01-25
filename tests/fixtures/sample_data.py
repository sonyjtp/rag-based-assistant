"""
Test fixtures and sample data for RAG Assistant tests.
"""

import pytest


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "Artificial intelligence is transforming how we process information. Machine learning algorithms learn from "
        "data to make predictions.",
        "Quantum computing represents a paradigm shift in computational power, using quantum bits (qubits) for "
        "processing.",
        "Ancient Egyptian civilization was one of the most advanced societies of its time, with impressive "
        "architectural  achievements.",
        "Contemporary art encompasses diverse movements and styles, reflecting modern societal values and "
        "technological innovation.",
        "Consciousness research explores the nature of human awareness and subjective experience through neuroscience "
        "and philosophy.",
    ]


@pytest.fixture
def sample_documents_empty():
    """Provide empty documents list for edge case testing."""
    return []


@pytest.fixture
def sample_documents_single():
    """Provide single document for edge case testing."""
    return ["Artificial intelligence is a branch of computer science."]


@pytest.fixture
def sample_queries_in_scope():
    """Provide sample queries that should be answerable from documents."""
    return [
        "What is artificial intelligence?",
        "How does quantum computing work?",
        "Tell me about ancient Egypt.",
        "What is contemporary art?",
        "What is consciousness research?",
    ]


@pytest.fixture
def sample_queries_out_of_scope():
    """Provide sample queries that are out of scope (not in documents)."""
    return [
        "What is the etymology of Pharaonic?",
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "What is the weather today?",
        "Tell me about deep sea creatures.",
    ]


@pytest.fixture
def sample_queries_ambiguous():
    """Provide ambiguous queries that could be interpreted multiple ways."""
    return [
        "Tell me more",
        "What else?",
        "Anything else?",
        "Continue",
        "Go on",
    ]


@pytest.fixture
def sample_queries_gibberish():
    """Provide gibberish/nonsensical queries."""
    return [
        "asdfgh",
        "???",
        "sdadsad",
        "xyzabc123!@#",
        "zzzzzzzzz",
    ]


@pytest.fixture
def sample_queries_special_cases():
    """Provide special case queries."""
    return {
        "greeting": "Hello! How are you?",
        "thanks": "Thank you for your help!",
        "goodbye": "Goodbye!",
        "vague": "What topics do you have?",
        "about_limitations": "What are your limitations?",
    }


@pytest.fixture
def sample_context():
    """Provide sample context retrieved from documents."""
    return """
    Artificial intelligence (AI) refers to the simulation of human intelligence by machines,
    particularly computer systems. These systems are designed to perform tasks that typically
    require human intelligence, such as learning from experience, recognizing patterns,
    understanding language, and making decisions.
    """


@pytest.fixture
def sample_context_empty():
    """Provide empty context for edge case testing."""
    return ""


@pytest.fixture
def search_results_with_docs():
    """Provide mock search results with documents."""
    return {
        "documents": [
            [
                "Artificial intelligence is a field of computer science.",
                "Machine learning is a subset of AI.",
            ],
            ["Quantum computing uses quantum mechanics principles."],
        ],
        "ids": [["doc1", "doc2"], ["doc3"]],
        "distances": [[0.1, 0.2], [0.15]],
    }


@pytest.fixture
def search_results_empty():
    """Provide mock search results with no documents."""
    return {
        "documents": [[], []],
        "ids": [[], []],
        "distances": [[], []],
    }


@pytest.fixture
def memory_messages():
    """Provide sample memory messages."""
    return [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence..."},
        {"role": "user", "content": "Tell me more"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    ]


@pytest.fixture
def mock_llm_response():
    """Provide mock LLM response."""
    return "Artificial intelligence is a field of computer science that aims to create intelligent machines."


@pytest.fixture
def mock_llm_response_out_of_scope():
    """Provide mock LLM response for out-of-scope queries."""
    return "I'm sorry, that information is not known to me."


@pytest.fixture
def mock_llm_response_unclear():
    """Provide mock LLM response for unclear queries."""
    return "Could you please ask a clear, meaningful question?"
