"""
Pytest configuration file for RAG Assistant tests.
Defines fixtures, plugins, and test behavior.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "test_data_dir": os.path.join(os.path.dirname(__file__), "fixtures"),
        "mock_llm_model": "test-model",
        "mock_strategy": "rag_enhanced_reasoning",
    }


@pytest.fixture
def mock_logger():
    """Provide mock logger for testing."""
    from unittest.mock import MagicMock
    return MagicMock()


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark all tests in test_hallucination_prevention.py as integration
        if "hallucination_prevention" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # Mark all other tests as unit
        else:
            item.add_marker(pytest.mark.unit)
