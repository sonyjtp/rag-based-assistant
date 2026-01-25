"""
Integration tests for app.py CLI interface.
Tests the main entry point with mocked user input.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.app import main


class TestAppMain:
    """Test the main CLI application."""

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        """Reset mocks after each test."""
        yield

    @pytest.fixture
    def mocked_assistant(self):
        """Fixture providing a mocked RAGAssistant instance."""
        mock_assistant_instance = MagicMock()
        mock_assistant_instance.invoke.return_value = "Response"
        yield mock_assistant_instance
        mock_assistant_instance.reset_mock()

    @pytest.fixture
    def app_mocks(self):
        """Fixture providing mocked app dependencies."""
        with patch("src.app.input") as mock_input, patch(
            "src.app.RAGAssistant"
        ) as mock_assistant, patch(
            "src.app.load_documents", return_value=["Doc 1"]
        ) as mock_load_docs:
            yield {
                "input": mock_input,
                "assistant": mock_assistant,
                "load_docs": mock_load_docs,
            }

    def test_main_loads_documents(self, app_mocks, mocked_assistant):
        """Test that main() loads documents on startup."""
        app_mocks["input"].side_effect = ["What is AI?", "quit"]
        app_mocks["assistant"].return_value = mocked_assistant
        main()
        app_mocks["load_docs"].assert_called_once()

    @pytest.mark.parametrize(
        "user_input,expected_invoke_calls",
        [
            (["What is AI?", "quit"], 1),
            (["What is AI?", "Tell me more", "quit"], 2),
        ],
    )
    def test_main_assistant_behavior(
        self, app_mocks, mocked_assistant, user_input, expected_invoke_calls
    ):
        """Test that main() initializes assistant and processes queries correctly."""
        app_mocks["input"].side_effect = user_input
        app_mocks["assistant"].return_value = mocked_assistant
        main()
        app_mocks["assistant"].assert_called_once()
        mocked_assistant.add_documents.assert_called_once()
        assert mocked_assistant.invoke.call_count == expected_invoke_calls

    def test_main_quit_immediately(self, app_mocks, mocked_assistant):
        """Test that main() handles quit command immediately."""
        app_mocks["input"].side_effect = ["quit"]
        app_mocks["assistant"].return_value = mocked_assistant
        main()
        mocked_assistant.invoke.assert_not_called()

    @pytest.mark.parametrize(
        "exception,target",
        [
            (FileNotFoundError("Documents not found"), "src.app.load_documents"),
            (ValueError("Invalid document format"), "src.app.load_documents"),
            (RuntimeError("Model loading failed"), "src.app.RAGAssistant"),
        ],
    )
    def test_main_handles_exceptions(self, exception, target):
        """Test that main() handles various exceptions gracefully."""
        with patch(target, side_effect=exception):
            main()
