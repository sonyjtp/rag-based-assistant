"""
Integration tests for app.py CLI interface.
Tests the main entry point with mocked user input.
"""

from unittest.mock import MagicMock, patch

from src.app import main


class TestAppMain:
    """Test the main CLI application."""

    @patch("src.app.input", side_effect=["What is AI?", "quit"])
    @patch("src.app.RAGAssistant")
    @patch("src.app.load_documents")
    def test_main_loads_documents(self, mock_load_docs, mock_assistant):
        """Test that main() loads documents on startup."""
        mock_load_docs.return_value = ["Doc 1", "Doc 2", "Doc 3"]
        mock_assistant_instance = MagicMock()
        mock_assistant.return_value = mock_assistant_instance

        main()

        # Verify documents were loaded
        mock_load_docs.assert_called_once()

    @patch("src.app.input", side_effect=["What is AI?", "quit"])
    @patch("src.app.RAGAssistant")
    @patch("src.app.load_documents")
    def test_main_initializes_assistant(self, mock_load_docs, mock_assistant):
        """Test that main() initializes the RAG assistant."""
        mock_load_docs.return_value = ["Doc 1"]
        mock_assistant_instance = MagicMock()
        mock_assistant.return_value = mock_assistant_instance

        main()

        # Verify assistant was created and documents were added
        mock_assistant.assert_called_once()
        mock_assistant_instance.add_documents.assert_called_once()

    @patch("src.app.input", side_effect=["What is AI?", "Tell me more", "quit"])
    @patch("src.app.RAGAssistant")
    @patch("src.app.load_documents")
    def test_main_processes_multiple_queries(self, mock_load_docs, mock_assistant):
        """Test that main() handles multiple user queries."""
        mock_load_docs.return_value = ["Doc 1"]
        mock_assistant_instance = MagicMock()
        mock_assistant_instance.invoke.return_value = "Response"
        mock_assistant.return_value = mock_assistant_instance

        main()

        # Verify invoke was called for each question (2 times: "What is AI?" and "Tell me more")
        assert mock_assistant_instance.invoke.call_count == 2

    @patch("src.app.input", side_effect=["quit"])
    @patch("src.app.RAGAssistant")
    @patch("src.app.load_documents")
    def test_main_quit_immediately(self, mock_load_docs, mock_assistant):
        """Test that main() handles quit command immediately."""
        mock_load_docs.return_value = ["Doc 1"]
        mock_assistant_instance = MagicMock()
        mock_assistant.return_value = mock_assistant_instance

        main()

        # Verify invoke was NOT called if user quits
        mock_assistant_instance.invoke.assert_not_called()

    @patch("src.app.load_documents", side_effect=Exception("File error"))
    def test_main_error_handling(self, mock_load_docs):
        """Test that main() handles errors gracefully."""
        # Should not raise - error should be caught and logged
        main()

        # Verify load_documents was attempted
        mock_load_docs.assert_called_once()
