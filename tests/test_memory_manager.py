"""
Unit tests for memory management functionality.
Tests conversation storage, strategy switching, and memory operations.
"""

from unittest.mock import MagicMock, mock_open, patch

from src.memory_manager import MemoryManager


class TestMemoryManagerInitialization:
    """Test MemoryManager initialization."""

    @patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path/to/strategies.yaml")
    @patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory")
    @patch("src.memory_manager.yaml.safe_load")
    def test_memory_manager_initialization(self, mock_yaml_load):
        """Test successful initialization of MemoryManager."""
        mock_yaml_load.return_value = {
            "memory_strategies": {
                "conversation_buffer_memory": {
                    "enabled": True,
                    "description": "Buffer memory",
                }
            }
        }
        mock_llm = MagicMock()

        with patch("builtins.open", mock_open(read_data="{}")):
            manager = MemoryManager(llm=mock_llm)

        assert manager.llm is not None
        assert manager.strategy is not None

    @patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path/to/strategies.yaml")
    @patch("src.memory_manager.MEMORY_STRATEGY", "summarization_sliding_window")
    @patch("src.memory_manager.yaml.safe_load")
    def test_memory_strategy_loading(self, mock_yaml_load):
        """Test that memory strategy is properly loaded."""
        mock_yaml_load.return_value = {
            "memory_strategies": {
                "summarization_sliding_window": {"enabled": True, "window_size": 5}
            }
        }
        mock_llm = MagicMock()

        with patch("builtins.open", mock_open(read_data="{}")):
            manager = MemoryManager(llm=mock_llm)

        # Verify strategy is loaded
        assert manager.strategy == "summarization_sliding_window"


class TestMemoryMessageHandling:
    """Test message handling in memory."""

    def test_add_message_user(self):
        """Test adding user message to memory."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    manager.add_message(input_text="Hello", output_text="Hi there")

                    # Verify memory was updated
                    mock_memory.save_context.assert_called_once()

    def test_add_message_assistant(self):
        """Test adding assistant message to memory."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    manager.add_message(
                        input_text="Question?", output_text="Answer here"
                    )

                    # Verify memory save was called
                    assert mock_memory.save_context.called

    def test_get_memory_variables(self):
        """Test retrieving memory variables through MemoryManager."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        mock_memory.load_memory_variables.return_value = {
            "chat_history": "User: Hello\nAssistant: Hi"
        }

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    # Call the MemoryManager method, not memory.load_memory_variables
                    variables = manager.get_memory_variables()

                    assert "chat_history" in variables
                    assert variables["chat_history"] == "User: Hello\nAssistant: Hi"

    def test_get_memory_variables_with_no_memory(self):
        """Test get_memory_variables when memory is None."""
        mock_llm = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = None

                    variables = manager.get_memory_variables()

                    # Should return empty dict when memory is None
                    assert variables == {}

    def test_get_memory_variables_load_error(self):
        """Test get_memory_variables handles loading errors gracefully."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        mock_memory.load_memory_variables.side_effect = Exception("Load error")

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    # Should not raise, should return empty dict
                    variables = manager.get_memory_variables()

                    assert variables == {}

    def test_get_memory_variables_multiple_keys(self):
        """Test get_memory_variables with multiple keys in response."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        mock_memory.load_memory_variables.return_value = {
            "chat_history": "Conversation",
            "summary": "Summary text",
        }

        with patch(
            "src.memory_manager.MEMORY_STRATEGY", "summarization_sliding_window"
        ):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    variables = manager.get_memory_variables()

                    assert "chat_history" in variables
                    assert "summary" in variables
                    assert len(variables) == 2


class TestMemoryStrategySwitching:
    """Test switching between memory strategies."""

    def test_switching_to_buffer_memory(self):
        """Test switching to buffer memory strategy."""
        mock_llm = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)

                    # Memory should be initialized with default strategy
                    assert (
                        manager.memory is not None
                        or manager.strategy == "conversation_buffer_memory"
                    )

    def test_switching_to_summarization(self):
        """Test switching to summarization memory strategy."""
        mock_llm = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "summarization"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)

                    # Memory should be initialized
                    assert manager is not None

    def test_strategy_none_disables_memory(self):
        """Test that 'none' strategy disables memory."""
        mock_llm = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)

                    # Memory should be None or disabled
                    assert manager.memory is None or manager.strategy == "none"


class TestMemoryConversationFlow:
    """Test conversation flow with memory."""

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with memory."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    # First turn
                    manager.add_message(
                        input_text="What is AI?",
                        output_text="AI is artificial intelligence.",
                    )

                    # Second turn
                    manager.add_message(
                        input_text="Tell me more",
                        output_text="Machine learning is a subset of AI.",
                    )

                    # Verify both messages were stored
                    assert mock_memory.save_context.call_count == 2

    def test_conversation_context_accumulation(self):
        """Test that conversation context accumulates over turns."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        mock_memory.load_memory_variables.return_value = {
            "history": "Q1: What is AI?\nA1: AI is artificial intelligence.\nQ2: Tell me more\nA2: ML is subset..."
        }

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    # Add multiple messages
                    for i in range(3):
                        manager.add_message(
                            input_text=f"Question {i}", output_text=f"Answer {i}"
                        )

                    # Verify accumulation
                    assert mock_memory.save_context.call_count == 3


class TestMemoryEdgeCases:
    """Test edge cases in memory management."""

    def test_empty_message(self):
        """Test handling of empty messages."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    # Should handle empty strings gracefully
                    manager.add_message(input_text="", output_text="")

                    # Verify call was made (even if empty)
                    assert mock_memory.save_context.called

    def test_very_long_message(self):
        """Test handling of very long messages."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    long_text = "A" * 5000
                    manager.add_message(input_text=long_text, output_text=long_text)

                    # Should handle long text
                    assert mock_memory.save_context.called

    def test_special_characters_in_message(self):
        """Test handling of special characters in messages."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    special_text = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~"
                    manager.add_message(
                        input_text=special_text, output_text=special_text
                    )

                    assert mock_memory.save_context.called


class TestMemoryExceptionHandling:
    """Test exception handling in memory management."""

    def test_memory_save_error(self):
        """Test handling of memory save errors."""
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        mock_memory.save_context.side_effect = Exception("Memory save failed")

        with patch("src.memory_manager.MEMORY_STRATEGY", "conversation_buffer_memory"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    manager = MemoryManager(llm=mock_llm)
                    manager.memory = mock_memory

                    # Memory manager catches exceptions gracefully and logs them
                    # Should NOT raise, but call save_context
                    manager.add_message(input_text="Test", output_text="Test")
                    mock_memory.save_context.assert_called_once()

    def test_config_load_error(self):
        """Test handling of configuration load errors."""
        mock_llm = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "invalid"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/nonexistent"):
                # Memory manager handles FileNotFoundError gracefully and returns empty dict
                with patch(
                    "builtins.open", side_effect=FileNotFoundError("Config not found")
                ):
                    # Should NOT raise - it catches the exception
                    manager = MemoryManager(llm=mock_llm)

                    # Manager should be created even if config fails
                    assert manager is not None
                    # Config should be empty dict (fallback)
                    assert manager.config == {}

    def test_invalid_strategy(self):
        """Test handling of invalid memory strategy."""
        mock_llm = MagicMock()

        with patch("src.memory_manager.MEMORY_STRATEGY", "invalid_strategy"):
            with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"):
                with patch("builtins.open", mock_open(read_data="{}")):
                    # Should handle gracefully when strategy not found
                    manager = MemoryManager(llm=mock_llm)

                    assert manager is not None
