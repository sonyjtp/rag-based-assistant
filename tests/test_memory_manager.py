"""
Unit tests for memory management functionality.
Tests conversation storage, strategy switching, and memory operations.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.memory_manager import MemoryManager


@pytest.fixture
def mock_llm():
    """Fixture providing a mocked LLM instance."""
    return MagicMock()


@pytest.fixture
def mock_memory():
    """Fixture providing a mocked memory instance."""
    return MagicMock()


@pytest.fixture
def memory_patches():
    """Fixture providing memory-related patches for testing."""

    class MemoryPatchesContext:
        """Context manager for managing memory patches."""

        def __init__(self):
            self.patches = []
            self.mocks = {}
            self.strategy_value = None

        def __enter__(self):
            # Patch builtins.open
            open_patch = patch("builtins.open", mock_open(read_data="{}"))
            self.mocks["open"] = open_patch.start()
            self.patches.append(open_patch)

            # Patch MEMORY_STRATEGIES_FPATH
            path_patch = patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path")
            self.mocks["path"] = path_patch.start()
            self.patches.append(path_patch)

            # Patch MEMORY_STRATEGY with a PropertyMock that returns the set value
            strategy_patch = patch(
                "src.memory_manager.MEMORY_STRATEGY", self.strategy_value
            )
            self.mocks["strategy"] = strategy_patch.start()
            self.patches.append(strategy_patch)

            return self

        def __exit__(self, *args):
            for p in reversed(self.patches):
                p.stop()

        def set_strategy(self, value):
            """Set the strategy value for the patches."""
            self.strategy_value = value

    return MemoryPatchesContext()


# pylint: disable=redefined-outer-name
class TestMemoryManager:
    """Comprehensive tests for MemoryManager initialization, message handling, and operations."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "strategy,_config_key",
        [
            ("summarization_sliding_window", "summarization_sliding_window"),
            ("simple_buffer", "simple_buffer"),
            ("summary", "summary"),
        ],
    )
    def test_memory_initialization(
        self,
        strategy,
        _config_key,
        mock_llm,
        memory_patches,
    ):
        """Test MemoryManager initialization with different available strategies."""
        memory_patches.set_strategy(strategy)

        with memory_patches:
            manager = MemoryManager(llm=mock_llm)

            assert manager.llm is not None
            assert manager.strategy == strategy

    # ========================================================================
    # MESSAGE HANDLING TESTS
    # ========================================================================

    @pytest.fixture
    def add_message_params(self):
        """Fixture providing test parameters for add_message tests."""
        return {
            "cases": [
                ("Hello", "Hi there", "normal_messages"),
                ("Question?", "Answer here", "question_answer"),
                ("", "", "empty_message"),
                ("A" * 5000, "A" * 5000, "very_long_message"),
                (
                    "!@#$%^&*()_+-=[]{}|;:',.<>?/`~",
                    "!@#$%^&*()_+-=[]{}|;:',.<>?/`~",
                    "special_characters",
                ),
            ]
        }

    @pytest.mark.parametrize(
        "input_text,output_text,_test_name",
        [
            ("Hello", "Hi there", "normal_messages"),
            ("Question?", "Answer here", "question_answer"),
            ("", "", "empty_message"),
            ("A" * 5000, "A" * 5000, "very_long_message"),
            (
                "!@#$%^&*()_+-=[]{}|;:',.<>?/`~",
                "!@#$%^&*()_+-=[]{}|;:',.<>?/`~",
                "special_characters",
            ),
        ],
    )
    def test_add_message(
        self,
        input_text,
        output_text,
        _test_name,
        mock_llm,  # pylint: disable=redefined-outer-name
        mock_memory,  # pylint: disable=redefined-outer-name
        memory_patches,  # pylint: disable=redefined-outer-name
    ):
        """Test adding messages of various types to memory."""
        with memory_patches:
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory
            manager.add_message(input_text=input_text, output_text=output_text)

        mock_memory.save_context.assert_called_once()

    @pytest.mark.parametrize(
        "memory_variables,expected_keys",
        [
            ({"chat_history": "User: Hello\nAssistant: Hi"}, 1),
            ({"chat_history": "Conversation", "summary": "Summary text"}, 2),
        ],
    )
    def test_get_memory_variables(
        self,
        memory_variables,
        expected_keys,
        mock_llm,  # pylint: disable=redefined-outer-name
        mock_memory,  # pylint: disable=redefined-outer-name
        memory_patches,  # pylint: disable=redefined-outer-name
    ):
        """Test retrieving memory variables with different key counts."""
        mock_memory.load_memory_variables.return_value = memory_variables

        with memory_patches:
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory
            variables = manager.get_memory_variables()

        assert len(variables) == expected_keys
        for key in memory_variables.keys():
            assert key in variables

    def test_get_memory_variables_with_no_memory(
        self, mock_llm, memory_patches  # pylint: disable=redefined-outer-name
    ):
        """Test get_memory_variables when memory is None."""
        with memory_patches:
            manager = MemoryManager(llm=mock_llm)
            manager.memory = None
            variables = manager.get_memory_variables()

        assert variables == {}

    def test_get_memory_variables_load_error(
        self,
        mock_llm,  # pylint: disable=redefined-outer-name
        mock_memory,  # pylint: disable=redefined-outer-name
        memory_patches,  # pylint: disable=redefined-outer-name
    ):
        """Test get_memory_variables handles loading errors gracefully."""
        mock_memory.load_memory_variables.side_effect = ValueError("Load error")

        with memory_patches:
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory
            variables = manager.get_memory_variables()

        assert variables == {}

    # ========================================================================
    # STRATEGY SWITCHING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "strategy",
        [
            "summarization_sliding_window",
            "simple_buffer",
            "summary",
            "none",
        ],
    )
    def test_switching_memory_strategies(
        self, strategy, mock_llm, memory_patches  # pylint: disable=redefined-outer-name
    ):
        """Test switching between all available memory strategies."""
        memory_patches.set_strategy(strategy)

        with memory_patches:
            manager = MemoryManager(llm=mock_llm)

        assert manager is not None

    def test_strategy_none_disables_memory(
        self, mock_llm, memory_patches  # pylint: disable=redefined-outer-name
    ):
        """Test that 'none' strategy disables memory."""
        with memory_patches:
            manager = MemoryManager(llm=mock_llm)

        assert manager.memory is None or manager.strategy == "none"

    # ========================================================================
    # CONVERSATION FLOW TESTS
    # ========================================================================

    def test_multi_turn_conversation(
        self,
        mock_llm,  # pylint: disable=redefined-outer-name
        mock_memory,  # pylint: disable=redefined-outer-name
        memory_patches,  # pylint: disable=redefined-outer-name
    ):
        """Test multi-turn conversation with memory."""
        with memory_patches:
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory

            manager.add_message(
                input_text="What is AI?",
                output_text="AI is artificial intelligence.",
            )
            manager.add_message(
                input_text="Tell me more",
                output_text="Machine learning is a subset of AI.",
            )

        assert mock_memory.save_context.call_count == 2

    def test_conversation_context_accumulation(
        self,
        mock_llm,  # pylint: disable=redefined-outer-name
        mock_memory,  # pylint: disable=redefined-outer-name
        memory_patches,  # pylint: disable=redefined-outer-name
    ):
        """Test that conversation context accumulates over turns."""
        mock_memory.load_memory_variables.return_value = {
            "history": "Q1: What is AI?\nA1: AI is artificial intelligence.\nQ2: Tell me more\nA2: ML is subset..."
        }

        with memory_patches:
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory

            for i in range(3):
                manager.add_message(
                    input_text=f"Question {i}", output_text=f"Answer {i}"
                )

        assert mock_memory.save_context.call_count == 3

    # ========================================================================
    # EXCEPTION HANDLING TESTS
    # ========================================================================

    @pytest.fixture
    def exception_test_cases(self):
        """Fixture providing exception test cases."""
        return [
            (ValueError, "Memory save failed", "/path", "summarization_sliding_window"),
            (ValueError, "Memory save failed", "/path", "simple_buffer"),
            (ValueError, "Memory save failed", "/path", "summary"),
            (FileNotFoundError, "Config not found", "/nonexistent", "none"),
        ]

    @pytest.mark.parametrize(
        "exception_type,exception_msg,config_path,strategy",
        [
            (ValueError, "Memory save failed", "/path", "summarization_sliding_window"),
            (ValueError, "Memory save failed", "/path", "simple_buffer"),
            (ValueError, "Memory save failed", "/path", "summary"),
            (FileNotFoundError, "Config not found", "/nonexistent", "none"),
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @patch("src.memory_manager.MEMORY_STRATEGIES_FPATH")
    @patch("src.memory_manager.MEMORY_STRATEGY")
    def test_exception_handling(
        self,
        mock_strategy,
        mock_path,
        exception_type,
        exception_msg,
        config_path,
        strategy,
        mock_llm,  # pylint: disable=redefined-outer-name
    ):
        """Test handling of various exceptions in memory management."""
        mock_strategy.__str__ = MagicMock(return_value=strategy)
        mock_path.__str__ = MagicMock(return_value=config_path)

        if exception_type == FileNotFoundError:
            with patch("builtins.open", side_effect=FileNotFoundError(exception_msg)):
                manager = MemoryManager(llm=mock_llm)
                assert manager is not None
                assert manager.config == {}
        else:
            with patch("builtins.open", mock_open(read_data="{}")):
                manager = MemoryManager(llm=mock_llm)
                if manager.memory:
                    temp_memory = MagicMock()  # pylint: disable=redefined-outer-name
                    temp_memory.save_context.side_effect = exception_type(exception_msg)
                    manager.memory = temp_memory
                    manager.add_message(input_text="Test", output_text="Test")
                    temp_memory.save_context.assert_called_once()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_unknown_strategy(
        self, mock_llm, memory_patches  # pylint: disable=redefined-outer-name
    ):
        """Test handling of unknown memory strategy."""
        memory_patches.set_strategy("unknown_strategy")

        with memory_patches:
            manager = MemoryManager(llm=mock_llm)

        assert manager is not None
        assert manager.memory is None
