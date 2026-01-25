"""Memory manager for the RAG assistant.

This module provides MemoryManager which selects and initializes a
conversation memory strategy based on configuration. It supports multiple
memory implementations and falls back gracefully when optional dependencies
are not available.
"""

from config import MEMORY_STRATEGIES_FPATH, MEMORY_STRATEGY
from logger import logger
from simple_buffer_memory import SimpleBufferMemory
from sliding_window_memory import SlidingWindowMemory
from summary_memory import SummaryMemory

# Make PyYAML optional to avoid import errors in lightweight environments
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


class MemoryManager:
    """Manages conversation memory based on configured strategy."""

    def __init__(self, llm):
        """Initialize memory manager with the configured strategy."""
        self.strategy = MEMORY_STRATEGY
        self.config = self._load_memory_strategy_config()
        self.llm = llm
        self.memory = None
        self._initialize_memory()

    def _load_memory_strategy_config(self):
        """Load memory strategy configuration from YAML file.

        If PyYAML is not installed, return an empty configuration and log a
        warning so the application can continue running without memory.
        """
        if yaml is None:
            logger.warning("PyYAML not installed; memory strategies unavailable.")
            return {}

        try:
            with open(MEMORY_STRATEGIES_FPATH, "r", encoding="utf-8") as f:
                strategies = yaml.safe_load(f) or {}
            return strategies.get(self.strategy, {})
        except FileNotFoundError:
            logger.warning(
                f"Memory strategies config not found at {MEMORY_STRATEGIES_FPATH}"
            )
            return {}

    def _initialize_memory(self):
        """Initialize the appropriate memory strategy.

        Supports:
        - summarization_sliding_window: Sliding window with summarization
        - simple_buffer: Simple in-memory buffer
        - summary: LLM-based running summary
        - none: No memory

        Falls back gracefully if initialization fails.
        """
        if self.strategy == "summarization_sliding_window":
            self._initialize_sliding_window_memory()
        elif self.strategy == "simple_buffer":
            self._initialize_simple_buffer_memory()
        elif self.strategy == "summary":
            self._initialize_summary_memory()
        elif self.strategy == "none":
            self.memory = None
        else:
            logger.warning(
                f"Unknown memory strategy: {self.strategy}. No memory applied."
            )
            self.memory = None

    def _initialize_sliding_window_memory(self):
        """Initialize sliding window memory strategy.

        This is the primary memory implementation as it's the most reliable
        and doesn't depend on external langchain memory classes that may not
        be available or may have compatibility issues.
        """
        try:
            window_size = self.config.get("parameters", {}).get("window_size", 20)
            memory_key = self.config.get("parameters", {}).get(
                "memory_key", "chat_history"
            )
            self.memory = SlidingWindowMemory(
                llm=self.llm,
                window_size=window_size,
                memory_key=memory_key,
            )
            logger.info(
                f"SlidingWindowMemory initialized with strategy: {self.strategy}"
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                f"SlidingWindowMemory initialization failed: {e}. Falling back to no memory."
            )
            self.memory = None

    def _initialize_simple_buffer_memory(self):
        """Initialize simple buffer memory strategy.

        Stores conversation history in a simple in-memory buffer.
        No summarization or advanced features.
        """
        try:
            memory_key = self.config.get("parameters", {}).get(
                "memory_key", "chat_history"
            )
            max_messages = self.config.get("parameters", {}).get("max_messages", 50)
            self.memory = SimpleBufferMemory(
                memory_key=memory_key,
                max_messages=max_messages,
            )
            logger.info(
                f"SimpleBufferMemory initialized with max_messages={max_messages}"
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                f"SimpleBufferMemory initialization failed: {e}. Falling back to no memory."
            )
            self.memory = None

    def _initialize_summary_memory(self):
        """Initialize summary memory strategy.

        Maintains a running summary of the conversation using the LLM.
        """
        try:
            memory_key = self.config.get("parameters", {}).get(
                "memory_key", "chat_history"
            )
            summary_prompt = self.config.get("parameters", {}).get(
                "summary_prompt",
                "Summarize the conversation so far in a few sentences.",
            )
            self.memory = SummaryMemory(
                llm=self.llm,
                memory_key=memory_key,
                summary_prompt=summary_prompt,
            )
            logger.info("SummaryMemory initialized")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                f"SummaryMemory initialization failed: {e}. Falling back to no memory."
            )
            self.memory = None

    def add_message(self, input_text: str, output_text: str) -> None:
        """Add a message pair to memory."""
        if self.memory:
            try:
                self.memory.save_context({"input": input_text}, {"output": output_text})
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error saving to memory: {e}")

    def get_memory_variables(self) -> dict:
        """Get current memory variables for the chain."""
        if self.memory:
            try:
                return self.memory.load_memory_variables({})
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error loading memory variables: {e}")
                return {}
        return {}
