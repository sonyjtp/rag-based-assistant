"""
Unit tests for reasoning strategy functionality.
Tests strategy loading, application, and configuration.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.reasoning_strategy_loader import ReasoningStrategyLoader


class TestReasoningStrategyLoaderInitialization:
    """Test ReasoningStrategyLoader initialization."""

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_initialization_success(self, mock_config, mock_load_yaml):
        """Test successful initialization of ReasoningStrategyLoader."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "rag_enhanced_reasoning"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "rag_enhanced_reasoning": {
                    "name": "RAG-Enhanced Reasoning",
                    "enabled": True,
                    "prompt_instructions": ["Instruction 1"]
                }
            }
        }

        loader = ReasoningStrategyLoader()

        assert loader.active_strategy == "rag_enhanced_reasoning"
        assert loader.strategies is not None

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_custom_config_path(self, mock_config, mock_load_yaml):
        """Test initialization with custom config path."""
        mock_config.REASONING_STRATEGIES_FPATH = "/default/path.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "test_strategy": {"enabled": True}
            }
        }

        custom_path = "/custom/path.yaml"
        loader = ReasoningStrategyLoader(config_path=custom_path)

        # Should use provided custom path
        mock_load_yaml.assert_called_with(custom_path)


class TestReasoningStrategyRetrieval:
    """Test retrieving reasoning strategy information."""

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_get_active_strategy(self, mock_config, mock_load_yaml):
        """Test retrieving active strategy configuration."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "chain_of_thought"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "chain_of_thought": {
                    "name": "Chain-of-Thought",
                    "enabled": True,
                    "description": "Step by step reasoning"
                }
            }
        }

        loader = ReasoningStrategyLoader()
        strategy = loader.get_active_strategy()

        assert strategy["name"] == "Chain-of-Thought"
        assert strategy["enabled"] == True

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_get_strategy_name(self, mock_config, mock_load_yaml):
        """Test retrieving strategy name."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "rag_enhanced_reasoning"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "rag_enhanced_reasoning": {
                    "name": "RAG-Enhanced Reasoning"
                }
            }
        }

        loader = ReasoningStrategyLoader()
        name = loader.get_strategy_name()

        assert name == "RAG-Enhanced Reasoning"

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_get_strategy_description(self, mock_config, mock_load_yaml):
        """Test retrieving strategy description."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "test_strategy": {
                    "description": "This is a test strategy"
                }
            }
        }

        loader = ReasoningStrategyLoader()
        description = loader.get_strategy_description()

        assert description == "This is a test strategy"

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_get_strategy_instructions(self, mock_config, mock_load_yaml):
        """Test retrieving strategy instructions."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        instructions = [
            "First, analyze the question",
            "Then, retrieve relevant information",
            "Finally, synthesize the answer"
        ]

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "test_strategy": {
                    "prompt_instructions": instructions
                }
            }
        }

        loader = ReasoningStrategyLoader()
        retrieved_instructions = loader.get_strategy_instructions()

        assert retrieved_instructions == instructions
        assert len(retrieved_instructions) == 3

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_get_few_shot_examples(self, mock_config, mock_load_yaml):
        """Test retrieving few-shot examples if available."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        examples = [
            {
                "question": "Example Q1",
                "answer": "Example A1"
            },
            {
                "question": "Example Q2",
                "answer": "Example A2"
            }
        ]

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "test_strategy": {
                    "examples": examples
                }
            }
        }

        loader = ReasoningStrategyLoader()
        retrieved_examples = loader.get_few_shot_examples()

        assert len(retrieved_examples) == 2
        assert retrieved_examples[0]["question"] == "Example Q1"


class TestReasoningStrategyValidation:
    """Test strategy validation and checks."""

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_is_strategy_enabled(self, mock_config, mock_load_yaml):
        """Test checking if strategy is enabled."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "enabled_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "enabled_strategy": {
                    "enabled": True
                }
            }
        }

        loader = ReasoningStrategyLoader()
        assert loader.is_strategy_enabled() == True

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_is_strategy_disabled(self, mock_config, mock_load_yaml):
        """Test checking if strategy is disabled."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "disabled_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "disabled_strategy": {
                    "enabled": False
                }
            }
        }

        loader = ReasoningStrategyLoader()
        assert loader.is_strategy_enabled() == False

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_invalid_strategy_raises_error(self, mock_config, mock_load_yaml):
        """Test that requesting invalid strategy raises error."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "nonexistent_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "existing_strategy": {"enabled": True}
            }
        }

        loader = ReasoningStrategyLoader()

        with pytest.raises(ValueError):
            loader.get_active_strategy()


class TestReasoningStrategyAllEnabled:
    """Test querying all enabled strategies."""

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_get_all_enabled_strategies(self, mock_config, mock_load_yaml):
        """Test retrieving all enabled strategies."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "rag_enhanced_reasoning"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "rag_enhanced_reasoning": {"enabled": True},
                "chain_of_thought": {"enabled": True},
                "self_consistency": {"enabled": False},
                "tree_of_thought": {"enabled": False}
            }
        }

        loader = ReasoningStrategyLoader()
        enabled = loader.get_all_enabled_strategies()

        assert len(enabled) == 2
        assert "rag_enhanced_reasoning" in enabled
        assert "chain_of_thought" in enabled
        assert "self_consistency" not in enabled


class TestReasoningStrategyPromptBuilding:
    """Test building complete strategy prompts."""

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_build_strategy_prompt(self, mock_config, mock_load_yaml):
        """Test building complete prompt for strategy."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "test_strategy": {
                    "name": "Test Strategy",
                    "description": "A test reasoning strategy",
                    "prompt_instructions": ["Step 1", "Step 2"]
                }
            }
        }

        loader = ReasoningStrategyLoader()
        prompt = loader.build_strategy_prompt()

        assert "Test Strategy" in prompt
        assert "A test reasoning strategy" in prompt
        assert "Step 1" in prompt
        assert "Step 2" in prompt


class TestReasoningStrategyEdgeCases:
    """Test edge cases in reasoning strategy handling."""

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_empty_instructions(self, mock_config, mock_load_yaml):
        """Test strategy with no instructions."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "no_instructions"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "no_instructions": {
                    "prompt_instructions": []
                }
            }
        }

        loader = ReasoningStrategyLoader()
        instructions = loader.get_strategy_instructions()

        assert instructions == []

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_missing_optional_fields(self, mock_config, mock_load_yaml):
        """Test strategy with missing optional fields."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "minimal_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "minimal_strategy": {
                    "enabled": True
                }
            }
        }

        loader = ReasoningStrategyLoader()

        # Should handle gracefully
        assert loader.get_strategy_name() == "minimal_strategy"
        assert loader.get_strategy_description() == ""
        assert loader.get_strategy_instructions() == []

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_very_long_instructions(self, mock_config, mock_load_yaml):
        """Test strategy with very long instruction list."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "long_instructions"

        long_instructions = [f"Instruction {i}" for i in range(100)]

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "long_instructions": {
                    "prompt_instructions": long_instructions
                }
            }
        }

        loader = ReasoningStrategyLoader()
        instructions = loader.get_strategy_instructions()

        assert len(instructions) == 100


class TestReasoningStrategyExceptionHandling:
    """Test exception handling in reasoning strategy."""

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_config_load_error(self, mock_config, mock_load_yaml):
        """Test handling of configuration load errors."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"
        mock_load_yaml.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(FileNotFoundError):
            ReasoningStrategyLoader()

    @patch('src.reasoning_strategy_loader.load_yaml')
    @patch('src.reasoning_strategy_loader.config')
    def test_malformed_yaml(self, mock_config, mock_load_yaml):
        """Test handling of malformed YAML."""
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"
        mock_load_yaml.side_effect = Exception("Invalid YAML format")

        with pytest.raises(Exception):
            ReasoningStrategyLoader()
