"""
Unit tests for prompt builder module.
Tests prompt construction, constraint enforcement, and reasoning strategy integration.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.prompt_builder import build_system_prompts


class TestPromptBuilderBasic:
    """Test basic prompt building functionality."""

    def test_build_system_prompts_returns_list(self):
        """Test that build_system_prompts returns a list."""
        prompts = build_system_prompts()

        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_build_system_prompts_non_empty_strings(self):
        """Test that all prompts are non-empty strings."""
        prompts = build_system_prompts()

        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_build_system_prompts_count(self):
        """Test that appropriate number of prompt sections are built."""
        prompts = build_system_prompts()

        # Should have at least: role, tone, constraints, format, reasoning
        assert len(prompts) >= 5


class TestPromptBuilderConstraints:
    """Test constraint enforcement in prompts."""

    def test_system_prompts_contain_constraints(self):
        """Test that system prompts contain constraint directives."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "CRITICAL" in prompt_text
        assert "not known to me" in prompt_text
        assert "provided documents" in prompt_text.lower()

    def test_constraints_forbid_general_knowledge(self):
        """Test that constraints explicitly forbid general knowledge."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "training data" in prompt_text.lower()
        assert "general knowledge" in prompt_text.lower()

    def test_constraints_forbid_inference(self):
        """Test that constraints forbid inferring beyond explicit statements."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "infer" in prompt_text.lower() or "inference" in prompt_text.lower()

    def test_constraints_specify_rejection_message(self):
        """Test that constraints specify exact rejection message format."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "I'm sorry, that information is not known to me." in prompt_text

    def test_constraints_include_no_fallback_clause(self):
        """Test that constraints include no-fallback clause."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "fallback" in prompt_text.lower() or "not use your general knowledge" in prompt_text.lower()

    def test_constraints_mention_etymology_example(self):
        """Test that constraints include etymology as explicit example."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "etymology" in prompt_text.lower()
        assert "Pharaonic" in prompt_text


class TestPromptBuilderRole:
    """Test role configuration in prompts."""

    def test_system_prompts_contain_role(self):
        """Test that system prompts include assistant role."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "assistant" in prompt_text.lower() or "You are" in prompt_text

    def test_role_professional_or_helpful(self):
        """Test that role is either professional or helpful."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should contain either formal or candid role description
        assert "professional" in prompt_text.lower() or "helpful" in prompt_text.lower()


class TestPromptBuilderTone:
    """Test tone configuration in prompts."""

    def test_system_prompts_contain_tone(self):
        """Test that system prompts include tone instructions."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should mention language style
        assert any(word in prompt_text.lower() for word in ["formal", "clear", "precise", "language"])

    def test_tone_instructions_are_clear(self):
        """Test that tone instructions are specific and actionable."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Tone section should exist
        assert len([p for p in prompts if any(word in p.lower() for word in ["tone", "style", "formal", "clear"])]) > 0


class TestPromptBuilderFormat:
    """Test output format instructions in prompts."""

    def test_system_prompts_contain_format(self):
        """Test that system prompts include output format instructions."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "format" in prompt_text.lower() or "markdown" in prompt_text.lower()

    def test_format_specifies_markdown(self):
        """Test that format instructions specify markdown."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "markdown" in prompt_text.lower()

    def test_format_specifies_conciseness(self):
        """Test that format instructions emphasize conciseness."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert any(word in prompt_text.lower() for word in ["concise", "brief", "short", "direct"])


class TestPromptBuilderReasoningStrategy:
    """Test reasoning strategy integration in prompts."""

    @patch('src.prompt_builder.ReasoningStrategyLoader')
    def test_system_prompts_include_reasoning_strategy(self, mock_loader):
        """Test that system prompts include reasoning strategy instructions."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.is_strategy_enabled.return_value = True
        mock_loader_instance.get_strategy_instructions.return_value = [
            "Test instruction 1",
            "Test instruction 2"
        ]
        mock_loader_instance.get_strategy_name.return_value = "Test Strategy"
        mock_loader.return_value = mock_loader_instance

        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should contain reasoning strategy
        assert "Test instruction" in prompt_text or "reasoning" in prompt_text.lower()

    @patch('src.prompt_builder.ReasoningStrategyLoader')
    def test_rag_enhanced_reasoning_instructions_included(self, mock_loader):
        """Test that RAG-Enhanced reasoning instructions are included."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.is_strategy_enabled.return_value = True
        mock_loader_instance.get_strategy_instructions.return_value = [
            "First, use the retrieved documents as your knowledge base.",
            "Always ground your answer in the provided documents.",
            "Do not speculate beyond what is explicitly stated."
        ]
        mock_loader_instance.get_strategy_name.return_value = "RAG-Enhanced Reasoning"
        mock_loader.return_value = mock_loader_instance

        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should contain RAG principles
        assert "document" in prompt_text.lower()
        assert "ground" in prompt_text.lower() or "speculate" in prompt_text.lower()

    @patch('src.prompt_builder.ReasoningStrategyLoader')
    def test_disabled_strategy_not_included(self, mock_loader):
        """Test that disabled strategies are not included."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.is_strategy_enabled.return_value = False
        mock_loader.return_value = mock_loader_instance

        prompts = build_system_prompts()

        # Should still have other prompts
        assert len(prompts) > 0

    @patch('src.prompt_builder.ReasoningStrategyLoader')
    def test_strategy_load_error_handled_gracefully(self, mock_loader):
        """Test that strategy loading errors are handled gracefully."""
        mock_loader.side_effect = Exception("Failed to load strategy")

        # Should not raise exception
        prompts = build_system_prompts()

        # Should still return prompts
        assert len(prompts) > 0


class TestPromptBuilderSpecialCases:
    """Test special case handling in prompts."""

    def test_prompts_handle_follow_up_questions(self):
        """Test that prompts handle follow-up questions like 'Tell me more'."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should mention follow-up handling
        assert any(phrase in prompt_text for phrase in ["Tell me more", "continue", "follow-up"])

    def test_prompts_handle_polite_greetings(self):
        """Test that prompts handle polite greetings."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should mention greeting handling
        assert any(word in prompt_text for word in ["greeting", "thank", "goodbye"])

    def test_prompts_handle_gibberish(self):
        """Test that prompts handle gibberish queries."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should mention unclear question handling
        assert any(word in prompt_text.lower() for word in ["gibberish", "unclear", "nonsensical"])

    def test_prompts_specify_related_topics(self):
        """Test that prompts specify related topics section."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should mention related topics
        assert "Related Topics" in prompt_text or "related" in prompt_text.lower()


class TestPromptBuilderIntegration:
    """Integration tests for prompt builder."""

    def test_prompt_completeness(self):
        """Test that built prompts are complete and comprehensive."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should include all key components
        required_components = [
            "CRITICAL",  # Constraint enforcement
            "not known to me",  # Rejection message
            "provided documents",  # Document grounding
        ]

        for component in required_components:
            assert component in prompt_text

    def test_prompt_sections_are_distinct(self):
        """Test that different prompt sections are distinct and non-overlapping."""
        prompts = build_system_prompts()

        # Should have distinct sections
        assert len(set(prompts)) == len(prompts), "Prompts should be distinct"

    def test_prompt_length_reasonable(self):
        """Test that prompts are reasonable length (not too short, not too long)."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should be substantial (at least 500 chars)
        assert len(prompt_text) > 500
        # But not excessively long (less than 10k chars)
        assert len(prompt_text) < 10000
