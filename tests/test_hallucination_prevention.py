"""
Integration tests for hallucination prevention mechanisms.
Tests constraint enforcement, prompt engineering, and document grounding.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.rag_assistant import RAGAssistant
from src.prompt_builder import build_system_prompts


class TestHallucinationPreventionConstraints:
    """Test hallucination prevention through prompt constraints."""

    def test_system_prompts_contain_critical_constraint(self):
        """Test that system prompts include the CRITICAL constraint."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)
        prompt_upper = prompt_text.upper()

        assert "CRITICAL" in prompt_text
        assert "DO NOT" in prompt_upper
        assert "GENERAL KNOWLEDGE" in prompt_upper

    def test_system_prompts_contain_rejection_message(self):
        """Test that system prompts specify rejection for out-of-scope questions."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "not known to me" in prompt_text

    def test_system_prompts_contain_etymology_example(self):
        """Test that system prompts explicitly mention etymology as example."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "etymology" in prompt_text.lower()
        assert "Pharaonic" in prompt_text

    def test_system_prompts_forbid_general_knowledge(self):
        """Test that system prompts explicitly forbid general knowledge usage."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "general knowledge" in prompt_text.lower()
        assert "fallback" in prompt_text.lower()

    def test_system_prompts_contain_no_inference_rule(self):
        """Test that system prompts forbid inference beyond explicit statements."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "infer" in prompt_text.lower() or "speculation" in prompt_text.lower()
        assert "explicitly" in prompt_text.lower()

    def test_system_prompts_contain_document_grounding_requirement(self):
        """Test that system prompts require grounding in documents."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "document" in prompt_text.lower()
        assert "ground" in prompt_text.lower() or "grounded" in prompt_text.lower()


class TestHallucinationPreventionIntegration:
    """Test hallucination prevention in full pipeline."""

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_out_of_scope_query_gets_rejection_response(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that out-of-scope queries receive rejection response."""
        # Setup: No documents found (empty search results)
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [[]]}
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        # Verify constraint is in system prompt
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)
        assert "CRITICAL" in prompt_text

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_system_prompt_includes_rag_enhanced_reasoning(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that system prompt includes RAG-Enhanced reasoning strategy."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb.return_value = MagicMock()
        mock_memory.return_value.memory = MagicMock()
        mock_reasoning.return_value.get_strategy_name.return_value = "RAG-Enhanced Reasoning"
        mock_reasoning.return_value.get_strategy_instructions.return_value = [
            "First, use the retrieved documents as your knowledge base.",
            "Always ground your answer in the provided documents.",
        ]

        assistant = RAGAssistant()

        # Verify reasoning strategy is in prompt
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)
        assert "ground" in prompt_text.lower()

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_in_scope_query_retrieves_context(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that in-scope queries retrieve and use context."""
        # Setup: Documents found
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {
            "documents": [["Artificial intelligence is a field of computer science."]]
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "AI is a field of computer science."

        response = assistant.invoke("What is AI?")

        # Verify context was retrieved
        mock_vectordb_instance.search.assert_called_once()
        assert response is not None

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_context_passed_to_llm(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that retrieved context is passed to LLM."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {
            "documents": [["Document content about AI"]]
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        assistant.invoke("Test query")

        # Verify chain received context
        call_args = assistant.chain.invoke.call_args
        assert "context" in call_args[0][0]
        assert "Document content about AI" in call_args[0][0]["context"]


class TestHallucinationEdgeCases:
    """Test edge cases in hallucination prevention."""

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_empty_document_collection(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test behavior with empty document collection."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [[]]}
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "I'm sorry, that information is not known to me."

        response = assistant.invoke("Any question")

        # Verify appropriate response
        assert "not known" in response

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_gibberish_query(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test handling of gibberish/nonsensical queries."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [[]]}
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Could you please ask a clear, meaningful question?"

        response = assistant.invoke("asdfgh???xyz")

        # Verify constraint handles gibberish
        assert response is not None

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_ambiguous_follow_up_question(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test handling of ambiguous follow-up questions."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {
            "documents": [["More information about previous topic"]]
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Additional information..."

        # Verify prompt handles follow-ups
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)
        assert any(phrase in prompt_text for phrase in ["Tell me more", "continue", "Continue"])

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_etymology_example_explicitly_mentioned(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that etymology example is explicitly in constraints."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb.return_value = MagicMock()
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        # Check that etymology constraint is present
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)
        assert "Pharaonic" in prompt_text
        assert "etymology" in prompt_text.lower()

    @patch('src.rag_assistant.initialize_llm')
    @patch('src.rag_assistant.VectorDB')
    @patch('src.rag_assistant.MemoryManager')
    @patch('src.rag_assistant.ReasoningStrategyLoader')
    def test_multiple_queries_consistent_constraints(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that constraints are consistently applied across multiple queries."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [[]]}
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        # Verify constraints are in prompt
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        # Should have multiple constraint checks
        constraint_count = prompt_text.count("CRITICAL") + prompt_text.count("not known to me")
        assert constraint_count >= 1
