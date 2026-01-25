# RAG Assistant Testing Guide

Comprehensive test suite for the RAG-based AI Assistant, covering unit tests, integration tests, and edge cases.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                          # Pytest configuration and shared fixtures
├── pytest.ini                           # Pytest settings
├── fixtures/
│   ├── __init__.py
│   └── sample_data.py                   # Test fixtures and sample data
├── test_rag_assistant.py                # RAG Assistant unit tests
├── test_prompt_builder.py               # Prompt builder tests
├── test_hallucination_prevention.py     # Hallucination prevention integration tests
├── test_memory_manager.py               # Memory management tests
└── test_reasoning_strategy.py           # Reasoning strategy tests
```

## Installation

### Install test dependencies

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run with coverage report

```bash
pytest --cov=src --cov-report=html
```

### Run specific test file

```bash
pytest tests/test_rag_assistant.py -v
```

### Run specific test class

```bash
pytest tests/test_rag_assistant.py::TestRAGAssistantInitialization -v
```

### Run specific test

```bash
pytest tests/test_rag_assistant.py::TestRAGAssistantInitialization::test_initialization_success -v
```

### Run tests by marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only hallucination prevention tests
pytest -m hallucination

# Run all except slow tests
pytest -m "not slow"
```

### Run with parallel execution

```bash
# Requires pytest-xdist
pytest -n auto
```

### Run with timeout

```bash
# Each test times out after 30 seconds
pytest --timeout=30
```

### Run tests with detailed output

```bash
pytest -vv --tb=long
```

### Generate HTML report

```bash
pytest --html=report.html --self-contained-html
```

## Test Coverage

### View coverage summary

```bash
pytest --cov=src --cov-report=term-missing
```

### Generate HTML coverage report

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Check coverage for specific file

```bash
pytest --cov=src/rag_assistant --cov-report=term-missing
```

## Test Categories

### 1. RAG Assistant Unit Tests (`test_rag_assistant.py`)

Tests for the core `RAGAssistant` class:

- **Initialization Tests**
  - LLM initialization
  - VectorDB connection
  - Memory manager setup
  - Reasoning strategy loading
  - Chain building

- **Document Addition Tests**
  - Adding string list documents
  - Adding dictionary documents
  - Handling empty documents

- **Invocation Tests**
  - Basic invoke functionality
  - Context retrieval
  - Custom n_results parameter
  - Memory tracking
  - Empty context handling

- **Exception Handling Tests**
  - VectorDB connection errors
  - Strategy loading errors
  - LLM invocation errors

### 2. Prompt Builder Tests (`test_prompt_builder.py`)

Tests for prompt construction and constraint enforcement:

- **Basic Functionality**
  - Prompt list generation
  - Non-empty strings
  - Appropriate count

- **Constraints**
  - Critical constraint present
  - General knowledge forbidden
  - No inference allowed
  - Rejection message specified
  - No fallback clause

- **Role & Tone**
  - Role configuration
  - Tone instructions
  - Language style

- **Format Instructions**
  - Markdown specification
  - Conciseness requirement
  - Output formatting

- **Reasoning Strategy Integration**
  - Strategy instructions included
  - RAG-Enhanced reasoning specifics
  - Disabled strategy handling
  - Error handling

- **Special Cases**
  - Follow-up questions
  - Polite greetings
  - Gibberish handling
  - Related topics

### 3. Hallucination Prevention Tests (`test_hallucination_prevention.py`)

Integration tests for preventing LLM hallucinations:

- **Constraint Tests**
  - Critical constraint verification
  - Rejection message verification
  - Etymology example presence
  - General knowledge forbiddance
  - No inference rule
  - Document grounding requirement

- **Integration Tests**
  - Out-of-scope query rejection
  - RAG-Enhanced reasoning inclusion
  - In-scope query context retrieval
  - Context passage to LLM

- **Edge Cases**
  - Empty document collection
  - Gibberish queries
  - Ambiguous follow-ups
  - Etymology example handling
  - Multi-query consistency

### 4. Memory Manager Tests (`test_memory_manager.py`)

Tests for conversation memory functionality:

- **Initialization**
  - Memory manager setup
  - Strategy loading

- **Message Handling**
  - User message addition
  - Assistant message addition
  - Memory variable retrieval

- **Strategy Switching**
  - Buffer memory switching
  - Summarization switching
  - None strategy (disabled memory)

- **Conversation Flow**
  - Multi-turn conversations
  - Context accumulation

- **Edge Cases**
  - Empty messages
  - Very long messages
  - Special characters

- **Exception Handling**
  - Memory save errors
  - Config load errors
  - Invalid strategies

### 5. Reasoning Strategy Tests (`test_reasoning_strategy.py`)

Tests for reasoning strategy loading and application:

- **Initialization**
  - Successful initialization
  - Custom config path

- **Strategy Retrieval**
  - Active strategy retrieval
  - Strategy name
  - Strategy description
  - Strategy instructions
  - Few-shot examples

- **Validation**
  - Strategy enabled check
  - Strategy disabled check
  - Invalid strategy error

- **Query All Enabled**
  - Get enabled strategies

- **Prompt Building**
  - Complete strategy prompt

- **Edge Cases**
  - Empty instructions
  - Missing optional fields
  - Very long instruction lists

- **Exception Handling**
  - Config load errors
  - Malformed YAML

## Test Fixtures

### Sample Data Fixtures (`fixtures/sample_data.py`)

Provides reusable test data:

- `sample_documents` - List of varied documents
- `sample_documents_empty` - Empty document list
- `sample_documents_single` - Single document
- `sample_queries_in_scope` - Answerable queries
- `sample_queries_out_of_scope` - Out-of-scope queries
- `sample_queries_ambiguous` - Ambiguous queries
- `sample_queries_gibberish` - Nonsensical queries
- `sample_queries_special_cases` - Special case queries
- `sample_context` - Retrieved context example
- `sample_context_empty` - Empty context
- `search_results_with_docs` - Mock search results
- `search_results_empty` - Empty search results
- `memory_messages` - Sample conversation history
- `mock_llm_response` - Mock response
- `mock_llm_response_out_of_scope` - Out-of-scope response
- `mock_llm_response_unclear` - Unclear query response

## Test Markers

Use markers to organize and filter tests:

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Hallucination prevention tests
pytest -m hallucination

# Memory tests
pytest -m memory

# Reasoning tests
pytest -m reasoning

# Prompt tests
pytest -m prompt

# Edge case tests
pytest -m edge_case

# Slow tests (skip with "not slow")
pytest -m "not slow"
```

## Continuous Integration

### Running tests in CI/CD pipeline

```bash
# Install dependencies
pip install -r requirements-test.txt

# Run tests with coverage
pytest --cov=src --cov-report=xml --cov-report=term-missing

# Generate reports
pytest --html=report.html --self-contained-html
```

## Best Practices

1. **Use fixtures** - Leverage `conftest.py` and `fixtures/sample_data.py` for common test data
2. **Mock external calls** - Use `unittest.mock` for LLM, VectorDB, and file I/O
3. **Test one thing** - Each test should verify a single behavior
4. **Descriptive names** - Test names should describe what is being tested
5. **Clear assertions** - Use specific assertions with meaningful messages
6. **Handle edge cases** - Test empty inputs, very large inputs, special characters
7. **Test exceptions** - Verify proper exception handling and error messages

## Troubleshooting

### Import errors

If you get import errors, ensure the `src` directory is in Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/src"
pytest
```

### Mock issues

If mocks aren't working, check:
- Correct patch path (should be where the object is used, not defined)
- Mock return values are set correctly
- MagicMock is used for complex objects

### Coverage not accurate

Run with fresh environment:

```bash
rm -rf .pytest_cache htmlcov .coverage
pytest --cov=src --cov-report=html
```

## Example Test Commands

```bash
# Quick smoke test
pytest -x

# Full test suite with coverage
pytest --cov=src --cov-report=html

# Unit tests only with verbose output
pytest -m unit -vv

# Integration tests with detailed output
pytest -m integration --tb=long

# Test specific component
pytest tests/test_rag_assistant.py -v

# Test with HTML report
pytest --html=report.html --self-contained-html

# Parallel execution (faster)
pytest -n auto

# Run and stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run failed tests first, then others
pytest --ff

# Generate coverage report for visualization
pytest --cov=src --cov-report=html && open htmlcov/index.html
```

## Contributing Tests

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests pass
3. Check coverage remains above 70%
4. Update this README if adding new test categories

## Coverage Goals

- **Target**: 70% code coverage minimum
- **Ideal**: 80%+ code coverage
- **Critical paths**: 100% coverage for hallucination prevention

View current coverage:

```bash
pytest --cov=src --cov-report=term-missing
```
