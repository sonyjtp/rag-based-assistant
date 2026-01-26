# 🤖 RAG-Based AI Assistant

> A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions exclusively from a set of custom documents using LangChain, ChromaDB, and multiple LLM providers.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Coverage](https://img.shields.io/badge/coverage-91.28%25-brightgreen.svg)]()
[![Pylint](https://github.com/sonyjtp/rag-based-assistant/actions/workflows/pylint.yml/badge.svg)](https://github.com/sonyjtp/rag-based-assistant/actions/workflows/pylint.yml)

[Quick Start](#-quick-start) • [Features](#-features) • [Installation](#-installation)


---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Customization Guide](#-customization-guide)
- [Memory Management](#-memory-management)
- [Reasoning Strategies](#-reasoning-strategies)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that:

- 📚 **Loads custom documents** from your `data/` directory
- 🔍 **Chunking**: Split documents into chunks and add metadata.
- 💾 **Storage**: Store each chunk's embedding (vector), the chunk text, and metadata in ChromaDB for retrieval.
- 🎤 **Answers questions** exclusively from your documents
- 🧠 **Maintains conversation** memory across multiple interactions
- 🔌 **Supports multiple LLMs**: OpenAI, Groq, Google Gemini
- 🛡️ **Prevents hallucination** with strict prompt constraints
- 📊 **Tracks reasoning** with configurable strategies

**Key Constraint**: The assistant **only answers questions based on the provided documents**. Questions that cannot be answered from the documents are rejected with: *"I'm sorry, that information is not known to me."*

---

## ✨ Features

### Core RAG Capabilities
- ✅ Document loading from text files
- ✅ Intelligent text chunking with overlap
- ✅ Semantic search using embeddings
- ✅ Context-aware question answering
- ✅ Document metadata preservation (title, tags, filename)

### Memory Management
- ✅ **Buffer Memory** (simple_buffer): Stores full conversation history.
- ✅ **Sliding Window Memory** (summarization_sliding_window) — default: keeps recent messages plus a running summarized history to stay within token limits.
- ✅ **Summarization** (summary): Maintains a running summary of the conversation.
- ✅ **None** (none): Disables conversation memory.
- ✅ **Memory Strategy Switching**: Change via `MEMORY_STRATEGY` in `src/config.py` or by toggling `enabled` in `config/memory_strategies.yaml`.

### LLM Integration
- ✅ **OpenAI GPT-4** / GPT-4o-mini
- ✅ **Groq Llama 3.1** (fast inference)
- ✅ **Google Gemini** Pro
- ✅ Automatic fallback to next available provider
- ✅ Device detection & selection — Automatically picks the best available compute device for model inference and embeddings

**Device Detection order**:
  1. `CUDA` — NVIDIA GPUs (highest performance).
  2. `MPS` — Apple Metal Performance Shaders on Apple Silicon (macOS).
  3. `CPU` — Fallback when no GPU acceleration is available.

### Reasoning Strategies

- ✅ **RAG-Enhanced Reasoning** (rag_enhanced_reasoning) — default: Retrieve relevant documents first, then apply reasoning grounded in those documents; `enabled: true`.
- ✅ **Chain-of-Thought** (chain_of_thought): Step-by-step internal reasoning before the final answer; `enabled: true`.
- ✅ **Few-Shot Prompting** (few_shot_prompting): Include examples in the prompt to guide format and style; `enabled: true`.
- ✅ **Structured Prompting** (structured_prompting): Use templates/format specifications for consistent, parseable outputs; `enabled: true`.
- ✅ **Metacognitive Prompting** (metacognitive_prompting): Reflect on confidence, limitations, and uncertainty; `enabled: true`.



### Safety & Quality
- ✅ **Hallucination Prevention**: Strict prompt constraints
- ✅ **Input Validation**: Document and query validation
- ✅ **Error Handling**: Comprehensive exception handling
- ✅ **Logging**: Detailed logging throughout
- ✅ **Test Cases**: Code coverage maintained above 85%

### User Interfaces
- ✅ **CLI Interface** (`app.py`): Command-line chatbot
- ✅ **Streamlit UI** (`streamlit_app.py`): Web-based interface
- ✅ **API Ready**: Can be integrated with FastAPI/Flask

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (Tested with 3.12.12 ✅)
- **API Key** for at least one LLM provider:
  - OpenAI: `OPENAI_API_KEY`
  - Groq: `GROQ_API_KEY`
  - Google: `GOOGLE_API_KEY`

### 1️⃣ Clone & Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/sonyjtp/rag-based-assistant.git
cd rag-based-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure API Key (1 minute)

```bash
# Copy example env file
cp .env_example .env

# Edit .env with your API key
# Choose ONE provider:
# Option 1: OpenAI
OPENAI_API_KEY=your_openai_key_here

# Option 2: Groq (recommended - fast and free)
GROQ_API_KEY=your_groq_key_here

# Option 3: Google Gemini
GOOGLE_API_KEY=your_google_key_here
```

### 3️⃣ Add Your Documents (2 minutes)

```bash
# Replace sample files in data/ with your documents
# Files should be .txt format

ls data/
# Output: your_document.txt, another_doc.txt, ...
```

### 4️⃣ Run the Assistant (30 seconds)

**CLI Version:**
```bash
python src/app.py
```

**Web UI (Streamlit):**
```bash
streamlit run src/streamlit_app.py
```

---

## 📦 Installation

### Full Installation with Development Tools

```bash
# Clone repository
git clone https://github.com/yourusername/rt-aaidc-rag-based-assistant.git
cd rt-aaidc-rag-based-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development/test dependencies (optional)
pip install -r requirements-dev.txt

# Set up pre-commit hooks for automatic code formatting
pre-commit install
```

---

## ⚙️ Configuration

See [Quick Start](#-quick-start) for environment variable setup (`OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`).

For advanced configuration options, see:
- `src/config.py` — Core settings (chunk size, embedding model, LLM selection)
- `config/memory_strategies.yaml` — Memory strategy definitions
- `config/reasoning_strategies.yaml` — Reasoning approach configurations
- `config/prompt-config.yaml` — System prompts and safety constraints

Detailed strategy information is documented in [Memory Management](#-memory-management) and [Reasoning Strategies](#-reasoning-strategies) sections.
---

## 💬 Usage

### CLI Usage

```bash
python src/app.py

# Prompts you to ask questions
# Type 'quit' to exit

> What is the main topic of the documents?
Assistant: Based on the documents, the main topics are...

> Tell me more
Assistant: [Provides additional context from memory]

> quit
Goodbye!
```

### Streamlit Web Interface

```bash
streamlit run src/streamlit_app.py

# Opens http://localhost:8501
# - Sidebar: Clear history, configure settings
# - Main: Chat interface
# - Auto-saves conversation
```


---

## 🏗️ Project Architecture

### System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      User Interface                       │
│  ┌─────────────────┐          ┌─────────────────┐       │
│  │   CLI App       │          │  Streamlit      │       │
│  │   (app.py)      │          │   (web UI)      │       │
│  └────────┬────────┘          └────────┬────────┘       │
└───────────┼───────────────────────────┼────────────────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────┐
    │         RAGAssistant Core                   │
    │  - invoke(query) → Generate response        │
    │  - add_documents(docs) → Index documents    │
    │  - retrieve_context(query, k) → Search      │
    └──────┬──────────────┬──────────────┬────────┘
           │              │              │
    ┌──────▼──────┐ ┌──────▼─────┐ ┌──────▼──────┐
    │  VectorDB   │ │ Prompt     │ │ Reasoning   │
    │             │ │ Builder    │ │ Strategy    │
    │ ┌─────────┐ │ │            │ │ Loader      │
    │ │ChromaDB │ │ │ System     │ │             │
    │ │ Client  │ │ │ Prompts    │ │ (Chain of   │
    │ └────┬────┘ │ │ Constraints│ │ Thought,    │
    │      │      │ └─────┬──────┘ │ ReAct, etc) │
    │ ┌────▼──────────┐   │        └─────┬───────┘
    │ │  Embeddings   │   │              │
    │ │ (HuggingFace  │   │              │
    │ │ Transformer)  │   │              │
    │ └───────────────┘   │              │
    └────────┬────────────┴──────────────┘
             │
    ┌────────▼────────────────────────┐
    │    Memory Manager               │
    │  ┌──────────────────────────┐   │
    │  │ Strategy Pattern         │   │
    │  │ ┌────────────────────┐   │   │
    │  │ │SlidingWindow       │   │   │
    │  │ │(default)           │   │   │
    │  │ ├────────────────────┤   │   │
    │  │ │SimpleBuffer        │   │   │
    │  │ ├────────────────────┤   │   │
    │  │ │Summary             │   │   │
    │  │ └────────────────────┘   │   │
    │  └──────────────────────────┘   │
    └────────┬────────────────────────┘
             │
    ┌────────▼────────────────────────┐
    │  LLM Integration                │
    │  ┌──────┐  ┌────┐  ┌──────────┐ │
    │  │OpenAI│  │Groq│  │ Google   │ │
    │  │      │  │    │  │ Gemini   │ │
    │  └──────┘  └────┘  └──────────┘ │
    └─────────────────────────────────┘

┌─────────────────────────────────────┐
│    Supporting Utilities             │
│  ├─ File Utils (document loading)   │
│  ├─ Logger (observability)          │
│  ├─ UI Utils (Streamlit styling)    │
│  └─ Config (centralized settings)   │
└─────────────────────────────────────┘
```

### Data Flow

```
User Query
    │
    ▼
Meta-Question Detection?
    ├─ Keywords: "what topics", "what can you", "what do you know"
    │ ├─ YES: Allow lower similarity matches
    │ └─ NO: Require high similarity (distance <= 0.35, i.e., similarity >= 0.65)
    │
    ▼
Document Search (VectorDB)
    │
    ├─► Convert query to embedding
    ├─► Search for similar documents (k results)
    ├─► Return ranked results with distances
    │
    ▼
Similarity Validation ⚡ (Hallucination Prevention)
    │
    ├─ Check: distance <= threshold?
    │ ├─ META-QUESTION: Allow any distance
    │ ├─ REGULAR QUESTION: Must pass threshold
    │ └─ NO MATCH: Return "couldn't find information" → END
    │
    ▼
Context Building
    │
    ├─► Extract and flatten documents
    ├─► Combine with conversation history (from Memory)
    ├─► Add system prompts & constraints
    ├─► Apply reasoning strategy
    │
    ▼
LLM Processing
    │
    ├─► Chain: [Prompt Template → LLM → Output Parser]
    ├─► Generate response grounded in context
    │
    ▼
Memory Update
    │
    ├─► Save Q&A pair to conversation history
    ├─► Apply memory strategy:
    │   ├─ SlidingWindow: Summarize when window full
    │   ├─ SimpleBuffer: Keep recent messages
    │   └─ Summary: Maintain running summary
    │
    ▼
Response to User ✅
    │
    └─► Return context-grounded answer
```

---

## 📁 Project Structure

```
rag-based-assistant/
│
├── src/                          # Source code
│   ├── app.py                   # CLI interface
│   ├── streamlit_app.py         # Web UI
│   ├── rag_assistant.py         # Core RAG logic
│   ├── vectordb.py              # Vector database wrapper
│   ├── chroma_client.py         # ChromaDB client
│   ├── embeddings.py            # Embedding model initialization
│   ├── llm_utils.py             # LLM provider selection
│   ├── prompt_builder.py        # Prompt generation
│   ├── memory_manager.py        # Memory handling
│   ├── sliding_window_memory.py # Summarization-based memory
│   ├── simple_buffer_memory.py  # Buffer memory implementation
│   ├── summary_memory.py        # Summary memory implementation
│   ├── reasoning_strategy_loader.py  # Reasoning strategies
│   ├── file_utils.py            # File I/O utilities
│   ├── config.py                # Configuration
│   ├── ui_utils.py              # Streamlit UI utilities
│   └── logger.py                # Logging setup
│
├── config/                       # Configuration files
│   ├── prompt-config.yaml       # System prompts & constraints
│   ├── memory_strategies.yaml   # Memory strategy configurations
│   └── reasoning_strategies.yaml # Reasoning strategy definitions
│
├── static/                       # Static assets
│   └── css/
│       └── styles.css           # Streamlit custom styling
│
├── data/                         # Document storage (user documents)
│   └── *.txt                    # Text documents for RAG
│
├── tests/                        # Test suite
│   ├── conftest.py              # Pytest configuration
│   ├── fixtures/
│   │   └── sample_data.py       # Test fixtures & sample data
│   ├── test_rag_assistant.py
│   ├── test_prompt_builder.py
│   ├── test_hallucination_prevention.py
│   ├── test_memory_manager.py
│   ├── test_reasoning_strategy.py
│   ├── test_embeddings.py
│   ├── test_file_utils.py
│   ├── test_sliding_window_memory.py
│   ├── test_buffer_and_summary_memory.py
│   ├── test_integrations.py
│   ├── test_ui_utils.py
│   └── test_app.py
│
├── logs/                         # Application logs
│   ├── debug.log
│   └── rag_assistant.log
│
├── .github/                      # GitHub configuration
│   └── workflows/               # CI/CD workflows (optional)
│
├── htmlcov/                      # HTML coverage reports (generated)
│
├── requirements.txt              # Production dependencies
├── requirements-test.txt         # Testing dependencies
├── requirements-dev.txt          # Development tools (pre-commit, black, isort, pylint)
├── pytest.ini                    # Pytest configuration
├── .pylintrc                     # Pylint configuration
├── .pre-commit-config.yaml       # Pre-commit hooks configuration
├── .coveragerc                   # Coverage configuration
├── .gitignore                    # Git ignore rules
├── .env_example                  # Example environment variables
│
├── update_coverage.py            # Coverage badge update script
├── TESTING.md                    # Testing guide & instructions
├── UI_GUIDE.md                   # Streamlit UI guide
├── README.md                     # This file
└── LICENSE                       # License
```

---

## 🧪 Testing

### Run Full Test Suite

```bash
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Pre-Commit Testing

Before you commit, the following checks run automatically:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Manual run of all checks
pre-commit run --all-files

# Pre-commit checks include:
# ✅ Standard checks (trailing whitespace, file endings, YAML, merge conflicts)
# ✅ Code formatting (Black, isort)
# ✅ Code linting (Flake8, Pylint ≥9.5 score)
# ✅ Tests (pytest - all tests must pass)
# ✅ Coverage (minimum 90% required)
```

**If a check fails**, fix the issues and commit again. Most checks (Black, isort, end-of-file-fixer) auto-fix issues, so you may need to stage the changes and retry.

**Note**: Commits will be rejected if test coverage drops below 90%. To bypass (not recommended):
```bash
git commit --no-verify  # Skip pre-commit hooks
```

### Coverage Requirements

- **Minimum Coverage**: 90% (enforced by pre-commit hooks)
- **Target Coverage**: 95%+
- **Critical Modules**: 100% (rag_assistant, config, reasoning_strategy_loader)

### Run Specific Tests

```bash
# Test RAG assistant
pytest tests/test_rag_assistant.py -v

# Test prompt building
pytest tests/test_prompt_builder.py -v

# Test hallucination prevention
pytest tests/test_hallucination_prevention.py -v

# Test memory management
pytest tests/test_memory_manager.py -v
```

### Coverage Badge Updates

The coverage badge in the README is automatically updated in CI/CD:

```bash
# Manual update (for local development)
python update_coverage.py

# This script:
# 1. Reads coverage.xml (generated by pytest)
# 2. Extracts coverage percentage
# 3. Updates README badge with current coverage
# 4. Colors badge based on coverage level (green/yellow/red)
```

The badge is updated:
- ✅ On every push to main (via GitHub Actions)
- ✅ Before pull requests (verify coverage meets threshold)
- ✅ Manually via `python update_coverage.py`


## 🎛️ Customization Guide

### Change Memory Strategy

Edit `config.py` to change the memory strategy:

```python
# In src/config.py
MEMORY_STRATEGY = "summarization_sliding_window"  # Options: summarization_sliding_window, simple_buffer, summary, none
```

Available memory strategies (defined in `config/memory_strategies.yaml`):
- **summarization_sliding_window** (default): Summarizes last N messages using sliding window
- **simple_buffer**: Stores recent conversation history in a buffer
- **summary**: Maintains a running summary of the conversation
- **none**: Disables conversation memory entirely

### Switch LLM Provider

```bash
# In .env - set which API key to use
OPENAI_API_KEY=sk-...    # Uses OpenAI
# GROQ_API_KEY=...       # Commented out - won't use Groq
# GOOGLE_API_KEY=...     # Commented out - won't use Google
```

### Adjust Document Chunking

```bash
# In config.py
CHUNK_SIZE=2000          # Larger chunks
CHUNK_OVERLAP=400        # More overlap for context
RETRIEVAL_K=10           # Retrieve more documents
```

### Configure Reasoning Strategy

```yaml
# In config/reasoning_strategies.yaml
reasoning_strategies:
  chain_of_thought:
    enabled: true
    instructions: "Think through this step by step..."
  tree_of_thought:
    enabled: true
    instructions: "Explore multiple reasoning paths..."
```

### Add Custom Prompts

```python
# In src/prompt_builder.py
def build_system_prompts():
    return [
        "Your custom instruction 1",
        "Your custom instruction 2",
        # ... existing prompts
    ]
```

---

## 🧠 Memory Management

Three memory strategies are available (configured in [config/memory_strategies.yaml](config/memory_strategies.yaml)):

- **summarization_sliding_window** (default): Summarizes last N messages to stay within token limits
- **simple_buffer**: Stores recent conversation history without summarization
- **summary**: Maintains a running summary of the entire conversation
- **none**: Disables conversation memory

Change the strategy in `src/config.py`:
```python
MEMORY_STRATEGY = "summarization_sliding_window"
```

### Disable Memory
```bash
MEMORY_STRATEGY=none
```

---

## 🎯 Reasoning Strategies

Four reasoning strategies are available (configured in [config/reasoning_strategies.yaml](config/reasoning_strategies.yaml)):

- **chain_of_thought** (default): Step-by-step reasoning before final answer
- **self_consistency**: Multiple reasoning paths with consensus answer
- **few_shot_prompting**: Provides examples to guide model responses
- **rag_enhanced_reasoning**: RAG-specific reasoning constraints

Change the strategy in `src/config.py`:
```python
REASONING_STRATEGY = "rag_enhanced_reasoning"
```

---

## ❓ Troubleshooting

| Issue                | Solution                                                                 |
|----------------------|--------------------------------------------------------------------------|
| API Key not found    | Set `OPENAI_API_KEY`, `GROQ_API_KEY`, or `GOOGLE_API_KEY` in `.env`      |
| No documents found   | Add `.txt` files to `data/` directory or use `assistant.add_documents()` |
| Token limit exceeded | Reduce `CHUNK_SIZE` or enable memory summarization in config             |
| Low answer quality   | Increase `RETRIEVAL_K_DEFAULT` to retrieve more documents                |
| Hallucination issues | Ensure documents are loaded and similarity threshold is set correctly    |

---

### Debug Mode

```bash
# Enable detailed logging
# In logger.py, set logging level
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
pytest -v --log-cli-level=DEBUG
```

---



### Development Setup

```bash
# Fork and clone
git clone https://github.com/sonyjtp/rag-based-assistant.git
cd rag-based-assistant

# Create feature branch
git checkout -b feature/amazing-feature

# Install dev dependencies
pip install -r requirements-test.txt

# Make changes and run tests
pytest tests/ -v

# Commit and push
git add .
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature

# Create pull request on GitHub
```

### Testing Requirements

All contributions must include:
- ✅ Unit tests for new functionality
- ✅ Integration tests if applicable
- ✅ Documentation updates
- ✅ All tests must pass: `pytest -v`

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Comment complex logic

---

## 📚 Learning Resources

### RAG Concepts
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)

### LLM Integration
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Groq API Docs](https://console.groq.com/docs/)
- [Google Gemini Docs](https://ai.google.dev/docs/)

### Advanced Topics
- [Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Retrieval Strategies](https://arxiv.org/abs/2312.10997)
- [LLM Evaluation](https://github.com/openlifeScienceAI/ragger)

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0) - see [LICENSE](LICENSE) file for details.

**Key Points**:
- ✅ **Attribution**: You must credit the original authors
- ✅ **Share-Alike**: Any modifications must use the same license
- ❌ **Non-Commercial**: Cannot be used for commercial purposes
- ✅ **Modification**: You can modify the code

**What you CAN do**:
- Use for educational purposes
- Use in academic projects
- Use in non-commercial research
- Modify for personal use
- Share modifications (with same license)

**What you CANNOT do**:
- ❌ Use commercially
- ❌ Sell the software
- ❌ Use in commercial products
- ❌ Change the license

For the full license text, see [LICENSE](LICENSE) file.

```
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/
```

---

## 🎓 Author

**Sony Jacob Thomas**

---


**Last Updated**: January 2026
**Status**:  🛠️ Under Active Development
