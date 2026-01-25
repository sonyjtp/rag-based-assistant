# 🤖 RAG-Based AI Assistant - AAIDC Project

> A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions exclusively from a set of custom documents using LangChain, ChromaDB, and multiple LLM providers.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-175%20passing-brightgreen.svg)]()
[[[[[![Code Coverage](https://img.shields.io/badge/coverage-91.28%25-brightgreen.svg)]()
[![Pylint](https://github.com/sonyjtp/rag-based-assistant/actions/workflows/pylint.yml/badge.svg)](https://github.com/sonyjtp/rag-based-assistant/actions/workflows/pylint.yml)

[Quick Start](#-quick-start) • [Features](#-features) • [Installation](#-installation) • [Contributing](#-contributing)


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
- [Testing](#-testing)
- [Customization Guide](#-customization-guide)
- [Memory Management](#-memory-management)
- [Reasoning Strategies](#-reasoning-strategies)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that:

- 📚 **Loads custom documents** from your `data/` directory
- 🔍 **Chunks and embeds** text using advanced text splitting strategies
- 💾 **Stores vectors** in ChromaDB vector database
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
- ✅ **Buffer Memory**: Stores full conversation history
- ✅ **Sliding Window Memory**: Keeps recent messages + summarized history
- ✅ **Summarization**: Automatic conversation summarization when window fills
- ✅ **Memory Strategy Switching**: Change strategies on-the-fly

### LLM Integration
- ✅ **OpenAI GPT-4** / GPT-4o-mini
- ✅ **Groq Llama 3.1** (fast inference)
- ✅ **Google Gemini** Pro
- ✅ Automatic fallback to next available provider
- ✅ Device detection (CUDA, MPS, CPU)

### Reasoning Strategies
- ✅ **Chain-of-Thought**: Step-by-step reasoning
- ✅ **Tree-of-Thought**: Explores multiple reasoning paths
- ✅ **Self-Consistent**: Generates multiple outputs and votes
- ✅ Configurable via YAML

### Safety & Quality
- ✅ **Hallucination Prevention**: Strict prompt constraints
- ✅ **Input Validation**: Document and query validation
- ✅ **Error Handling**: Comprehensive exception handling
- ✅ **Logging**: Detailed logging throughout
- ✅ **191 Test Cases**: 78% code coverage

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
git clone https://github.com/sonyjtp/rt-aaidc-rag-based-assistant.git
cd rt-aaidc-rag-based-assistant

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

# Verify installation
python -c "import langchain; print('✓ LangChain installed')"
```

### Docker Installation (Optional)

```bash
# Build Docker image
docker build -t rag-assistant .

# Run container
docker run -e OPENAI_API_KEY=your_key -v $(pwd)/data:/app/data rag-assistant
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```env
# LLM Configuration
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIzaSy...
OPENAI_MODEL=gpt-4o-mini
GROQ_MODEL=llama-3.1-8b-instant

# Vector Database
CHROMA_API_KEY=your_api_key
CHROMA_TENANT=default
CHROMA_DATABASE=default

# Embedding Model
VECTOR_DB_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Memory Strategy
MEMORY_STRATEGY=conversation_buffer_memory  # or summarization_sliding_window

# Retrieval
RETRIEVAL_K=5  # Number of documents to retrieve

# Text Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Reasoning Strategy
REASONING_STRATEGY=chain_of_thought
```

### Configuration Files

**config.py** - Core configuration
```python
CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP_DEFAULT = 200
RETRIEVAL_K_DEFAULT = 5
```

**config/prompt-config.yaml** - System prompts and constraints
```yaml
system_prompts:
  - "Only answer based on provided documents"
  - "Do not use training data or general knowledge"
  - "If information not found: respond with 'I'm sorry, that information is not known to me.'"
```

**config/memory_strategies.yaml** - Memory configuration
```yaml
memory_strategies:
  conversation_buffer_memory:
    enabled: true
    parameters:
      memory_key: chat_history
  summarization_sliding_window:
    enabled: true
    parameters:
      window_size: 5
      memory_key: chat_history
```

**config/reasoning_strategies.yaml** - Reasoning approaches
```yaml
reasoning_strategies:
  chain_of_thought:
    enabled: true
    instructions: "Think step by step..."
  tree_of_thought:
    enabled: true
    instructions: "Explore multiple paths..."
```

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

### Python API

```python
from src.rag_assistant import RAGAssistant

# Initialize assistant
assistant = RAGAssistant()

# Add documents
documents = [
    {"content": "Document text...", "title": "Doc 1", "filename": "doc1.txt"}
]
assistant.add_documents(documents)

# Ask questions
response = assistant.invoke("What is the main topic?")
print(response)

# Get memory history
memory_vars = assistant.memory_manager.get_memory_variables()
print(memory_vars["chat_history"])
```

---

## 🏗️ Project Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  ┌──────────────┐      ┌──────────────┐                │
│  │   CLI App    │      │  Streamlit   │                │
│  │  (app.py)    │      │    (web UI)  │                │
│  └──────────────┘      └──────────────┘                │
└───────────────┬────────────────────────────┬────────────┘
                │                            │
                ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG Assistant Core                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  RAGAssistant                                    │  │
│  │  - invoke(query)                                 │  │
│  │  - add_documents(docs)                           │  │
│  │  - retrieve_context(query, k)                    │  │
│  └──────────────────────────────────────────────────┘  │
└───────────┬──────────────┬──────────────┬───────────────┘
            │              │              │
    ┌───────▼──┐    ┌──────▼────┐  ┌────▼─────┐
    │ VectorDB │    │   Memory   │  │ Prompt   │
    │          │    │  Manager   │  │ Builder  │
    │ ChromaDB │    │ (Buffer or │  │          │
    │          │    │ Summarized)│  │ System   │
    └──────────┘    └────────────┘  └──────────┘
            │              │              │
            ▼              ▼              ▼
    ┌─────────────────────────────────────┐
    │       LLM Integration               │
    │ ┌──────┐ ┌─────┐ ┌──────────┐      │
    │ │OpenAI│ │Groq │ │  Google  │      │
    │ └──────┘ └─────┘ └──────────┘      │
    └─────────────────────────────────────┘
```

### Data Flow

```
User Query
    │
    ▼
Document Search (VectorDB)
    │
    ├─► Retrieve relevant documents (k=5)
    │
    ▼
Context Building
    │
    ├─► Combine context with history
    ├─► Add system prompts
    │
    ▼
LLM Processing
    │
    ├─► Apply reasoning strategy
    ├─► Generate response
    │
    ▼
Memory Update
    │
    ├─► Save to conversation history
    ├─► Apply memory strategy
    │
    ▼
Response to User
```

---

## 📁 Project Structure

```
rt-aaidc-rag-based-assistant/
│
├── src/                          # Source code
│   ├── app.py                   # CLI interface
│   ├── streamlit_app.py         # Web UI
│   ├── rag_assistant.py         # Core RAG logic (98% tested)
│   ├── vectordb.py              # Vector database wrapper
│   ├── chroma_client.py         # ChromaDB client
│   ├── embeddings.py            # Embedding model initialization
│   ├── llm_utils.py             # LLM provider selection
│   ├── prompt_builder.py        # Prompt generation (97% tested)
│   ├── memory_manager.py        # Memory handling (81% tested)
│   ├── sliding_window_memory.py # Summarization-based memory (90% tested)
│   ├── reasoning_strategy_loader.py  # Reasoning strategies (100% tested)
│   ├── file_utils.py            # File I/O utilities
│   ├── config.py                # Configuration (100% tested)
│   └── logger.py                # Logging setup (96% tested)
│
├── config/                       # Configuration files
│   ├── prompt-config.yaml       # System prompts
│   ├── memory_strategies.yaml   # Memory configurations
│   └── reasoning_strategies.yaml # Reasoning strategies
│
├── data/                         # Document storage
│   ├── sample_doc1.txt
│   ├── sample_doc2.txt
│   └── ...
│
├── tests/                        # Test suite (191 tests)
│   ├── test_rag_assistant.py           (26 tests)
│   ├── test_prompt_builder.py          (35 tests)
│   ├── test_hallucination_prevention.py (15 tests)
│   ├── test_memory_manager.py          (16 tests)
│   ├── test_reasoning_strategy.py      (31 tests)
│   ├── test_embeddings.py              (16 tests)
│   ├── test_file_utils.py              (32 tests)
│   ├── test_sliding_window_memory.py   (39 tests)
│   ├── test_integrations.py            (20 tests)
│   └── test_app.py                     (5 tests)
│
├── logs/                         # Application logs
│   ├── debug.log
│   └── rag_assistant.log
│
├── requirements.txt              # Python dependencies
├── requirements-test.txt         # Testing dependencies
├── pytest.ini                    # Pytest configuration
├── .env.example                  # Example environment variables
├── .coveragerc                   # Coverage configuration
├── README.md                     # This file
├── LICENSE                       # MIT License
└── .gitignore
```

---

## 🧪 Testing

### Run Full Test Suite

```bash
# Run all 191 tests
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
# ✅ Code formatting (Black, isort)
# ✅ Code linting (Pylint, Flake8)
# ✅ Test coverage (minimum 90%)
```

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

### Test Coverage

```
Overall Coverage: 78%
┌──────────────────────────┬──────────┐
│ Module                   │ Coverage │
├──────────────────────────┼──────────┤
│ rag_assistant.py         │ 98%      │
│ prompt_builder.py        │ 97%      │
│ reasoning_strategy_loader│ 100%     │
│ config.py                │ 100%     │
│ memory_manager.py        │ 81%      │
│ sliding_window_memory.py │ 90%      │
│ embeddings.py            │ 90%      │
│ file_utils.py            │ 90%      │
│ chroma_client.py         │ 85%      │
└──────────────────────────┴──────────┘
```

---

## 🎛️ Customization Guide

### Change Memory Strategy

```python
# In config.py or .env
MEMORY_STRATEGY = "conversation_buffer_memory"  # Or "summarization_sliding_window"

# In code
from src.memory_manager import MemoryManager
memory = MemoryManager(llm=llm, strategy="summarization_sliding_window")
```

### Switch LLM Provider

```bash
# In .env - set which API key to use
OPENAI_API_KEY=sk-...    # Uses OpenAI
# GROQ_API_KEY=...       # Commented out - won't use Groq
# GOOGLE_API_KEY=...     # Commented out - won't use Google
```

### Adjust Document Chunking

```bash
# In .env
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

### Buffer Memory
- **Use case**: Short conversations (< 20 messages)
- **Pros**: Remembers everything, simple
- **Cons**: Token usage grows, no summarization

```yaml
conversation_buffer_memory:
  enabled: true
  parameters:
    memory_key: chat_history
```

### Sliding Window Memory
- **Use case**: Long conversations (100+ messages)
- **Pros**: Keeps recent context, summarizes old conversations
- **Cons**: Requires LLM for summarization

```yaml
summarization_sliding_window:
  enabled: true
  parameters:
    window_size: 5        # Keep last 5 messages
    memory_key: chat_history
```

### Disable Memory
```bash
MEMORY_STRATEGY=none
```

---

## 🎯 Reasoning Strategies

### Available Strategies

1. **Chain-of-Thought**
   - Step-by-step reasoning
   - Best for: Complex questions requiring multiple steps

2. **Tree-of-Thought**
   - Explores multiple reasoning paths
   - Best for: Questions with multiple valid approaches

3. **Self-Consistent**
   - Generates multiple answers, picks best
   - Best for: Ensuring consistent, reliable answers

```bash
# Set in .env
REASONING_STRATEGY=chain_of_thought
```

---

## ❓ Troubleshooting

### Common Issues

#### "API Key not found"
```bash
# Solution: Check your .env file
cat .env | grep API_KEY

# Make sure the file exists and has correct keys
cp .env_example .env
# Edit .env with your actual API key
```

#### "No documents found"
```bash
# Solution: Add .txt files to data/ directory
ls data/
# Should show your document files

# Or load documents programmatically
assistant.add_documents([{"content": "...", "title": "Doc1"}])
```

#### "Out of memory / token limit exceeded"
```bash
# Solution 1: Use smaller chunk size
CHUNK_SIZE=500

# Solution 2: Reduce retrieval results
RETRIEVAL_K=3

# Solution 3: Use sliding window memory
MEMORY_STRATEGY=summarization_sliding_window
```

#### "LLM not responding / Timeout"
```bash
# Solution: Switch to faster LLM
# In .env, use Groq (fastest and free):
GROQ_API_KEY=gsk_...
# Comment out other API keys
```

### Debug Mode

```bash
# Enable detailed logging
# In logger.py, set logging level
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
pytest -v --log-cli-level=DEBUG
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/rt-aaidc-rag-based-assistant.git
cd rt-aaidc-rag-based-assistant

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

**Your Name** - AAIDC Project Contributor

- 📧 Email: your.email@example.com
- 🐙 GitHub: [@yourusername](https://github.com/yourusername)
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM orchestration framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Groq](https://groq.com/) - Fast LLM inference
- [OpenAI](https://openai.com/) - GPT models
- [Google](https://ai.google.dev/) - Gemini models

---

## 📞 Support

Need help? Here are your options:

1. **Check Documentation**: Read this README and config files
2. **Review Examples**: Check `tests/` for usage examples
3. **Search Issues**: Look for similar issues on GitHub
4. **Create Issue**: If problem persists, create a GitHub issue
5. **Discussions**: Join community discussions on GitHub

---

**Last Updated**: January 2026
**Status**: ✅ Production Ready | 191 Tests Passing | 78% Coverage
