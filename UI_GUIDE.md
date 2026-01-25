# RAG Chatbot UI Guide

## Quick Start

### 1. Install Dependencies
Make sure Streamlit is installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run src/streamlit_app.py
```

This will open a local web server at `http://localhost:8501`

## Features

### 🤖 Web Interface
- **Clean, modern chat interface** with message history
- **Real-time responses** from the RAG assistant
- **Document-based answers** using Retrieval-Augmented Generation
- **Responsive design** that works on desktop and mobile

### ⚙️ Sidebar Controls
- **Initialize RAG Assistant** - Load documents and set up the chatbot
- **Clear Chat History** - Start a fresh conversation
- **Status indicator** - See if the assistant is ready
- **Configuration info** - API key requirements

### 💬 Chat Features
- **Persistent chat history** during the session
- **User and assistant message differentiation** with color coding
- **Loading indicators** while processing
- **Error handling** with helpful messages

## How It Works

1. **Click "Initialize RAG Assistant"** in the sidebar
   - Loads all documents from the `/data` folder
   - Initializes the RAG assistant with embeddings
   - Creates vector database for semantic search

2. **Ask Questions** about the documents
   - Type your question in the text input
   - Press "Send" or Enter
   - The assistant retrieves relevant documents and generates answers

3. **View Chat History**
   - All messages are displayed in chronological order
   - Color-coded: Blue for user, Gray for assistant
   - Automatically scrolls to latest message

## API Key Setup

The chatbot supports multiple LLM providers. Set up at least one in your `.env` file:

```
# OpenAI
OPENAI_API_KEY=your_openai_key

# Groq (Llama models - faster & free)
GROQ_API_KEY=your_groq_key

# Google Gemini
GOOGLE_API_KEY=your_google_key
```

## Customization

### Styling
Edit the CSS in `streamlit_app.py` to customize colors and layout:
```python
st.markdown("""
    <style>
    .chat-message { ... }
    .user-message { background-color: #e3f2fd; }
    .assistant-message { background-color: #f5f5f5; }
    </style>
""", unsafe_allow_html=True)
```

### Configuration
Modify these settings in `streamlit_app.py`:
- Page title and icon
- Layout (wide vs narrow)
- Message styling
- Button text and labels

## Troubleshooting

### "Error initializing assistant"
- Check that your API keys are set in `.env`
- Verify that document files exist in `/data` folder
- Ensure all dependencies are installed

### "Searching documents..." takes too long
- This is normal for the first query (embeddings are computed)
- Subsequent queries should be faster
- Try asking shorter, more specific questions

### Chat history disappeared
- Refresh the page to reload
- Click "Clear Chat History" and reinitialize to start fresh

## CLI Alternative

If you prefer command-line interaction, you can still use:
```bash
python src/app.py
```

This provides the same RAG functionality without the web UI.
