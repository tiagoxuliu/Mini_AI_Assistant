# 🤖 AI Document Assistant - Local RAG System

A powerful **Retrieval-Augmented Generation (RAG)** system that allows you to chat with your documents using semantic search and local LLMs. Ask questions about PDFs, text files, and get accurate answers with source citations - all running locally on your machine!

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)

## ✨ Features

- 📄 **Multi-format Support**: Process PDF and TXT files
- 🔍 **Semantic Search**: Find answers based on meaning, not just keywords
- 🤖 **Local LLM**: Privacy-first - runs completely offline with Ollama
- 💬 **Streaming Responses**: Real-time answers like ChatGPT
- 🎯 **Smart Retrieval**: Filename boosting + semantic search hybrid approach
- 📚 **Source Attribution**: Know exactly which documents were used
- 🐛 **Debug Mode**: Inspect retrieved chunks and similarity scores
- ⚡ **Fast**: FAISS vector database for lightning-fast searches

## 🎬 Demo

```bash
❓ Your question: debug show me the poem from poem_minimal

🔍 RETRIEVED CHUNKS (for debugging):
✨ Filename boosting: ENABLED
============================================================

📄 Chunk 1 - poem_minimal.txt (distance: 1.4365)
   Rain taps the window,
   time pauses,
   and the city exhales.

📚 Sources: poem_minimal.txt (distance: 1.44)

💬 Answer: The poem is found in the source file `poem_minimal.txt`:

Rain taps the window,
time pauses,
and the city exhales.
```

## 🏗️ Architecture

```
┌─────────────────┐
│  Your Documents │ (PDF, TXT)
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Document Ingestion     │
│  - Load files           │
│  - Split into chunks    │
│  - Create embeddings    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  FAISS Vector Store     │
│  - 384-dim embeddings   │
│  - L2 distance search   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Semantic Search        │
│  - Query embedding      │
│  - Top-K retrieval      │
│  - Filename boosting    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  LLM (Mistral via       │
│  Ollama)                │
│  - Context + Question   │
│  - Streaming response   │
└─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Basic familiarity with command line

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-document-assistant.git
   cd ai-document-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the Mistral model**
   ```bash
   ollama pull mistral
   ```

### Usage

1. **Add your documents**
   ```bash
   # Place your PDF and TXT files in the data/ folder
   mkdir -p data
   cp your_documents.pdf data/
   cp your_notes.txt data/
   ```

2. **Process documents (ingestion)**
   ```bash
   python ingest.py
   ```
   
   Expected output:
   ```
   ✅ Loaded: your_document.pdf
   ✅ Loaded: your_notes.txt
   ✅ Created 42 chunks
   ✅ Successfully saved 42 chunks to FAISS index
   ```

3. **Start Ollama** (in a separate terminal)
   ```bash
   ollama serve
   ```

4. **Ask questions!**
   ```bash
   python main.py
   ```

### Example Queries

```bash
# Normal query
❓ Your question: what is the main topic of the document?

# Debug mode (see retrieved chunks)
❓ Your question: debug what are the key findings?

# Rank all chunks (see full ranking)
❓ Your question: rank what does the PDF say about AI?

# Disable filename boosting
❓ Your question: noboost show me python examples
```

## 📁 Project Structure

```
ai-document-assistant/
│
├── data/                      # Place your documents here
│   ├── document1.pdf
│   ├── notes.txt
│   └── ...
│
├── faiss_index/               # Generated vector database
│   ├── faiss_index.bin        # FAISS index file
│   └── chunks.pkl             # Chunk metadata
│
├── config.py                  # Configuration settings
├── ingest.py                  # Document processing pipeline
├── qa.py                      # Question-answering engine
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Configuration

Edit [`config.py`](config.py) to customize:

```python
# Model settings
MODEL_NAME = "mistral"          # Ollama model to use
TEMPERATURE = 0.5               # LLM creativity (0=factual, 1=creative)

# Chunking settings
CHUNK_SIZE = 1000               # Characters per chunk
CHUNK_OVERLAP = 200             # Overlap between chunks (preserves context)

# Retrieval settings
TOP_K = 10                      # Number of chunks to retrieve

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Recommended Settings

| Use Case | CHUNK_SIZE | CHUNK_OVERLAP | TOP_K | TEMPERATURE |
|----------|------------|---------------|-------|-------------|
| **Short documents** (emails, notes) | 500 | 100 | 5 | 0.3 |
| **Medium documents** (articles, reports) | 1000 | 200 | 7 | 0.5 |
| **Long documents** (books, research papers) | 1500 | 300 | 10 | 0.3 |
| **Code documentation** | 800 | 150 | 5 | 0.2 |

## 🧠 How It Works

### 1. Document Ingestion
```python
# Load documents (PDF + TXT)
documents = load_documents()

# Split into chunks with overlap
chunks = create_chunks(documents)

# Generate embeddings (384-dim vectors)
embeddings = create_embeddings(chunks)

# Store in FAISS index
save_to_faiss(embeddings, chunks)
```

### 2. Question Answering
```python
# Convert question to embedding
question_vector = embed_query(question)

# Search FAISS for similar chunks
relevant_chunks = faiss_search(question_vector, top_k=10)

# Apply filename boosting (optional)
boosted_chunks = boost_filename_matches(question, relevant_chunks)

# Build context and prompt LLM
context = build_context(boosted_chunks)
answer = llm.stream(context + question)
```

### 3. Filename Boosting (Hybrid Search)

The system combines **semantic search** with **keyword matching**:

```python
# If question contains "python" and file is "python_example.txt"
# Boost its ranking by reducing distance score

boost = -0.3  # Makes it rank higher
adjusted_distance = original_distance + boost
```

**Example:**
- Question: "show me python code"
- `python_example.txt` gets boosted
- Ranks higher even if semantic similarity is lower

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Ingestion Speed** | ~10 docs/second |
| **Embedding Model** | 384 dimensions |
| **Query Speed** | ~100ms (FAISS search) |
| **LLM Response Time** | 2-5 seconds (streaming) |
| **Memory Usage** | ~500MB (for 1000 chunks) |

## 🔧 Advanced Features

### Custom Embedding Models

Try different models for better performance:

```python
# config.py

# Faster, smaller (384 dims)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Better quality (768 dims)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Specialized for long documents (768 dims)
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
```

### Multiple LLM Models

```bash
# Try different Ollama models
ollama pull llama2
ollama pull codellama
ollama pull neural-chat

# Update config.py
MODEL_NAME = "llama2"
```

## 🐛 Troubleshooting

### Issue: "No connection could be made"

**Solution:** Make sure Ollama is running
```bash
ollama serve
```

### Issue: "Model 'mistral' not found"

**Solution:** Pull the model first
```bash
ollama pull mistral
ollama list  # Verify it's installed
```

### Issue: "poem_minimal.txt not retrieved"

**Causes:**
- Document wasn't in `data/` folder during ingestion
- Query is too vague

**Solutions:**
1. Re-run ingestion after adding document
2. Use more specific queries with keywords
3. Enable debug mode to inspect rankings

### Issue: High distance scores (>100)

**Cause:** Wrong embedding model configuration

**Solution:**
1. Delete `faiss_index/` folder
2. Update `config.py` with correct model
3. Re-run `python ingest.py`

## 📚 Dependencies

```
langchain>=0.1.0
langchain-community>=0.0.1
langchain-huggingface>=0.0.1
langchain-text-splitters>=0.0.1
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
PyPDF2>=3.0.0
ollama>=0.1.0
numpy>=1.24.0
```

## 🔮 Future Improvements

This project has room for exciting enhancements! Here are planned features:

### 📄 Document Support
- [ ] **DOCX files** - Microsoft Word documents
- [ ] **CSV files** - Structured data with column-aware chunking
- [ ] **JSON files** - API responses and structured data
- [ ] **Markdown files** - Code documentation and wikis
- [ ] **HTML/Web pages** - Scrape and index web content
- [ ] **Excel files** - Spreadsheets with multi-sheet support

### 💬 Conversation Features
- [ ] **Conversation history** - Remember previous questions
- [ ] **Multi-turn dialogue** - Follow-up questions with context
- [ ] **Session management** - Save and load conversation threads
- [ ] **Conversation summarization** - Auto-summarize long chats

### 🎨 User Interface
- [ ] **Streamlit web UI** - Beautiful browser-based interface
- [ ] **Gradio interface** - Quick demo-ready UI
- [ ] **CLI improvements** - Better terminal experience with Rich library
- [ ] **REST API** - Expose as microservice
- [ ] **Discord/Slack bot** - Chat with documents in team channels

### 🔍 Search Enhancements
- [ ] **Hybrid search** - BM25 (keyword) + semantic search combined
- [ ] **Re-ranking** - Use cross-encoder for better result ordering
- [ ] **Query expansion** - Auto-suggest related questions
- [ ] **Filters** - Search by date, file type, or custom metadata
- [ ] **Highlighted excerpts** - Show matching text with highlights

### 🗄️ Vector Database Options
- [ ] **ChromaDB** - Open-source alternative to FAISS
- [ ] **Pinecone** - Cloud-based vector database
- [ ] **Weaviate** - GraphQL-based vector search
- [ ] **Qdrant** - Rust-based high-performance DB
- [ ] **Milvus** - Scalable for production workloads

### 🧠 AI Features
- [ ] **Document summarization** - Auto-generate executive summaries
- [ ] **Key phrase extraction** - Identify important concepts
- [ ] **Question generation** - Suggest questions based on content
- [ ] **Multi-language support** - Translate queries and responses
- [ ] **Model comparison** - Side-by-side results from different LLMs
- [ ] **Confidence scores** - Show answer reliability

### ⚡ Performance
- [ ] **Batch processing** - Process multiple documents in parallel
- [ ] **Incremental updates** - Add new docs without full re-index
- [ ] **Caching** - Cache frequent queries for faster responses
- [ ] **GPU support** - Accelerate embedding generation
- [ ] **Distributed indexing** - Scale to millions of documents

### 🔐 Production Features
- [ ] **User authentication** - Multi-user support
- [ ] **Access control** - Document-level permissions
- [ ] **Audit logging** - Track all queries and access
- [ ] **Monitoring dashboard** - Usage analytics and metrics
- [ ] **Rate limiting** - API throttling
- [ ] **Docker deployment** - Containerized for easy deployment

### 🧪 Advanced RAG Techniques
- [ ] **Parent-child chunking** - Small chunks with large context retrieval
- [ ] **Hypothetical questions** - Generate questions from chunks for better matching
- [ ] **Chain-of-thought** - Multi-step reasoning for complex queries
- [ ] **Self-querying** - LLM extracts filters from natural language
- [ ] **Agentic RAG** - LLM decides when to retrieve more context

---

**Want to implement any of these?** Feel free to fork the repo and submit a pull request!

**Priority roadmap:**
1. Web UI (Streamlit) - Makes it accessible to non-technical users
2. Conversation history - Better user experience
3. Hybrid search - Improved retrieval quality
4. DOCX support - Most requested file format

## 📖 Learning Resources

This project demonstrates:
- **RAG architecture** (Retrieval-Augmented Generation)
- **Vector databases** (FAISS)
- **Embedding models** (HuggingFace Transformers)
- **LLM integration** (Ollama)
- **Document processing** (LangChain)
- **Streaming APIs**

**Related concepts:**
- Semantic search vs keyword search
- Chunking strategies and overlap
- Prompt engineering
- Vector similarity (L2 distance, cosine similarity)

## 🎓 Educational Notes

### What I Learned Building This

1. **RAG Pipeline Design**
   - Document loading → Chunking → Embedding → Vector storage → Retrieval → LLM

2. **Embedding Trade-offs**
   - Smaller models (384 dims) = faster, less accurate
   - Larger models (768 dims) = slower, more accurate

3. **Chunking Strategy**
   - `CHUNK_SIZE`: Balance context vs precision
   - `CHUNK_OVERLAP`: Prevents losing information at boundaries

4. **Retrieval Quality**
   - Pure semantic search has limitations
   - Hybrid approaches (semantic + keywords) work better
   - Filename boosting improves user experience

5. **LLM Prompting**
   - Clear instructions improve accuracy
   - Asking for source citations builds trust
   - Streaming responses improve UX

## 🙏 Acknowledgments

- **LangChain** for the document processing framework
- **Ollama** for local LLM infrastructure
- **FAISS** by Facebook AI for vector search
- **HuggingFace** for embedding models
- **Mistral AI** for the Mistral model

---

**Built with ❤️ while learning AI engineering**
