# Collabry AI Core Engine - Study Copilot

**Pedagogical AI Learning Assistant** - A modular AI backend powered by the Hugging Face Inference API, designed to help students learn effectively through proven educational strategies.

## 🚀 What's New: Cloud Hugging Face Router API

**January 2025 Update:** The AI engine uses the new Hugging Face Router API for cloud-based AI processing, providing access to state-of-the-art models through a unified OpenAI-compatible interface.

✅ **Cloud AI processing** - All LLM calls via Hugging Face Router API  
✅ **OpenAI-compatible** - Uses OpenAI client library  
✅ **High-quality models** - Access to GPT-OSS-120B and other models  
✅ **Intent classification via LLM**  
✅ **Zero local hardware requirements**  

📖 See the `ai-engine/DEPLOYMENT.md` for deployment and API key setup.

## Architecture

This is a **backend-only AI Core Engine** with:
- **Study Copilot Agent** - Pedagogical AI optimized for learning
- **Hugging Face Inference API** - Cloud LLM backend (no local models)
- **FastAPI REST API** with JWT authentication
- **Multi-user isolation** (see [MULTI_USER_ARCHITECTURE.md](MULTI_USER_ARCHITECTURE.md))
- LangChain-compatible agent orchestration
- RAG pipeline with FAISS + sentence-transformers (user-scoped document filtering)
- **MongoDB persistence** for conversation memory (REQUIRED, no fallback)
- Modular tool system for study platform features
- **Multiple sessions per user** (ChatGPT-style)
- **Background task processing** for document ingestion

## Study Copilot Features

### 🎓 Pedagogical Approach
The Study Copilot employs research-backed learning strategies:
- **Step-by-step explanations** - Breaks complex topics into digestible chunks
- **Examples & analogies** - Makes abstract concepts concrete and relatable
- **Clarifying questions** - Detects vague input and asks for specifics
- **No hallucination** - Only cites sources from retrieved documents or tools
- **Follow-up questions** - Encourages active recall and deeper thinking

📖 **See [STUDY_COPILOT.md](STUDY_COPILOT.md) for complete pedagogical documentation**

### 📚 Learning Capabilities
- **Q&A over documents** - Retrieves and synthesizes from uploaded materials
- **Summarization** - Creates study-focused summaries with key points
- **Concept extraction** - Identifies and explains core concepts with examples
- **Follow-up question generation** - Promotes active learning (3 levels: recall, apply, connect)

### 🚀 FastAPI Server
- **RESTful API** with OpenAPI documentation at `/docs`
- **JWT-based authentication** for all endpoints
- **Streaming & non-streaming** chat responses
- **Background tasks** for document embedding
- **Health check** endpoint for monitoring
- **CORS support** for frontend integration

### 🔒 Multi-User Isolation
- **Memory isolation**: Each user's conversations stored with `user_id` in MongoDB
- **Session management**: Multiple chat sessions per user (UUID-based)
- **RAG filtering**: User-specific + public documents only
- **JWT authentication**: User identity extracted from validated tokens
- **No cross-user data leakage**: Permission checks enforce isolation

📖 **See [MULTI_USER_ARCHITECTURE.md](MULTI_USER_ARCHITECTURE.md) for complete details**

### 📚 Study Platform Endpoints
- `POST /ai/chat` - Conversational AI with tool invocation
- `POST /ai/chat/stream` - Streaming chat with SSE
- `POST /ai/upload` - Document upload for RAG (background processing)
- `POST /ai/summarize` - Text summarization
- `POST /ai/qa` - Question answering with RAG
- `POST /ai/mindmap` - Mind map generation
- `GET /ai/sessions` - List user sessions
- `POST /ai/sessions` - Create new session

### Active Tools
- **web_search**: Hybrid web search (Serper API + DuckDuckGo fallback)
- **web_scrape**: Full content extraction from URLs
- **read_file** / **write_file**: Local file operations
- **doc_generator**: Create Word documents (.docx) for study notes
- **ppt_generator**: Generate PowerPoint presentations (.pptx)
- **ocr_read**: Extract text from images (Tesseract)
- **image_gen**: Generate images (requires Stable Diffusion WebUI)

### Legacy Components (Moved to `legacy_tools/`)
- CLI interface (`main_cli.py`) - for local testing only
- Browser control
- System automation
- Task scheduler

### Quick Start

### Prerequisites

1. **Get Hugging Face API Token** (REQUIRED - for LLM + embeddings):
   - Visit: https://huggingface.co/settings/tokens
   - Sign in or create an account
   - Click "New token" → choose `read` scope (or `inference` where required)
   - Copy the generated token

2. **Install MongoDB** (REQUIRED for memory persistence):
   ```powershell
   # Option 1: Docker (recommended for development)
   docker run -d -p 27017:27017 --name collabry-mongo mongo:latest
   
   # Option 2: Windows installer
   # Download from https://www.mongodb.com/try/download/community
   ```

3. **Create virtual environment** and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (REQUIRED for security):
   ```powershell
   # Copy the example environment file
   copy .env.example .env
   
   # Edit .env with your values (use notepad, VSCode, etc.)
   notepad .env
   ```
   
   **Minimum required variables:**
   - `HUGGINGFACE_API_KEY` - Hugging Face API token (from step 1)
   - `MONGO_URI` - MongoDB connection string
   - `JWT_SECRET_KEY` - Secret for JWT validation (CHANGE IN PRODUCTION!)
   
   **Optional but recommended:**
   - `HUGGINGFACE_MODEL` - Model to use (default: configured in `config.py`)
   - `SERPER_API_KEY` - Enhanced web search (get free key at https://serper.dev)

### Running the FastAPI Server

Start the production server:
```powershell
python run_server.py
```

Development mode with auto-reload:
```powershell
python run_server.py --reload
```

Custom host/port:
```powershell
python run_server.py --host 0.0.0.0 --port 8080
```

Access the API:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/

### Production Deployment

For hosting platforms (Render, Heroku, Railway, etc.), use the simplified production script:

```bash
python start_production.py
```

This automatically:
- Uses the `PORT` environment variable set by hosting platforms
- Binds to `0.0.0.0` for external access
- Optimized for production (no development features)

**Hosting Configuration:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python start_production.py`

📖 **See [DEPLOYMENT.md](DEPLOYMENT.md) for complete hosting setup**

### Testing

**FastAPI Integration Tests** (requires server running):
```powershell
# Terminal 1: Start server
python run_server.py

# Terminal 2: Run tests
python test_fastapi_server.py
```

**Component Tests:**
```powershell
# Tool loading test
python test_tools_loading.py

# Memory system test (MongoDB required)
python test_memory_mongodb.py

# Agent execution test
python test_agent_execution.py
```

**Multi-user isolation tests:**
```powershell
python scripts/test_multi_user_isolation.py
```

### Local CLI Testing (Legacy)

Test with different users/sessions:
```powershell
# User Alice, work session
python legacy_tools/main_cli.py --user alice --session work

# User Alice, personal session (different terminal)
python legacy_tools/main_cli.py --user alice --session personal

# User Bob, default session (different terminal)
python legacy_tools/main_cli.py --user bob --session default
```

CLI commands: `sessions`, `new session`, `switch <session_id>`, `exit`

## Configuration

Edit [`config.py`](config.py) to customize:
- LLM model (`llm_model`)
- Ollama host (`ollama_host`)
- **MongoDB settings** (`mongo_uri`, `mongo_db`, `memory_collection`)
- Embedding model (`embedding_model`)
- RAG retrieval settings (`retrieval_top_k`)

Environment variables override config defaults:
```powershell
# Hugging Face configuration (set in .env or environment)
$env:HUGGINGFACE_API_KEY = "<your-token>"
$env:HUGGINGFACE_MODEL = "gpt2"  # or another HF-hosted model
$env:LLM_TIMEOUT = "60"
```
# Legacy ENV variables (still supported)
$env:OLLAMA_HOST = "http://localhost:11434"
$env:COLLABRY_LLM_MODEL = "mistral"
$env:COLLABRY_TEMPERATURE = "0.3"

# MongoDB Configuration
$env:MONGO_URI = "mongodb://localhost:27017"
$env:MONGO_DB = "collabry"
```

## Roadmap

- [x] Backend-only architecture (CLI moved to legacy)
- [x] **MongoDB persistence (no fallback)** ✓
- [x] **JWT-based multi-user isolation** ✓
- [x] **Multiple sessions per user** ✓
- [x] **User-scoped RAG retrieval** ✓
- [ ] FastAPI REST API layer
- [ ] Role-based access control (admin/teacher/student)
- [ ] Production deployment configuration
- [ ] Rate limiting per user
- [ ] Document sharing between users

## Project Structure

```
ai-engine/
├── core/                  # Core AI components
│   ├── agent.py          # LangChain agent orchestration
│   ├── local_llm.py      # Ollama LLM wrapper
│   ├── memory.py         # Conversation memory (LangGraph checkpointing)
│   ├── embeddings.py     # sentence-transformers embeddings
│   ├── rag_retriever.py  # FAISS-based RAG
│   ├── intent_classifier.py  # TF-IDF intent classification
│   └── nlp.py            # NLP preprocessing
├── tools/                 # Modular tool system
├── legacy_tools/          # Archived CLI/system tools
├── models/                # Pretrained models
├── memory/                # Memory persistence
├── documents/             # RAG document store
└── config.py             # Configuration
