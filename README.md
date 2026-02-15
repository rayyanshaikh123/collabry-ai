# Collabry AI Engine

**Production-grade AI backend with LangChain agents, RAG document retrieval, and real-time streaming.**

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      AI-ENGINE (FastAPI Backend)                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │          server/main.py (FastAPI Application)                 │   │
│  │  • JWT Authentication Middleware (server/deps.py)            │   │
│  │  • Redis Rate Limiting                                       │   │
│  │  • CORS Configuration                                        │   │
│  └────┬─────────────────────────────────────────────────────────┘   │
│       │                                                              │
│  ┌────▼──────────────────── API ROUTES ─────────────────────────┐   │
│  │                                                               │   │
│  │  📝 /ai/chat         ──► Main chat endpoint (streaming)       │   │
│  │  📝 /ai/sessions     ──► Session management & history         │   │
│  │  📝 /ai/qa           ──► Question answering                  │   │
│  │  📝 /ai/summarize    ──► Document summarization              │   │
│  │  📝 /ai/mindmap      ──► Mind map generation                 │   │
│  │  📤 /ai/upload       ──► Document ingestion (RAG)            │   │
│  │  📊 /ai/usage        ──► Usage analytics                     │   │
│  │  📚 /ai/studyplan    ──► Study plan generation               │   │
│  │  ❤️  /health         ──► Health check endpoint                │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────── CORE: AGENT LAYER ───────────────────────┐    │
│  │                                                              │    │
│  │  core/agent.py (LangChain-based AI Agent)                   │    │
│  │  ✅ Native tool calling (function calling)                   │    │
│  │  ✅ Streaming SSE support                                    │    │
│  │  ✅ Provider-agnostic (OpenAI-compatible APIs)               │    │
│  │  ✅ Automatic artifact detection & formatting                │    │
│  │  ✅ RAG integration for context-aware responses              │    │
│  │                                                              │    │
│  │  Dependencies:                                               │    │
│  │  ├──► core/llm.py (Unified LLM client)                      │    │
│  │  ├──► core/embeddings.py (Unified embeddings)               │    │
│  │  ├──► core/conversation.py (MongoDB chat history)           │    │
│  │  ├──► core/artifact_templates.py (Quiz/Mindmap templates)   │    │
│  │  ├──► core/rag_retriever.py (Document retrieval)            │    │
│  │  └──► tools/* (LangChain tools)                             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────────── RAG & RETRIEVAL ──────────────────────────┐    │
│  │  core/rag_retriever.py (FAISS-based vector search)          │    │
│  │  • User-isolated document storage                           │    │
│  │  • Metadata filtering (user_id, session_id, notebook_id)    │    │
│  │  • HuggingFace embeddings                                   │    │
│  │  • MongoDB GridFS backup                                    │    │
│  │  • Semantic search with relevance scoring                   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────── TOOLS LIBRARY ──────────────────────────┐    │
│  │  tools/generate_quiz.py          - Quiz generation          │    │
│  │  tools/generate_flashcards.py    - Flashcard creation       │    │
│  │  tools/mindmap_generator.py      - Mind map generation      │    │
│  │  tools/generate_infographic.py   - Infographic creation     │    │
│  │  tools/summarize.py              - Text summarization       │    │
│  │  tools/search_sources.py         - RAG source search        │    │
│  │  tools/search_web.py             - Web search               │    │
│  │  tools/generate_study_plan.py    - Study plan creation      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────────── STORAGE & DATA ───────────────────────────┐    │
│  │  MongoDB:                                                   │    │
│  │    • Sessions (chat sessions)                               │    │
│  │    • Messages (chat history with user attribution)          │    │
│  │    • Conversations (legacy compatibility)                   │    │
│  │    • Usage tracking & analytics                             │    │
│  │    • Documents metadata (GridFS)                            │    │
│  │                                                              │    │
│  │  Redis:                                                      │    │
│  │    • Rate limiting                                          │    │
│  │    • Session caching                                        │    │
│  │    • Real-time state management                             │    │
│  │                                                              │    │
│  │  Vector Store (FAISS - local filesystem):                   │    │
│  │    • User documents (RAG embeddings)                        │    │
│  │    • HuggingFace embeddings                                 │    │
│  │    • Per-user/notebook isolation via metadata               │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES                              │
│  • OpenAI / DeepSeek / OpenRouter (LLM providers)                    │
│  • MongoDB Atlas (Database)                                          │
│  • Redis Cloud/Upstash (Caching & Rate Limiting)                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Features

### Core Capabilities
- **LangChain-based Agent**: Native tool calling with function calling API
- **Multi-user Isolated RAG**: User-specific document retrieval with metadata filtering
- **Real-time Streaming**: Server-Sent Events (SSE) for token-by-token responses
- **JWT Authentication**: Secure user isolation across all endpoints
- **Usage Tracking**: Request/token analytics with MongoDB persistence
- **Rate Limiting**: Redis-based rate limiting per user/IP
- **Session Management**: Persistent chat history with user attribution

### AI Tools & Artifacts
- **Study Tools**: Quiz generation, flashcards, mind maps, study plans
- **Document Tools**: Summarization, Q&A, concept extraction
- **Search Tools**: Web search, RAG-based source search
- **Generation Tools**: Infographics, structured content

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- MongoDB (local or Atlas)
- Redis (local or cloud)
- OpenAI API key (or compatible provider)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/collabry.git
cd collabry/ai-engine
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

**Required Environment Variables:**

```env
# ==========================================
# MongoDB Configuration
# ==========================================
MONGO_URI=mongodb://localhost:27017/collabry
# For MongoDB Atlas:
# MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/collabry?retryWrites=true&w=majority

# ==========================================
# JWT Authentication
# ==========================================
JWT_SECRET=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# ==========================================
# LLM Provider (OpenAI-compatible)
# ==========================================
OPENAI_API_KEY=your-openai-key
# Or use OpenRouter or DeepSeek:
# OPENAI_API_KEY=your-openrouter-key
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# Or use local Ollama:
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama

# LLM Model Configuration
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000

# ==========================================
# Embeddings
# ==========================================
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# For custom embeddings:
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-small

# ==========================================
# Redis Configuration
# ==========================================
REDIS_URL=redis://localhost:6379/0
# For Redis Cloud/Upstash:
# REDIS_URL=redis://default:password@redis-xxxx.cloud.redislabs.com:12345

# ==========================================
# CORS Origins
# ==========================================
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:5000
# For production, add your domain:
# CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# ==========================================
# Server Configuration
# ==========================================
PORT=8000
HOST=0.0.0.0
WORKERS=4
LOG_LEVEL=INFO

# ==========================================
# Rate Limiting
# ==========================================
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# ==========================================
# Optional: External Search (for web search tool)
# ==========================================
# SERPAPI_API_KEY=your-serpapi-key
# BRAVE_API_KEY=your-brave-search-key
```

### 5. Start the Server

**Development Mode:**
```bash
python run_server.py
```

**Production Mode:**
```bash
# Using uvicorn directly
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 4

# Using gunicorn (recommended for production)
gunicorn server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Server will be available at: **http://localhost:8000**
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# From project root
docker-compose up -d

# Or build specific service
docker-compose up -d ai-engine
```

The Docker Compose configuration includes:
- AI Engine service
- MongoDB service
- Redis service
- Persistent volumes for data
- Automatic health checks
- Network isolation

### Standalone Docker

```bash
# Build image
docker build -t collabry-ai-engine .

# Run container
docker run -d \
  --name ai-engine \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/documents:/app/documents \
  -v $(pwd)/memory:/app/memory \
  collabry-ai-engine
```

---

## ☁️ Cloud Deployment

### Deploy to Render

1. **Create New Web Service**
   - Connect your GitHub repository
   - Select `ai-engine` as root directory
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn server.main:app --host 0.0.0.0 --port $PORT --workers 4`

2. **Environment Variables**
   - Add all variables from `.env.example`
   - Use Render's Redis addon or external Redis URL
   - Use MongoDB Atlas for database

3. **Scaling**
   - Render auto-scales based on traffic
   - Recommended: Standard instance or higher for production

### Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Add MongoDB addon (or use Atlas)
heroku addons:create mongolab:sandbox

# Add Redis addon
heroku addons:create heroku-redis:mini

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key
heroku config:set JWT_SECRET=your-secret
heroku config:set LLM_MODEL=gpt-4o-mini

# Deploy
git subtree push --prefix ai-engine heroku main

# Or if using buildpack
git push heroku main
```

### Deploy to Railway

1. **Create New Project**
   - Connect GitHub repository
   - Railway auto-detects Dockerfile
   - Select `ai-engine` directory

2. **Add Environment Variables**
   - Copy from `.env.example`
   - Use Railway's PostgreSQL or MongoDB addon
   - Use Railway's Redis addon

3. **Deploy**
   - Railway automatically deploys on push
   - Custom domain available in project settings

### Deploy to DigitalOcean App Platform

1. **Create New App**
   - Connect repository
   - Select `ai-engine` directory
   - Choose Python runtime

2. **Configure Build**
   - Build command: `pip install -r requirements.txt`
   - Run command: `gunicorn server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080`

3. **Add Services**
   - MongoDB (DigitalOcean Managed Database)
   - Redis (DigitalOcean Managed Database)

---

## 🔧 Development

### Project Structure
```
ai-engine/
├── api/                    # Vercel serverless entrypoint
├── core/                   # Core business logic
│   ├── agent.py           # LangChain agent (main AI logic)
│   ├── llm.py             # Unified LLM client
│   ├── embeddings.py      # Unified embeddings
│   ├── rag_retriever.py   # RAG document retrieval
│   ├── conversation.py    # MongoDB chat history
│   ├── mongo_store.py     # MongoDB operations
│   ├── redis_client.py    # Redis client wrapper
│   ├── session_state.py   # Session state management
│   ├── artifact_prompts.py # AI prompts for artifacts
│   ├── artifact_templates.py # Artifact templates
│   └── schemas.py         # Pydantic models
├── server/                 # FastAPI application
│   ├── main.py            # App initialization & routing
│   ├── deps.py            # JWT authentication
│   ├── middleware.py      # Custom middleware
│   ├── limit_middleware.py # Rate limiting
│   ├── routes/            # API endpoints
│   │   ├── sessions.py    # Session management
│   │   ├── chat.py        # Chat endpoints
│   │   ├── qa.py          # Q&A endpoints
│   │   ├── summarize.py   # Summarization
│   │   ├── mindmap.py     # Mind map generation
│   │   ├── ingest.py      # Document upload
│   │   └── usage.py       # Usage tracking
│   └── schemas.py         # API schemas
├── tools/                  # LangChain tools
│   ├── generate_quiz.py
│   ├── generate_flashcards.py
│   ├── mindmap_generator.py
│   ├── generate_infographic.py
│   ├── search_sources.py
│   ├── search_web.py
│   ├── summarize.py
│   └── generate_study_plan.py
├── data/                   # Static data
│   ├── curricula/         # Learning curricula
│   └── eval/              # Evaluation datasets
├── documents/              # User-uploaded documents (RAG)
├── memory/                 # FAISS indexes storage
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── run_server.py          # Development server launcher
├── Dockerfile             # Docker image definition
└── .env.example           # Environment template
```

### Code Quality

```bash
# Format code
black core/ server/ tools/

# Lint code
flake8 core/ server/ tools/ --max-line-length=120

# Type checking
mypy core/ server/ --ignore-missing-imports

# Sort imports
isort core/ server/ tools/
```

### Running in Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG  # Linux/Mac
set LOG_LEVEL=DEBUG     # Windows

python run_server.py
```

---

## 📊 API Usage Examples

### 1. Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-13T10:30:00Z",
  "version": "1.0.0"
}
```

### 2. Create Session
```bash
curl -X POST http://localhost:8000/ai/sessions \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Biology Study Session"
  }'
```

### 3. Chat with Streaming
```bash
curl -X POST http://localhost:8000/ai/sessions/SESSION_ID/chat/stream \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain photosynthesis",
    "notebook_id": "biology-101",
    "source_ids": ["source1", "source2"]
  }'
```

### 4. Upload Document (RAG)
```bash
curl -X POST http://localhost:8000/ai/upload \
  -H "Authorization: Bearer YOUR_JWT" \
  -F "file=@notes.pdf" \
  -F "session_id=session-123" \
  -F "notebook_id=biology-101"
```

### 5. Get Session Messages
```bash
curl http://localhost:8000/ai/sessions/SESSION_ID/messages \
  -H "Authorization: Bearer YOUR_JWT"
```

### 6. Generate Quiz
```bash
curl -X POST http://localhost:8000/ai/chat \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Generate a 5-question quiz on photosynthesis",
    "session_id": "session-123",
    "notebook_id": "biology-101"
  }'
```

The agent will automatically detect the quiz request and use the quiz generation tool.

---

## 🔐 Authentication

All endpoints (except `/health`) require JWT authentication:

```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**JWT Payload Structure:**
```json
{
  "sub": "user-id-123",
  "email": "user@example.com",
  "exp": 1234567890,
  "iat": 1234567800
}
```

The JWT must be generated and signed by your authentication service using the shared `JWT_SECRET`.

---

## 📈 Monitoring & Logging

### Logging Configuration

The application uses Python's `logging` module with configurable levels:

```python
# Set log level via environment
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Usage Tracking

Usage statistics are automatically tracked in MongoDB:

```python
# Collection: usage_stats
{
  "user_id": "user-123",
  "endpoint": "/ai/chat",
  "timestamp": "2026-02-13T10:30:00Z",
  "tokens_used": 150,
  "response_time_ms": 1200,
  "model": "gpt-4o-mini"
}
```

### Health Monitoring

```bash
# Check health endpoint
curl http://localhost:8000/health

# Monitor logs in real-time
tail -f logs/app.log

# Docker logs
docker logs -f ai-engine
```

---
## 🐛 Troubleshooting

### MongoDB Connection Issues
```bash
# Check MongoDB is running locally
mongosh --eval "db.runCommand({ ping: 1 })"

# Test Atlas connection
mongosh "mongodb+srv://user:pass@cluster.mongodb.net/collabry"

# Verify connection string in .env
MONGO_URI=mongodb://localhost:27017/collabry
```

### Redis Connection Issues
```bash
# Check Redis is running locally
redis-cli ping

# Test Redis connection
redis-cli -u redis://localhost:6379/0

# For cloud Redis, check connection string
REDIS_URL=redis://default:password@host:port/0
```

### LLM Provider Issues
```bash
# Test OpenAI connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"

# Test OpenRouter connection
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"

# Test Ollama connection (local)
curl http://localhost:11434/api/tags
```

### CORS Issues
```bash
# Add frontend origin to .env
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# Restart server after changing CORS_ORIGINS
# Check browser console for CORS errors
```

### Rate Limiting Issues
```bash
# Disable rate limiting temporarily for debugging
RATE_LIMIT_ENABLED=false

# Check Redis for rate limit keys
redis-cli keys "rate_limit:*"

# Clear rate limits for user
redis-cli del "rate_limit:user:USER_ID"
```

### Document Upload/RAG Issues
```bash
# Check documents directory permissions
ls -la documents/

# Verify FAISS index
ls -la memory/faiss_index/

# Clear FAISS index (will rebuild on next upload)
rm -rf memory/faiss_index/*

# Check MongoDB GridFS for document metadata
mongosh
> use collabry
> db.fs.files.find()
```

### Common Errors

**Error: "OPENAI_API_KEY not set"**
```bash
# Ensure .env file exists and contains:
OPENAI_API_KEY=your-key-here

# Reload environment
source .env  # Linux/Mac
# Or restart application
```

**Error: "Failed to connect to MongoDB"**
```bash
# Check if MongoDB is running
# For local: brew services start mongodb-community
# For Atlas: check network access whitelist
```

**Error: "Redis connection refused"**
```bash
# Start Redis locally
# Windows: redis-server
# Linux: sudo service redis-server start
# Mac: brew services start redis
```

---

## 🔒 Security Best Practices

### Production Checklist

- [ ] Change `JWT_SECRET` to a strong random value
- [ ] Use HTTPS for all API endpoints
- [ ] Enable rate limiting (`RATE_LIMIT_ENABLED=true`)
- [ ] Whitelist only necessary CORS origins
- [ ] Use environment variables, never commit `.env`
- [ ] Enable MongoDB authentication
- [ ] Use Redis password authentication
- [ ] Set up firewall rules for MongoDB/Redis
- [ ] Implement request size limits
- [ ] Monitor usage and set quota limits
- [ ] Enable logging for security audits
- [ ] Regularly update dependencies

### Environment Security
```bash
# Never commit .env files
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore

# Use strong JWT secret (32+ characters)
JWT_SECRET=$(openssl rand -hex 32)

# Restrict file permissions
chmod 600 .env
```

---

## 📚 Additional Resources

- **LangChain Documentation**: https://python.langchain.com/docs
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **MongoDB Documentation**: https://www.mongodb.com/docs
- **Redis Documentation**: https://redis.io/docs
- **Docker Documentation**: https://docs.docker.com
- **Deployment Guides**: See `/docs` folder

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Follow PEP 8 style guide
   - Add tests for new features
   - Update documentation
4. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Code Style

```bash
# Format code before committing
black .
isort .
flake8 . --max-line-length=120
```

---

## 📝 License

MIT License - see [LICENSE](../LICENSE) file for details

---

## 📞 Support

For issues, questions, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/your-org/collabry/issues)
- **Documentation**: See `/docs` folder
- **Email**: support@collabry.com

---

## 🗺️ Roadmap

### Completed ✅
- LangChain-based agent with function calling
- Multi-user RAG with FAISS
- Session management with MongoDB
- Real-time streaming with SSE
- Rate limiting with Redis
- Docker deployment support
- Comprehensive API documentation

### In Progress 🚧
- Enhanced RAG with multiple vector database support
- Advanced analytics dashboard
- Batch processing for large documents
- Multi-language support

### Planned 📋
- WebSocket support for real-time collaboration
- Voice interaction capabilities
- Advanced caching strategies
- Plugin system for custom tools
- GraphQL API option

---

**Built with ❤️ by the Collabry Team**

*Last updated: February 2026*
