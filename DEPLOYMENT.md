# Production Deployment

## For Hosting Platforms (Render, Heroku, Railway, etc.)

Use the production startup script for clean deployments:

```bash
python start_production.py
```

This script:
- Automatically uses the `PORT` environment variable
- Binds to `0.0.0.0` for external access
- No argument parsing or complex logic
- Optimized for production hosting

## For Development

Use the full-featured development script:

```bash
python run_server.py
```

## Environment Variables

Make sure these are set in your hosting platform:

- `PORT` - The port to bind to (automatically detected)
- `MONGODB_URI` - MongoDB connection string
- `JWT_SECRET_KEY` - For authentication
- `HUGGINGFACE_API_KEY` - For embeddings
- `GEMINI_API_KEY` - For LLM (optional, falls back to Ollama)

## Render.com Configuration

In your Render service settings:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python start_production.py`

## Troubleshooting

### "Connection reset by peer" during pip install
If you see network errors during package installation:
1. The `sentence-transformers` dependency has been removed to avoid large PyTorch downloads
2. All embeddings now use cloud Hugging Face API (no local ML models)
3. Re-deploy should be much faster and more reliable

### Port binding issues
- Use `python start_production.py` instead of `python run_server.py`
- The script automatically detects the `PORT` environment variable
- No manual port configuration needed