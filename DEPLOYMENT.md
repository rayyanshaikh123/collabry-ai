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
- `HF_TOKEN` - **REQUIRED** for Hugging Face Router API
- `HUGGINGFACE_MODEL` - Model name (default: openai/gpt-oss-120b:groq)

## Cloud AI Processing

The system uses **Hugging Face Router API** for all AI processing:
- **No local models** - All processing happens in the cloud
- **API key required** - Get your token from https://huggingface.co/settings/tokens
- **Supported models** - Access to high-quality open-source models via router
- **OpenAI-compatible** - Uses OpenAI client library for API calls
- **Rate limits apply** - Free tier has usage limits

### Getting Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Add it to your `.env` file as `HF_TOKEN=your_token_here`

### Port binding issues
- Use `python start_production.py` instead of `python run_server.py`
- The script automatically detects the `PORT` environment variable
- No manual port configuration needed