"""
Vercel serverless entrypoint for Collabry AI Engine
Imports and exposes the FastAPI app from server/main.py
"""
from server.main import app

# Vercel will automatically use this 'app' variable
__all__ = ["app"]
