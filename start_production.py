#!/usr/bin/env python3
"""
Production startup script for hosting platforms (Render, Heroku, Railway, etc.)

This script provides a simple, clean startup for production deployments.
No argument parsing, no process management - just starts the FastAPI server.

Usage:
    python start_production.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the server
import uvicorn

if __name__ == "__main__":
    # Get port from environment (required for hosting platforms)
    port = int(os.environ.get("PORT", 8000))

    print("🚀 Starting Collabry AI Core - Production Mode")
    print(f"📡 Port: {port}")
    print(f"🌐 Host: 0.0.0.0")

    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )