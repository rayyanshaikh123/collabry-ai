"""
FastAPI Server Startup Script

Starts the Collabry AI Core FastAPI server.

Usage:
    python run_server.py                    # Development mode
    python run_server.py --production       # Production mode (for hosting)
    python run_server.py --host 0.0.0.0 --port 8000
    python run_server.py --reload
"""
import uvicorn
import argparse
import os
from pathlib import Path
import sys
import platform
import subprocess
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def kill_process_on_port(port):
    """
    Kill any process using the specified port.
    
    Args:
        port: Port number to free up
    """
    system = platform.system()
    
    try:
        if system == "Windows":
            # Find process using the port
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        try:
                            # Kill the process
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                check=True
                            )
                            print(f"✓ Stopped process {pid} using port {port}")
                            time.sleep(1)  # Wait for port to be released
                        except subprocess.CalledProcessError:
                            pass
        else:
            # Unix-like systems (Linux, macOS)
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        print(f"✓ Stopped process {pid} using port {port}")
                    except subprocess.CalledProcessError:
                        pass
                time.sleep(1)  # Wait for port to be released
                        
    except Exception as e:
        print(f"⚠ Could not check/kill processes on port {port}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Collabry AI Core FastAPI server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to bind (default: 8000 or PORT env var)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Production mode - simplified startup for hosting platforms"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Production mode for hosting platforms (Render, Heroku, etc.)
    if args.production or os.environ.get("RENDER") or os.environ.get("DYNO"):
        print("🏭 Production mode detected - starting simplified server for hosting platform")
        
        # Get port from environment (required for hosting)
        port = int(os.environ.get("PORT", 8000))
        print(f"📡 Hosting platform detected, using PORT: {port}")
        
        # Simple startup for hosting - no process killing, no complex logic
        uvicorn.run(
            "server.main:app",
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        sys.exit(0)
    
    # Development mode with full features
    # Determine port (hosting platforms like Render/Heroku set PORT env var)
    port = args.port
    if os.environ.get("PORT"):
        try:
            port = int(os.environ.get("PORT"))
            print(f"📡 Using PORT from environment: {port}")
        except ValueError:
            print(f"⚠️ Invalid PORT environment variable: {os.environ.get('PORT')}, using default {args.port}")
            port = args.port
    else:
        print(f"📡 Using port: {port}")
    
    # Validate port range
    if not (1 <= port <= 65535):
        print(f"❌ Invalid port number: {port}. Port must be between 1 and 65535.")
        sys.exit(1)
    
    # Kill any existing process on the port
    print(f"🔍 Checking port {port}...")
    kill_process_on_port(port)
    
    print("=" * 60)
    print("🚀 Starting Collabry AI Core FastAPI Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {port}")
    print(f"Reload: {args.reload}")
    print(f"Docs: http://{args.host}:{port}/docs")
    print(f"Health: http://{args.host}:{port}/health")
    print("=" * 60)
    
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level=args.log_level
    )
