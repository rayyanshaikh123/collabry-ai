# ================================
# AI-Engine Dockerfile (FastAPI + Python)
# Production-optimized with FAISS support
# ================================

# --- Build Stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt ./

# Install Python packages to /install directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Production Stage ---
FROM python:3.11-slim AS runner

WORKDIR /app

# Create non-root user
RUN groupadd -g 1001 python && \
    useradd -r -u 1001 -g python python

# Install runtime dependencies (libgomp required for FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=python:python . ./

# Create directories for persistent data
RUN mkdir -p /app/data/faiss_index \
             /app/memory \
             /app/data/curricula \
             /app/data/training \
    && chown -R python:python /app/data /app/memory

# Switch to non-root user
USER python

EXPOSE 8000

# Python optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Healthcheck (longer timeout for FAISS loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Use uvicorn directly (not run_server.py which has venv activation logic)
# Single worker to avoid FAISS index duplication in memory
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
