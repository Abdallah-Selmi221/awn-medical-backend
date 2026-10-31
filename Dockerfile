# ==============================================================================
# AWN MEDICAL AI & LITERATURE RAG PLATFORM - PRODUCTION DOCKERFILE
# Stage: Lightweight, Secure, CPU-optimized Python 3.11 Runtime
# Compatible with: Hugging Face Spaces (Docker), Google Cloud Run, Render
# ==============================================================================

FROM python:3.11-slim AS runtime

# Security & Python optimization flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    HOST=0.0.0.0

# Working directory
WORKDIR /app

# Install minimal OS dependencies for health check probing and security certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first for layer caching and minimal image size (~700MB instead of ~3.5GB)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install minimal server requirements (FastAPI, Uvicorn, Gemini SDK, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy models directory (containing PyTorch multiclass models)
COPY models /app/models

# Copy server application code
COPY main.py /app/main.py

# Create a non-privileged user (UID 1000 required for Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port (7860 default for Hugging Face Spaces, dynamically overridden by $PORT on other hosts)
EXPOSE 7860

# Docker Healthcheck targeting the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Production entrypoint supporting dynamic $PORT injection
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
