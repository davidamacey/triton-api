# =============================================================================
# Unified Dockerfile for Triton YOLO Inference System
# =============================================================================
# This Dockerfile builds the unified yolo-api service that handles ALL tracks:
#   - Track A: PyTorch baseline (loaded at startup)
#   - Track B: TensorRT + CPU NMS (via Triton gRPC)
#   - Track C: TensorRT + GPU NMS (via Triton gRPC)
#   - Track D: DALI + TensorRT (via Triton gRPC ensembles)
#   - Track E: MobileCLIP Visual Search (via Triton + OpenSearch)
#
# All tracks are accessible on port 8000 via FastAPI
#
# VOLUME MOUNTS REQUIRED:
#   - ./pytorch_models:/app/pytorch_models  (MobileCLIP checkpoints)
#   - ./reference_repos:/app/reference_repos  (ml-mobileclip, open_clip repos)
# =============================================================================

FROM python:3.12-slim-trixie

# Set environment variables for performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# =============================================================================
# Stage 1: Install system dependencies and Python packages
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime dependencies
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    jq \
    git \
    # Build dependencies (temporary, for compiling C extensions)
    gcc \
    g++ \
    cmake \
    make \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        -r requirements.txt \
    && rm -rf /root/.cache/pip/* \
    # Remove build dependencies to reduce image size
    && apt-get purge -y --auto-remove gcc g++ cmake make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# Stage 2: Copy application code
# =============================================================================
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user for security best practices
# Note: reference_repos and pytorch_models are mounted from host
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/scripts/*.sh 2>/dev/null || true && \
    chmod +x /app/scripts/**/*.sh 2>/dev/null || true

USER appuser

# =============================================================================
# Default Configuration
# =============================================================================
ENV SERVICE_MODE=all \
    SERVICE_PORT=8000

EXPOSE ${SERVICE_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests, os; requests.get(f'http://localhost:{os.getenv(\"SERVICE_PORT\", \"8000\")}/health', timeout=5).raise_for_status()" || exit 1

# Default CMD - runs unified service with all tracks
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--backlog", "2048", \
     "--timeout-keep-alive", "5", \
     "--access-log", \
     "--log-level", "info"]
