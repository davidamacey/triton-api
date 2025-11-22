# =============================================================================
# Unified Dockerfile for Triton YOLO Inference System
# =============================================================================
# This Dockerfile builds the unified yolo-api service that handles ALL tracks:
#   - Track A: PyTorch baseline (loaded at startup)
#   - Track B: TensorRT + CPU NMS (via Triton gRPC)
#   - Track C: TensorRT + GPU NMS (via Triton gRPC)
#   - Track D: DALI + TensorRT (via Triton gRPC ensembles)
#
# All tracks are accessible on port 9600 via FastAPI
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

# Install system dependencies and build Python packages
# Build dependencies (gcc, g++) are removed after pip install to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime dependencies
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    jq \
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

# Copy the application code
COPY src/ ./src/

# Copy scripts for management
COPY scripts/ ./scripts/

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/scripts/*.sh

USER appuser

# =============================================================================
# Default Configuration
# =============================================================================
# The CMD is overridden in docker-compose.yml for each service
# Default: All tracks enabled (useful for development/testing)
# =============================================================================

ENV SERVICE_MODE=all \
    SERVICE_PORT=9600

EXPOSE ${SERVICE_PORT}

# Health check (port will be overridden by docker-compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests, os; requests.get(f'http://localhost:{os.getenv(\"SERVICE_PORT\", \"9600\")}/health', timeout=5).raise_for_status()" || exit 1

# Default CMD - runs unified service with all tracks
# Override this in docker-compose.yml for specific configurations
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "9600", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--backlog", "2048", \
     "--timeout-keep-alive", "5", \
     "--access-log", \
     "--log-level", "info"]
