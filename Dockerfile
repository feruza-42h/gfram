# GFRAM Docker Image
# Multi-stage build for smaller final image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.10-slim

# Labels
LABEL maintainer="Ortiqova F.S. <feruzaortiqova42@gmail.com>"
LABEL version="3.0.0"
LABEL description="GFRAM - Geometric Face Recognition and Matching"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GFRAM_CACHE_DIR=/app/cache

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install Python packages
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY gfram/ ./gfram/
COPY setup.py pyproject.toml README.md ./

# Install gfram
RUN pip install --no-cache-dir -e .

# Create cache directory
RUN mkdir -p /app/cache

# Create non-root user
RUN useradd --create-home --shell /bin/bash gfram && \
    chown -R gfram:gfram /app
USER gfram

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import gfram; print(gfram.__version__)"

# Default command
CMD ["python", "-c", "import gfram; print(f'GFRAM v{gfram.__version__} ready!')"]
