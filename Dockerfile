# ==============================================================================
# Stage 1: Build & Dependency Wheel Compilation
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install compilation toolchain and uv package manager
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

# Create isolated production virtual environment
RUN uv venv /app/.venv --python python3.11
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency specifications and pre-compile wheels
COPY requirements.txt pyproject.toml /app/
RUN uv pip install --no-cache -r requirements.txt

# Install the application package
COPY src/ /app/src/
RUN uv pip install -e . --no-deps

# ==============================================================================
# Stage 2: Hardened Minimal Runtime Container
# ==============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install runtime OpenMP shared library required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system group and user (UID 10001, no login shell)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -M appuser

WORKDIR /app

# Initialize directory structure with explicit ownership
RUN mkdir -p /app/data/raw /app/data/processed /app/data/audit /app/artifacts/models /app/mlruns && \
    chown -R appuser:appgroup /app

# Copy virtualenv and runtime assets from builder with strict ownership
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup frontend/ /app/frontend/
COPY --chown=appuser:appgroup pyproject.toml /app/pyproject.toml
COPY --chown=appuser:appgroup data/raw/dataset.csv /app/data/raw/dataset.csv

# Enforce non-root execution
USER 10001:10001

# Expose API port (8000) and Streamlit dashboard port (8501)
EXPOSE 8000 8501

# Container healthcheck using standard library urllib (no curl/wget needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default entrypoint starts the FastAPI serving daemon
CMD ["uvicorn", "coffee_quality.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
