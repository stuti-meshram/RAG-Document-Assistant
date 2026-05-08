# ─────────────────────────────────────────────────
#  RAG Document Assistant — Dockerfile
#  Multi-stage build: lean production image
# ─────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for pdfplumber / faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

# Create data directory for FAISS index
RUN mkdir -p /app/data

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

# Streamlit port
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FAISS_INDEX_PATH=/app/data/faiss_index

ENTRYPOINT ["streamlit", "run", "app/main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
