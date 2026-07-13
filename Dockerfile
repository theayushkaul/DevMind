# =============================================================================
# DevMind — Dockerfile (Railway webhook receiver)
# =============================================================================
# This image runs the FastAPI webhook receiver on Railway.
# The Lambda processor is deployed separately via AWS (not Docker).
#
# Build strategy: multi-stage
#   Stage 1 (builder): install all deps including build tools
#   Stage 2 (runtime): copy only what's needed — no build tools in prod image
#
# Why multi-stage?
#   Some Python packages (cryptography, asyncpg) require C compilation during
#   install. The compiler (gcc, libssl-dev) adds ~300MB to the image. With
#   multi-stage builds, we compile in the builder stage and copy only the
#   resulting .so files and pure-Python packages into the slim runtime image.
#   Final image is ~180MB instead of ~480MB.
#
# Why python:3.13-slim-bookworm (not alpine)?
#   Alpine uses musl libc, which breaks binary Python wheels for cryptography
#   and asyncpg — they'd need to be compiled from source, making builds slow
#   and fragile. Slim Debian (bookworm) uses glibc, which all pre-built wheels
#   target. Faster builds, more reliable, still small.
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS builder

# Install C build tools needed for cryptography, asyncpg compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first — Docker caches this layer if requirements.txt
# hasn't changed, so pip install only re-runs when deps actually change.
COPY requirements.txt .

# Install into a prefix directory we can copy wholesale into the runtime stage.
# --no-cache-dir: don't store the pip cache in the image (saves ~50MB)
# --prefix: install to /install instead of the system Python, for clean copying
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS runtime

# Runtime system deps: libssl is needed at runtime by cryptography (not just
# build time). git is needed by the RAG indexer (git clone).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user — never run production services as root.
# Railway supports non-root containers natively.
RUN useradd --create-home --shell /bin/bash devmind
USER devmind
WORKDIR /home/devmind/app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source — only what the webhook receiver needs.
# The processor/ (Lambda handler) is NOT included — it's deployed separately.
COPY --chown=devmind:devmind app/ ./app/
COPY --chown=devmind:devmind alembic.ini ./alembic.ini

# Railway injects PORT at runtime. We default to 8080 if it's not set
# (local docker run without -e PORT=...).
ENV PORT=8080

# Uvicorn is the ASGI server. Key flags:
#   --host 0.0.0.0       Listen on all interfaces (required in containers)
#   --port $PORT         Railway's injected port
#   --workers 2          Two worker processes — Railway's free tier has 512MB
#                        RAM; 2 workers is safe, 4+ would OOM.
#   --log-level info     Structured logs for Railway's log viewer
#   --no-access-log      Access logs are noisy; Railway already captures
#                        request metrics separately
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 2 \
    --log-level info \
    --no-access-log
