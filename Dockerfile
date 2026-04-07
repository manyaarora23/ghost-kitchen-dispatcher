# KitchenFlow-v1 — Dockerfile
# ============================================================
# Multi-stage build: keeps the final image lean (~200MB).
# HF Spaces requires the app to listen on port 7860.
#
# Build:  docker build -t your-username/kitchenflow:latest .
# Run:    docker run -p 7860:7860 your-username/kitchenflow:latest
# Pull:   docker pull your-username/kitchenflow:latest

# ── Stage 1: dependency resolver ────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ──────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="your-username"
LABEL org.opencontainers.image.title="KitchenFlow-v1"
LABEL org.opencontainers.image.description="Ghost Kitchen Dispatcher — OpenEnv RL environment"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 kitchenuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY env.py      .
COPY tasks.py    .
COPY baseline.py .
COPY app.py      .
COPY openenv.yaml .
COPY README.md   .
COPY kitchenflow_dataset.csv .

# Ensure write permissions for result files
RUN chown -R kitchenuser:kitchenuser /app

USER kitchenuser

# Hugging Face Spaces expects port 7860
EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Validate environment on startup before serving
RUN python -c "from env import KitchenFlowEnv; e = KitchenFlowEnv(); e.reset(); print('✓ KitchenFlow-v1 environment OK')"

# Run the API server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
