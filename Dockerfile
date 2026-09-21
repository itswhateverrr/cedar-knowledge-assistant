# Image for the Cedar Health Alliance knowledge assistant.
# Build:  docker build -t cedar-assistant .
# Run:    docker run --rm -p 8000:8000 --env-file .env cedar-assistant

FROM python:3.12-slim

# PYTHONUNBUFFERED: log lines appear immediately in `docker logs`.
# PIP_NO_CACHE_DIR: keeps the image smaller.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Layer order matters: things that change rarely go first, so Docker can reuse
# its cached layers when only the application code changes.

# CPU-only PyTorch (a few hundred MB) instead of the default build that pulls
# in GPU libraries (several GB). This server has no GPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model at build time, so containers start fast and
# don't need to reach the internet for it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application code. Secrets (.env) are NOT copied; they are passed at runtime.
COPY app.py mock_docs.py ./
COPY documents/ documents/
COPY static/ static/
COPY prompts/ prompts/
COPY evals/ evals/

# Run as a normal user instead of root, so a bug in the app can't do as much.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Docker marks the container "unhealthy" if this fails 3 times in a row.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["python", "app.py"]
