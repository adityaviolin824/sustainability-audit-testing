FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

COPY requirements.txt .
RUN uv pip install --no-cache --system --index-strategy unsafe-best-match -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# 1. Copy your project files into the container
COPY . .

# 2. THE FIX: Create the user AND give them ownership of the entire /app directory
# This must happen as the LAST step before switching to the user.
RUN useradd -m esguser && chown -R esguser:esguser /app

# 3. Switch to the non-root user
USER esguser

EXPOSE ${PORT}

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://localhost:{PORT}/status')" || exit 1

# Start
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]