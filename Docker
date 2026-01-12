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

# 1. Install dependencies using uv
COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

# 2. BAKE THE MODEL
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# 3. Copy project files
COPY . .

RUN mkdir -p input memory outputs config && \
    chmod -R 777 input memory outputs config

# 4. Security: Non-root user
RUN useradd -m esguser
USER esguser

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://localhost:{PORT}/status')" || exit 1

# 6. Start the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]


