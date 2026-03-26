FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy agent source
COPY *.py ./

# Render exposes PORT env var (default 10000)
ENV PORT=10000

EXPOSE 10000

# Health check for Render
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health')"

CMD ["uv", "run", "python", "main.py"]
