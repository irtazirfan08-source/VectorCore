FROM python:3.12-slim

# Prevent Python from writing bytecode and enable unbuffered stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip and install server dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn httpx numpy

# Copy package metadata and project source
COPY pyproject.toml README.md ./
COPY vector_core/ ./vector_core/

# Install VectorCore in editable mode
RUN pip install --no-cache-dir -e .

# Security: run as non-root user
RUN useradd -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "vector_core.server.app:app", "--host", "0.0.0.0", "--port", "8001"]