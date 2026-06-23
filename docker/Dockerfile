FROM python:3.11-slim

# Install system deps needed by faiss and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .

# Corpus is mounted at runtime (see docker-compose.yml)
# so we just create the placeholder directory
RUN mkdir -p data/corpus

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "main.py"]
