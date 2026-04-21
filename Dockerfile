FROM python:3.11-slim

# System deps for sentence-transformers (torch needs build tools sometimes)
# and curl for the Lean install (optional — comment out the lean block if
# you don't want it in the image, the backend tolerates Lean being missing).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional: install Lean 4 via elan. Adds ~500MB. Comment out to skip.
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | \
    sh -s -- -y --default-toolchain leanprover/lean4:stable && \
    ln -sf /root/.elan/bin/lean /usr/local/bin/lean

ENV PATH="/root/.elan/bin:${PATH}"

WORKDIR /app

# Install Python deps first so Docker layer caching works
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model into the image so the first
# /papers/embed call doesn't pay the download cost. The model lives at
# ~/.cache/huggingface/hub by default; we keep it inside the image.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY agent_backend.py .

EXPOSE 8000

CMD ["uvicorn", "agent_backend:app", "--host", "0.0.0.0", "--port", "8000"]
