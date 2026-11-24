FROM python:3.13-slim

# Compilers for llama-cpp-python (built from source for this CPU); libgomp for parallel math
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (avoids ~2 GB of CUDA libraries), then the pinned requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# App code, shared RAG module and the index
COPY . .

# Bake both models into the image (one-off download at build time)
RUN python download_models.py

EXPOSE 7870

# One worker (the models are large), a few threads for concurrent requests; long timeout for CPU generation
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:7870", "--timeout", "600", "app:rag_api"]
