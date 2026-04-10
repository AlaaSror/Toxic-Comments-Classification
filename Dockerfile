# ================================================================
#  Toxic-Comments-Classification
#  Base: pytorch/pytorch — PyTorch + torchvision already included
#
#  Tag options (swap as needed):
#    Smaller  : pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime  ← used here
#    Full dev : pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# ================================================================
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

LABEL maintainer="AlaaSror"
LABEL description="Toxic Comments Classification — sentiment analysis & auto-response"

WORKDIR /app

# Minimal system dependencies (torch/torchvision already in base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Step 1: Install CUDA torch first
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.0+cu121 \
        torchvision==0.16.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Step 2: Install everything else, excluding torch/torchvision/transformers
RUN grep -vE "^(torch(vision)?|transformers)[>=<!]" requirements.txt > /tmp/requirements_notorch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_notorch.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir "transformers==4.36.0"

    
# Pre-download NLTK corpora so container works fully offline
RUN python -m nltk.downloader -d /usr/share/nltk_data \
        punkt \
        punkt_tab \
        stopwords \
        wordnet \
        averaged_perceptron_tagger \
        omw-1.4

# ── Environment variables ────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data \
    # Mount a volume at /app/hf_cache in production to avoid
    # re-downloading transformer models on every container restart
    TRANSFORMERS_CACHE=/app/hf_cache \
    HF_HOME=/app/hf_cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

# Copy project source
COPY . .

# Runtime directories + non-root user
RUN mkdir -p /app/hf_cache /app/saved_models /app/logs && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

# Default: FastAPI — override to run Streamlit:
#   docker run ... toxic-comments streamlit run app.py
CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0"]
