# Tensor-Truth Docker Image
# Base: PyTorch 2.9.0 with CUDA 12.8 runtime (Python 3.11.4)
# Purpose: Minimal containerized deployment for local RAG workflow with Ollama
#
# Build:
#   docker build -t tensor-truth:latest .
#
# Image size: ~5GB (base PyTorch image is 4.6GB)
# Indexes are downloaded automatically on first run via HuggingFace Hub

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install tensor-truth with docs extras for CLI tools support
# Includes: base package + documentation scraping (beautifulsoup4, markdownify,
# sphobjinv, arxiv) for tensor-truth-docs and tensor-truth-build commands
RUN pip install --no-cache-dir tensor-truth[docs]

# Set default Ollama host (can be overridden via -e flag)
# Use host.docker.internal for Docker Desktop on Mac/Windows
# For Linux: use --network=host or specify host IP
ENV OLLAMA_HOST=http://host.docker.internal:11434

# CLI path configuration environment variables
# These enable tensor-truth-docs and tensor-truth-build to use custom paths
ENV TENSOR_TRUTH_DOCS_DIR=/root/.tensortruth/library_docs
ENV TENSOR_TRUTH_SOURCES_CONFIG=/root/.tensortruth/sources.json
ENV TENSOR_TRUTH_INDEXES_DIR=/root/.tensortruth/indexes

# Expose Streamlit default port
EXPOSE 8501

# Create volume mount point for persistent data
VOLUME ["/root/.tensortruth"]

# Launch Streamlit app
CMD ["tensor-truth"]
