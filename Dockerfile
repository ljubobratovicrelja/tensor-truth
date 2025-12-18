# Tensor-Truth Docker Image
# Base: PyTorch 2.9.0 with CUDA 12.8 runtime (Python 3.11.4)
# Purpose: Minimal containerized deployment for local RAG workflow with Ollama

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install tensor-truth (base package only, no dev/docs extras)
# Core dependencies include: Streamlit, LlamaIndex, ChromaDB, 
# embeddings, rerankers, and PDF processing (pymupdf4llm, marker-pdf)
RUN pip install --no-cache-dir tensor-truth

# Expose Streamlit default port
EXPOSE 8501

# Create volume mount point for persistent data
# This will store: chat sessions, presets, vector indexes, config, and session PDFs
VOLUME ["/root/.tensortruth"]

# Set default Ollama host (can be overridden via -e flag)
# Use host.docker.internal for Docker Desktop on Mac/Windows
# For Linux: use --network=host or specify host IP
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Launch Streamlit app
CMD ["tensor-truth"]
