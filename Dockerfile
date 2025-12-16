# Tensor-Truth Docker Image
# Base: PyTorch 2.9.0 with CUDA 12.8 runtime (Python 3.11.4)
# Purpose: Minimal containerized deployment for local RAG workflow with Ollama

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install tensor-truth (base package only, no dev/utils extras)
RUN pip install --no-cache-dir tensor-truth

# Expose Streamlit default port
EXPOSE 8501

# Create volume mount point for persistent data
# This will store: chat sessions, presets, indexes, config
VOLUME ["/root/.tensortruth"]

# Set default Ollama host (can be overridden via -e flag)
# Use host.docker.internal for Docker Desktop, or specify custom host at runtime
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Launch Streamlit app
CMD ["tensor-truth"]
