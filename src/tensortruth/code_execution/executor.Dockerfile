# Dockerfile for Python code execution container
# Provides an isolated, sandboxed environment for running LLM-generated code

FROM python:3.11-slim

# Install common data science and utility packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    requests \
    pillow

# Create non-root user for code execution
RUN useradd -m -u 1000 coderunner && \
    mkdir -p /workspace && \
    chown coderunner:coderunner /workspace

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER coderunner

# Keep container alive for exec commands
# This allows the container to persist state across multiple executions
CMD ["tail", "-f", "/dev/null"]
