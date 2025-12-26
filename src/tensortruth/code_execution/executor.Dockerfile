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

# Install PyTorch CPU-only version
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Create non-root user for code execution
RUN useradd -m -u 1000 coderunner && \
    mkdir -p /workspace && \
    chown coderunner:coderunner /workspace

# Copy session runner script
COPY session_runner.py /home/coderunner/session_runner.py
RUN chown coderunner:coderunner /home/coderunner/session_runner.py

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER coderunner

# Run persistent Python session
# This maintains state (variables, imports) across multiple code executions
CMD ["python", "/home/coderunner/session_runner.py"]
