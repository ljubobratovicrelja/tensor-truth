# Tensor-Truth Docker Image

Official Docker image for running Tensor-Truth RAG application with GPU acceleration.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/ljubobratovicrelja/tensor-truth)
[![Docker Hub](https://img.shields.io/docker/v/ljubobratovicrelja/tensor-truth?label=version)](https://hub.docker.com/r/ljubobratovicrelja/tensor-truth)
[![Docker Pulls](https://img.shields.io/docker/pulls/ljubobratovicrelja/tensor-truth)](https://hub.docker.com/r/ljubobratovicrelja/tensor-truth)

## Quick Start

Pull and run the latest image:

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

Access the application at **http://localhost:8501**

## What's Included

This Docker image provides a complete, minimal environment for running Tensor-Truth:

### Base Image
- **PyTorch 2.9.0** with CUDA 12.8 runtime and cuDNN 9
- **Python 3.11.4**
- Pre-configured for NVIDIA GPU acceleration

### Installed Components
- **Streamlit** web interface
- **LlamaIndex** RAG orchestration framework
- **ChromaDB** vector database
- **HuggingFace embeddings** (BAAI/bge-m3)
- **Cross-encoder rerankers** (bge-reranker-v2-m3)
- **PDF processing** (pymupdf4llm, marker-pdf)
- **Torch** ML libraries with CUDA support

### What's NOT Included
The image excludes optional development and documentation tools to keep it minimal:
- `[docs]` extras (BeautifulSoup, arxiv, sphobjinv) - only needed for `tensor-truth-docs` CLI
- `[dev]` extras (pytest, black, mypy) - development dependencies

## Prerequisites

### Required
- **Docker** with GPU support (NVIDIA Container Toolkit)
- **NVIDIA GPU** with CUDA-capable drivers
- **Ollama** running locally or on accessible host (for LLM inference)

### Installation Links
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows)
- [Docker Engine](https://docs.docker.com/engine/install/) (Linux)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Ollama](https://ollama.ai/)

## Usage

### Basic Usage

Run with default settings (connects to Ollama on host machine):

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest
```

### Custom Ollama Host

If Ollama runs on a different machine or port:

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://192.168.1.100:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

### Linux Networking

On Linux, use host networking for easier Ollama connectivity:

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  --network host \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://localhost:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

Access at **http://localhost:8501** (no port mapping needed with `--network host`)

### Custom Port

Serve on a different host port (e.g., 8080):

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8080:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest
```

Access at **http://localhost:8080**

## Data Persistence

The image uses a volume mount at `/root/.tensortruth` for persistent data storage.

### What's Stored

- **Chat sessions** - conversation history and metadata
- **Presets** - saved RAG configurations
- **Vector indexes** - ChromaDB databases for document retrieval
- **Session PDFs** - uploaded documents and their conversions
- **Configuration** - user settings and preferences

### Backup Your Data

```bash
# Backup
docker cp tensor-truth:/root/.tensortruth ./tensortruth-backup

# Restore
docker cp ./tensortruth-backup/. tensor-truth:/root/.tensortruth
```

### Shared Data Directory

To share data between Docker and local installation:

```bash
-v ~/.tensortruth:/root/.tensortruth
```

This allows you to switch between Docker and pip-installed versions seamlessly.

## First Run Behavior

On the first launch, the application will:

1. **Create config** - Initialize default configuration file
2. **Download indexes** - Fetch pre-built vector indexes from Google Drive (~500MB)
3. **Setup directories** - Create session, preset, and index folders

This process takes 2-5 minutes depending on network speed. Subsequent runs are instant since data persists in the volume.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | URL for Ollama API endpoint |

### Setting Environment Variables

Via command line:

```bash
-e OLLAMA_HOST=http://192.168.1.100:11434
```

Via environment file:

```bash
# Create .env file
echo "OLLAMA_HOST=http://192.168.1.100:11434" > .env

# Run with env file
docker run -d --name tensor-truth --gpus all -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  --env-file .env \
  ljubobratovicrelja/tensor-truth:latest
```

## GPU Support

### Verify GPU Access

Check if container can see your GPU:

```bash
docker run --rm --gpus all pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime nvidia-smi
```

You should see your GPU listed.

### Troubleshooting GPU Issues

**Error: "could not select device driver"**

Install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Error: "Unknown runtime specified nvidia"**

On newer Docker versions, use `--gpus all` instead of `--runtime=nvidia`.

### CPU-Only Mode

While not recommended (significantly slower), you can run without GPU:

```bash
docker run -d \
  --name tensor-truth \
  -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest
```

Note: Embeddings and reranking will be much slower on CPU.

## Networking

### Connecting to Ollama

**Docker Desktop (Mac/Windows):**
- Use `http://host.docker.internal:11434` (default)

**Linux:**
- Option 1: Use `--network host` and `http://localhost:11434`
- Option 2: Find host IP with `ip addr show` and use `http://HOST_IP:11434`
- Option 3: Run Ollama in Docker too and link containers

### Running Ollama in Docker

```bash
# Start Ollama container
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama

# Pull a model
docker exec -it ollama ollama pull deepseek-r1:8b

# Connect Tensor-Truth to Ollama container
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  --link ollama:ollama \
  -e OLLAMA_HOST=http://ollama:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

## Common Issues

### Port Already in Use

If port 8501 is occupied:

```bash
# Use different host port
-p 8080:8501  # Access at localhost:8080
```

### Ollama Connection Failed

Check Ollama is accessible:

```bash
# From your host
curl http://localhost:11434/api/tags

# From inside container
docker exec tensor-truth curl http://host.docker.internal:11434/api/tags
```

If connection fails, verify firewall settings and OLLAMA_HOST configuration.

### Index Download Fails

If Google Drive download fails on first run:

1. Download indexes manually: [Google Drive Link](https://drive.google.com/file/d/12wZsBwrywl9nXOCLr50lpWB2SiFdu1XB/view?usp=sharing)
2. Extract to `~/.tensortruth/indexes/`
3. Restart container

### Out of Memory

If embeddings or reranking fail with OOM:

- Use smaller reranker model (e.g., `bge-reranker-base` instead of `bge-reranker-v2-m3`)
- Reduce `Top N` parameter in the UI (fewer documents retrieved/reranked)
- Ensure sufficient GPU VRAM (depends on the models used)
