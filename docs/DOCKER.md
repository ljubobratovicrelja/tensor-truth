# Tensor-Truth Docker Image

Official Docker image for running Tensor-Truth RAG application with GPU acceleration.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/ljubobratovicrelja/tensor-truth)
[![Docker Hub](https://img.shields.io/docker/v/ljubobratovicrelja/tensor-truth?label=version)](https://hub.docker.com/r/ljubobratovicrelja/tensor-truth)
[![Docker Pulls](https://img.shields.io/docker/pulls/ljubobratovicrelja/tensor-truth)](https://hub.docker.com/r/ljubobratovicrelja/tensor-truth)

## Quick Start

**Pull the pre-built image from Docker Hub:**

```bash
docker pull ljubobratovicrelja/tensor-truth:latest
```

**Run the container:**

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8000:8000 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

Access the application at **http://localhost:8000**

## What's Included

This Docker image provides a complete, minimal environment for running Tensor-Truth:

### Base Image
- **PyTorch 2.9.0** with CUDA 12.8 runtime and cuDNN 9
- **Python 3.11.4**
- Pre-configured for NVIDIA GPU acceleration

### Installed Components
- **React** web interface
- **LlamaIndex** RAG orchestration framework
- **ChromaDB** vector database
- **HuggingFace embeddings** (BAAI/bge-m3)
- **Cross-encoder rerankers** (bge-reranker-v2-m3)
- **PDF processing** (pymupdf4llm, marker-pdf)
- **Torch** ML libraries with CUDA support

### Image Size

**`ljubobratovicrelja/tensor-truth:latest`** - ~5GB (base PyTorch image is 4.6GB)

Includes all dependencies for CLI tools (`tensor-truth-docs`, `tensor-truth-build`).

Pre-built indexes are available on HuggingFace Hub and can be downloaded through the web UI on first launch.

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
  -p 8000:8000 \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest
```

### Custom Ollama Host

If Ollama runs on a different machine or port:

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8000:8000 \
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

Access at **http://localhost:8000** (no port mapping needed with `--network host`)

### Custom Port

Serve on a different host port (e.g., 8080):

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8080:8000 \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest
```

Access at **http://localhost:8080**

## Using CLI Tools (tensor-truth-docs, tensor-truth-build)

The image includes all dependencies for the full documentation pipeline:

- **`tensor-truth-docs`** - Fetch documentation (libraries, papers, books) and convert to markdown
- **`tensor-truth-build`** - Build vector indexes from the fetched documentation

These tools allow you to create custom indexes beyond the pre-built ones that download automatically on first run.

**Note**: Examples below use the Docker Hub image `ljubobratovicrelja/tensor-truth:latest`. If you built locally, replace with `tensor-truth:latest`.

### Running CLI Commands

The default `CMD` starts the TensorTruth server. To run CLI tools, override the command:

**Interactive shell (recommended for multiple commands):**

```bash
docker run -it --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  /bin/bash

# Inside the container:
tensor-truth-docs --list
tensor-truth-docs pytorch_2.9
tensor-truth-build --modules pytorch_2.9
```

**Single command execution:**

```bash
# List available sources
docker run --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  tensor-truth-docs --list

# Fetch documentation
docker run --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  tensor-truth-docs pytorch_2.9

# Build indexes
docker run --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  tensor-truth-build --modules pytorch_2.9
```

### Data Persistence with CLI Tools

**All data persists automatically** through the volume mount:

- `tensor-truth-docs` saves to `/root/.tensortruth/library_docs/` (mapped to `~/.tensortruth/library_docs/` on host)
- `tensor-truth-build` saves to `/root/.tensortruth/indexes/` (mapped to `~/.tensortruth/indexes/` on host)

When you restart the container (even the web app), your custom indexes are still there because they're stored on the host filesystem, not in the container.

**Example workflow (adding arXiv papers):**

```bash
# 1. Interactive mode (recommended) - guides you through the process
docker run -it --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  tensor-truth-docs --add

# OR use command-line mode with specific ArXiv IDs
docker run --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  tensor-truth-docs --type papers --category foundation_models --arxiv-ids 1706.03762 1810.04805

# 2. Build index for the category
docker run --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  tensor-truth-build --modules foundation_models

# 3. Restart web app - new indexes are immediately available
docker restart tensor-truth
```

**For comprehensive guides** on adding libraries, papers, and books, configuring chunk sizes, and troubleshooting, see [INDEXES.md](INDEXES.md).

### Advanced: Build Custom Indexes Only

If you prefer to build only your own custom indexes instead of downloading the pre-built ones:

1. **Before first launch**, enter the container shell and build your indexes:

```bash
docker run -it --rm --gpus all \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest \
  /bin/bash

# Inside container: fetch and build your custom indexes
tensor-truth-docs pytorch_2.9
tensor-truth-build --modules pytorch_2.9
exit
```

2. **Then start the app normally**:

```bash
docker run -d --name tensor-truth --gpus all -p 8000:8000 \
  -v ~/.tensortruth:/root/.tensortruth \
  ljubobratovicrelja/tensor-truth:latest
```

The app detects existing indexes in `~/.tensortruth/indexes/` and will use your custom-built indexes directly.

## Data Persistence

The image uses a volume mount at `/root/.tensortruth` for persistent data storage.

### What's Stored

- **Chat sessions** - conversation history and metadata
- **Presets** - saved RAG configurations
- **Vector indexes** - ChromaDB databases for document retrieval (including custom-built ones)
- **Library documentation** - Source files fetched via `tensor-truth-docs`
- **Session PDFs** - uploaded documents and their conversions
- **Configuration** - user settings and sources.json (created when using `tensor-truth-docs`)

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

This allows you to switch between Docker and pip-installed versions seamlessly. Indexes built in Docker are accessible to local installations and vice versa.

## First Run Behavior

On the first launch, the application will:

1. **Create config** - Initialize default configuration file
2. **Setup directories** - Create session, preset, and index folders
3. **Prompt for indexes** - The web UI will prompt you to download pre-built vector indexes from HuggingFace Hub
4. **Prompt for Ollama model** - The web UI will prompt you to pull `qwen2.5:0.5b` from the Ollama host for automatic chat titles

Index download and model pulls happen through the web UI, not automatically on startup. Once downloaded, data persists in the volume across container restarts.

**Note**: The Ollama model pull happens on your Ollama host machine (e.g., `host.docker.internal:11434`), not inside the Docker container. Make sure Ollama is running and accessible before starting tensor-truth.

## Environment Variables

### Application Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | URL for Ollama API endpoint |

### CLI Tools Configuration

These environment variables are used by `tensor-truth-docs` and `tensor-truth-build`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TENSOR_TRUTH_DOCS_DIR` | `~/.tensortruth/library_docs` | Source documentation directory |
| `TENSOR_TRUTH_SOURCES_CONFIG` | `~/.tensortruth/sources.json` | Sources configuration file |
| `TENSOR_TRUTH_INDEXES_DIR` | `~/.tensortruth/indexes` | Vector indexes output directory |

See [INDEXES.md](INDEXES.md#-configuration-paths-environment-variables) for details on using these for custom deployments.

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
docker run -d --name tensor-truth --gpus all -p 8000:8000 \
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
  -p 8000:8000 \
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
  -p 8000:8000 \
  -v ~/.tensortruth:/root/.tensortruth \
  --link ollama:ollama \
  -e OLLAMA_HOST=http://ollama:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

## Common Issues

### Port Already in Use

If port 8000 is occupied:

```bash
# Use different host port
-p 8080:8000  # Access at localhost:8080
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

### Index Download Issues

If HuggingFace Hub download fails:

1. Check your internet connection and firewall settings
2. Verify HuggingFace Hub is accessible: `curl https://huggingface.co`
3. Check container logs: `docker logs tensor-truth`

### Out of Memory

If embeddings or reranking fail with OOM:

- Use smaller reranker model (e.g., `bge-reranker-base` instead of `bge-reranker-v2-m3`)
- Reduce `Top N` parameter in the UI (fewer documents retrieved/reranked)
- Ensure sufficient GPU VRAM (depends on the models used)

## Building Custom Images

If you want to build the Docker image locally from source instead of using the pre-built image:

```bash
docker build -t tensor-truth:latest .
```

Then run it:

```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8000:8000 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  tensor-truth:latest
```

The image is ~5GB and includes all dependencies. Pre-built indexes can be downloaded through the web UI from HuggingFace Hub.

**Note**: Local builds use the simplified image name `tensor-truth:latest` (without the `ljubobratovicrelja/` prefix). The pre-built image from Docker Hub uses the full name `ljubobratovicrelja/tensor-truth:latest`.
