# Contributing to TensorTruth

Thanks for your interest in contributing. This is a single-developer project, so community input — whether bug reports, testing on different hardware, or code contributions — is especially valuable.

## Reporting Issues

The most impactful way to contribute is by reporting issues. The project is tested primarily on:
- Desktop Linux with RTX 3090 Ti (24GB VRAM)
- MacBook M1 Max (32GB unified memory)
- ASUS Ascent DX10

Testing with different GPUs, operating systems, and model combinations helps uncover problems that a single-machine setup can't catch.

**Useful bug reports include:**
- Hardware and OS details (GPU model, VRAM, OS version)
- LLM provider and model being used (Ollama, llama.cpp, OpenAI-compatible API)
- Steps to reproduce the issue
- Relevant log output (the terminal running `tensor-truth` shows backend logs)

**Areas where testing is particularly needed:**
- Different Ollama models, especially for agentic mode tool-calling reliability
- llama.cpp server setups (single model, router mode with multiple models)
- OpenAI-compatible API providers (vLLM, Groq, Together AI, LocalAI, etc.)
- macOS and Windows environments
- Lower-VRAM GPUs (8GB, 12GB) — what models and configurations work well
- Docker deployment on different host configurations

Open issues at [github.com/ljubobratovicrelja/tensor-truth/issues](https://github.com/ljubobratovicrelja/tensor-truth/issues).

## Feature Requests

If you have ideas for new features or improvements, open an issue describing the use case. Even if you don't plan to implement it yourself, the suggestion helps prioritize development.

## Code Contributions

Pull requests are welcome. Before starting significant work, open an issue first to discuss the approach — this avoids duplicated effort and ensures alignment with the project direction.

### Setting Up the Development Environment

```bash
git clone https://github.com/ljubobratovicrelja/tensor-truth.git
cd tensor-truth

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Running the App in Development Mode

```bash
# Terminal 1: Backend with auto-reload
tensor-truth --reload

# Terminal 2: Frontend dev server with hot-reload (port 5173)
tensor-truth-ui
```

### Running Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run with coverage
pytest tests/unit/ --cov=tensortruth
```

### Code Style

- **Python:** flake8 with max line length of 120 characters
- **Frontend:** TypeScript with the project's existing ESLint configuration
- **General:** match the style of surrounding code; avoid unrelated cleanups in the same PR

### Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include a clear description of what changed and why
- Add tests for new functionality where practical
- Make sure existing tests pass before submitting

## Extensions

You can contribute extensions (slash commands, agents) without modifying the core codebase. Drop YAML or Python files into the `extension_library/` directory and submit a PR. See [docs/EXTENSIONS.md](docs/EXTENSIONS.md) for the schema and examples.
