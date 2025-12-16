"""Default preset configurations for Tensor-Truth.

This module defines the standard presets that ship with Tensor-Truth.
On first launch, these presets are written to ~/.tensortruth/presets.json
with appropriate model filtering based on what's available in Ollama.
"""

# Model preference tiers - we'll try these in order for each preset category
MODEL_PREFERENCES = {
    "reasoning": ["deepseek-r1:14b", "deepseek-r1:8b", "qwen2.5:14b", "llama3:8b"],
    "coding": [
        "deepseek-coder-v2:16b",
        "qwen2.5-coder:14b",
        "qwen2.5-coder:7b",
        "codestral:latest",
    ],
    "lightweight": ["llama3:8b", "qwen2.5:7b", "phi3:latest"],
    "coder_light": ["qwen2.5-coder:7b", "qwen2.5-coder:3b", "llama3:8b"],
}


def get_default_presets():
    """
    Returns the default preset configurations.

    Note: Models are specified as preferences, not requirements.
    The actual model used will be selected from available Ollama models.
    """
    return {
        "DL Research Assistant": {
            "modules": [
                "deep_learning_books",
                "machine_learning_books",
                "dl_foundations",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.6,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.0,
        },
        "DL Coder": {
            "modules": [
                "pytorch_2.9",
                "deep_learning_books",
                "matplotlib_3.10",
                "dl_foundations",
            ],
            "model_preference": "coding",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 4096,
            "temperature": 0.1,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.3,
            "system_prompt": (
                "Focus on producing working code with proper PyTorch idioms. "
                "Reference the indexed documentation for API correctness."
            ),
        },
        "Math Foundations": {
            "modules": ["linear_algebra_books", "calculus_books"],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.2,
            "system_prompt": (
                "Provide rigorous mathematical explanations. "
                "Use LaTeX for all equations."
            ),
        },
        "ML Theory": {
            "modules": [
                "machine_learning_books",
                "linear_algebra_books",
                "numerical_optimization_books",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.1,
            "system_prompt": (
                "Connect theoretical concepts across optimization, algebra, "
                "and learning theory. Cite specific theorems and proofs when relevant."
            ),
        },
        "PyTorch Developer": {
            "modules": ["pytorch_2.9", "numpy_2.3"],
            "model_preference": "coding",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 4096,
            "temperature": 0.1,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.35,
            "system_prompt": (
                "Write efficient, idiomatic PyTorch code. Prioritize API "
                "documentation accuracy."
            ),
        },
        "Computer Vision": {
            "modules": [
                "pytorch_2.9",
                "opencv_4.12",
                "pillow_12.0",
                "vision_2d_generative",
                "dl_foundations",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 6144,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.2,
            "system_prompt": (
                "Integrate classical CV techniques with modern DL approaches. "
                "Reference both traditional and neural methods."
            ),
        },
        "CV Coding": {
            "modules": ["opencv_4.12", "pytorch_2.9", "pillow_12.0", "numpy_2.3"],
            "model_preference": "coding",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 4096,
            "temperature": 0.1,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.3,
            "system_prompt": (
                "Implement computer vision pipelines. Combine OpenCV "
                "preprocessing with PyTorch models efficiently."
            ),
        },
        "3D Vision Research": {
            "modules": [
                "3d_reconstruction_rendering",
                "linear_algebra_books",
                "pytorch_2.9",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.15,
            "system_prompt": (
                "Explain 3D reconstruction and rendering techniques with "
                "mathematical rigor. Connect theory to implementation.",
            ),
        },
        "Generative Models": {
            "modules": [
                "vision_2d_generative",
                "dl_foundations",
                "deep_learning_books",
                "pytorch_2.9",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 6144,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.2,
            "system_prompt": (
                "Discuss GANs, diffusion models, and other generative approaches. "
                "Balance theory with practical architectures.",
            ),
        },
        "NLP & Transformers": {
            "modules": ["transformers_4.57", "dl_foundations", "pytorch_2.9"],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 6144,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.25,
            "system_prompt": (
                "Reference HuggingFace Transformers API and attention mechanisms. "
                "Connect architecture papers to library usage."
            ),
        },
        "Data Science": {
            "modules": [
                "pandas_2.3",
                "numpy_2.3",
                "matplotlib_3.10",
                "scikit-learn_1.8",
            ],
            "model_preference": "coder_light",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 4096,
            "temperature": 0.2,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.3,
            "system_prompt": (
                "Write data analysis code. Focus on pandas operations, "
                "visualization, and classical ML workflows."
            ),
        },
        "Scientific Computing": {
            "modules": [
                "numpy_2.3",
                "scipy_1.15",
                "calculus_books",
                "numerical_optimization_books",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 6144,
            "temperature": 0.3,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.25,
            "system_prompt": (
                "Implement numerical algorithms correctly. Connect "
                "mathematical theory to NumPy/SciPy implementations."
            ),
        },
        "Optimization Expert": {
            "modules": [
                "numerical_optimization_books",
                "calculus_books",
                "linear_algebra_books",
                "dl_foundations",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.1,
            "system_prompt": (
                "Explain optimization algorithms from first principles. Cover both "
                "classical numerical methods and modern DL optimizers."
            ),
        },
        "Linear Algebra": {
            "modules": ["linear_algebra_books", "numpy_2.3"],
            "model_preference": "lightweight",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 4096,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.2,
            "system_prompt": (
                "Teach linear algebra concepts clearly. Connect abstract "
                "theory to concrete NumPy operations.",
            ),
        },
        "Quick Calc Helper": {
            "modules": ["calculus_books", "linear_algebra_books"],
            "model_preference": "lightweight",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "context_window": 3072,
            "temperature": 0.3,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.25,
            "system_prompt": (
                "Solve calculus and algebra problems "
                "step-by-step. Show all work clearly."
            ),
        },
        "Paper Reading": {
            "modules": [
                "dl_foundations",
                "vision_2d_generative",
                "3d_reconstruction_rendering",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.6,
            "reranker_top_n": 6,
            "confidence_cutoff": 0.1,
            "system_prompt": (
                "Analyze research papers in depth. Explain methods, "
                "compare approaches, and discuss limitations.",
            ),
        },
        "Lightweight Coder": {
            "modules": ["pytorch_2.9", "numpy_2.3"],
            "model_preference": "coder_light",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "context_window": 3072,
            "temperature": 0.1,
            "reranker_top_n": 2,
            "confidence_cutoff": 0.35,
            "system_prompt": "Write concise, correct code. Minimize VRAM usage.",
        },
        "Fast Researcher": {
            "modules": ["deep_learning_books", "machine_learning_books"],
            "model_preference": "lightweight",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "context_window": 3072,
            "temperature": 0.5,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.15,
            "system_prompt": (
                "Answer ML/DL questions quickly. "
                "Prioritize speed over exhaustive detail."
            ),
        },
    }


def resolve_model_for_preset(preset_config, available_models):
    """
    Resolve the best available model for a preset configuration.

    Args:
        preset_config: Dict containing preset configuration with 'model_preference' key
        available_models: List of available model names from Ollama

    Returns:
        str: The best available model name, or None if no suitable model found
    """
    preference_key = preset_config.get("model_preference", "reasoning")
    preferred_models = MODEL_PREFERENCES.get(
        preference_key, MODEL_PREFERENCES["reasoning"]
    )

    # Try each preferred model in order
    for model in preferred_models:
        if model in available_models:
            return model

    # Fallback: return first available model if any
    return available_models[0] if available_models else None


def ensure_presets_exist(presets_file, available_models=None):
    """
    Ensure presets file exists. If not, generate it from defaults.

    Args:
        presets_file: Path to presets.json
        available_models: Optional list of available models (will be fetched if not provided)

    Returns:
        bool: True if presets were generated, False if they already existed
    """
    import json
    import os

    if os.path.exists(presets_file):
        return False

    # Fetch available models if not provided
    if available_models is None:
        try:
            from tensortruth.app_utils.helpers import get_ollama_models

            available_models = get_ollama_models()
        except Exception:
            # Fallback to a sensible default if Ollama is unavailable
            available_models = ["deepseek-r1:8b"]

    # Ensure directory exists
    os.makedirs(os.path.dirname(presets_file), exist_ok=True)

    # Generate presets from defaults
    default_presets = get_default_presets()
    resolved_presets = {}

    for name, config in default_presets.items():
        # Create a copy without the model_preference key
        resolved_config = {k: v for k, v in config.items() if k != "model_preference"}

        # Resolve the model if available
        model = resolve_model_for_preset(config, available_models)
        if model:
            resolved_config["model"] = model

        resolved_presets[name] = resolved_config

    # Write to file with nice formatting
    with open(presets_file, "w", encoding="utf-8") as f:
        json.dump(resolved_presets, f, indent=2)

    return True
