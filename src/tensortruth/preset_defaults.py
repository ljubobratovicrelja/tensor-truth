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
            "description": (
                "Explore deep learning theory and research with comprehensive references"
            ),
            "favorite": True,
            "favorite_order": 0,
            "modules": [
                "book_deep_learning_goodfellow",
                "book_dive_deep_learning_zhang",
                "dl_foundations",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.6,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.0,
            "confidence_cutoff_hard": 0.0,
            "agent_max_iterations": 12,  # More iterations for thorough research
            "agent_min_pages": 3,  # Require at least 3 pages for research depth
            "agent_reasoning_model": None,  # Will use config default
        },
        "DL Coder": {
            "description": "Write deep learning code with PyTorch and API documentation",
            "favorite": True,
            "favorite_order": 1,
            "modules": [
                "pytorch_2.9",
                "book_dive_deep_learning_zhang",
                "matplotlib_3.10",
                "dl_foundations",
            ],
            "model_preference": "coding",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 16384,
            "temperature": 0.1,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.3,
            "confidence_cutoff_hard": 0.15,
            "system_prompt": (
                "Focus on producing working code with proper PyTorch idioms. "
                "Reference the indexed documentation for API correctness."
            ),
        },
        "Math Foundations": {
            "description": "Learn linear algebra and calculus with rigorous explanations",
            "modules": [
                "book_linear_algebra_cherney",
                "book_linear_algebra_axler",
                "book_diff_equations_trench",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.2,
            "confidence_cutoff_hard": 0.1,
            "system_prompt": (
                "Provide rigorous mathematical explanations. "
                "Use LaTeX for all equations."
            ),
        },
        "ML Theory": {
            "description": "Study machine learning theory with mathematical foundations",
            "modules": [
                "book_mathematics_ml_deisenroth",
                "book_linear_algebra_cherney",
                "book_convex_optimization_boyd",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.1,
            "confidence_cutoff_hard": 0.05,
            "system_prompt": (
                "Connect theoretical concepts across optimization, algebra, "
                "and learning theory. Cite specific theorems and proofs when relevant."
            ),
        },
        "PyTorch Developer": {
            "description": "Build PyTorch projects with API-accurate code examples",
            "favorite": True,
            "favorite_order": 2,
            "modules": ["pytorch_2.9", "numpy_2.3"],
            "model_preference": "coding",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 16384,
            "temperature": 0.1,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.35,
            "confidence_cutoff_hard": 0.2,
            "system_prompt": (
                "Write efficient, idiomatic PyTorch code. Prioritize API "
                "documentation accuracy."
            ),
        },
        "Computer Vision": {
            "description": "Combine classical CV and deep learning for vision tasks",
            "modules": [
                "pytorch_2.9",
                "opencv_4.12",
                "pillow_12.0",
                "vision_2d_generative",
                "dl_foundations",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.2,
            "confidence_cutoff_hard": 0.1,
            "system_prompt": (
                "Integrate classical CV techniques with modern DL approaches. "
                "Reference both traditional and neural methods."
            ),
        },
        "CV Coding": {
            "description": "Write computer vision pipelines with OpenCV and PyTorch",
            "modules": ["opencv_4.12", "pytorch_2.9", "pillow_12.0", "numpy_2.3"],
            "model_preference": "coding",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 16384,
            "temperature": 0.1,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.3,
            "confidence_cutoff_hard": 0.15,
            "system_prompt": (
                "Implement computer vision pipelines. Combine OpenCV "
                "preprocessing with PyTorch models efficiently."
            ),
        },
        "3D Vision Research": {
            "description": "Explore 3D reconstruction and rendering with math",
            "modules": [
                "3d_reconstruction_rendering",
                "book_linear_algebra_cherney",
                "book_linear_algebra_axler",
                "pytorch_2.9",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.15,
            "confidence_cutoff_hard": 0.07,
            "system_prompt": (
                "Explain 3D reconstruction and rendering techniques with "
                "mathematical rigor. Connect theory to implementation.",
            ),
        },
        "Generative Models": {
            "description": "Study GANs, diffusion models, and generative architectures",
            "modules": [
                "vision_2d_generative",
                "dl_foundations",
                "book_deep_learning_goodfellow",
                "book_dive_deep_learning_zhang",
                "pytorch_2.9",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.2,
            "confidence_cutoff_hard": 0.1,
            "system_prompt": (
                "Discuss GANs, diffusion models, and other generative approaches. "
                "Balance theory with practical architectures.",
            ),
        },
        "NLP & Transformers": {
            "description": "Work with transformers and NLP using HuggingFace",
            "modules": ["transformers_4.57", "dl_foundations", "pytorch_2.9"],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.25,
            "confidence_cutoff_hard": 0.12,
            "system_prompt": (
                "Reference HuggingFace Transformers API and attention mechanisms. "
                "Connect architecture papers to library usage."
            ),
        },
        "Data Science": {
            "description": "Analyze data with pandas, scikit-learn, and visualization",
            "modules": [
                "pandas_2.3",
                "numpy_2.3",
                "matplotlib_3.10",
                "scikit-learn_1.8",
            ],
            "model_preference": "coder_light",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 16384,
            "temperature": 0.2,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.3,
            "confidence_cutoff_hard": 0.15,
            "system_prompt": (
                "Write data analysis code. Focus on pandas operations, "
                "visualization, and classical ML workflows."
            ),
        },
        "Scientific Computing": {
            "description": "Implement numerical algorithms with NumPy and SciPy",
            "modules": [
                "numpy_2.3",
                "scipy_1.15",
                "book_diff_equations_trench",
                "book_convex_optimization_boyd",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.3,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.25,
            "confidence_cutoff_hard": 0.12,
            "system_prompt": (
                "Implement numerical algorithms correctly. Connect "
                "mathematical theory to NumPy/SciPy implementations."
            ),
        },
        "Optimization Expert": {
            "description": "Master optimization from classical methods to deep learning",
            "modules": [
                "book_convex_optimization_boyd",
                "book_diff_equations_trench",
                "book_linear_algebra_cherney",
                "book_linear_algebra_axler",
                "dl_foundations",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.1,
            "confidence_cutoff_hard": 0.05,
            "system_prompt": (
                "Explain optimization algorithms from first principles. Cover both "
                "classical numerical methods and modern DL optimizers."
            ),
        },
        "Linear Algebra": {
            "description": "Learn linear algebra theory with NumPy implementations",
            "modules": [
                "book_linear_algebra_cherney",
                "book_linear_algebra_axler",
                "numpy_2.3",
            ],
            "model_preference": "lightweight",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 16384,
            "temperature": 0.4,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.2,
            "confidence_cutoff_hard": 0.1,
            "system_prompt": (
                "Teach linear algebra concepts clearly. Connect abstract "
                "theory to concrete NumPy operations.",
            ),
        },
        "Paper Reading": {
            "description": "Analyze and compare deep learning research papers in depth",
            "modules": [
                "dl_foundations",
                "vision_2d_generative",
                "3d_reconstruction_rendering",
            ],
            "model_preference": "reasoning",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 32768,
            "temperature": 0.6,
            "reranker_top_n": 6,
            "confidence_cutoff": 0.1,
            "confidence_cutoff_hard": 0.05,
            "system_prompt": (
                "Analyze research papers in depth. Explain methods, "
                "compare approaches, and discuss limitations.",
            ),
        },
        "Lightweight Coder": {
            "description": "Quick coding with minimal VRAM and fast models",
            "modules": ["pytorch_2.9", "numpy_2.3"],
            "model_preference": "coder_light",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "context_window": 16384,
            "temperature": 0.1,
            "reranker_top_n": 2,
            "confidence_cutoff": 0.35,
            "confidence_cutoff_hard": 0.2,
            "system_prompt": "Write concise, correct code. Minimize VRAM usage.",
        },
        "Fast Researcher": {
            "description": "Get quick answers to ML/DL questions with speed focus",
            "modules": [
                "book_deep_learning_goodfellow",
                "book_dive_deep_learning_zhang",
                "book_mathematics_ml_deisenroth",
            ],
            "model_preference": "lightweight",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "context_window": 16384,
            "temperature": 0.5,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.15,
            "confidence_cutoff_hard": 0.07,
            "system_prompt": (
                "Answer ML/DL questions quickly. "
                "Prioritize speed over exhaustive detail."
            ),
            "agent_max_iterations": 6,  # Fewer iterations for speed
            "agent_reasoning_model": None,  # Will use config default
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
