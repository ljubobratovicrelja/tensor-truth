"""Default preset configurations for Tensor-Truth.

This module defines the standard presets that ship with Tensor-Truth.
On first launch, these presets are written to ~/.tensortruth/presets.json.
"""


def get_default_presets():
    """
    Returns the default preset configurations.

    These 4 presets cover the main use cases:
    - DL Researcher: Theory and research papers
    - Computer Vision: 2D vision with papers and APIs
    - 3D Vision Research: 3D reconstruction and rendering
    - PyTorch Developer: API-focused coding
    """
    return {
        "DL Researcher": {
            "description": "Explore deep learning theory and research papers",
            "modules": [
                "papers_dl_architectures_optimization",
                "book_dive_deep_learning_zhang",
                "book_mathematics_ml_deisenroth",
            ],
            "model": "qwen3:8b-q8_0",
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.6,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.15,
            "confidence_cutoff_hard": 0.08,
            "system_prompt": (
                "Explain deep learning concepts with mathematical rigor. "
                "Reference indexed papers and textbooks."
            ),
        },
        "Computer Vision": {
            "description": "2D vision with papers and OpenCV/PyTorch APIs",
            "modules": [
                "papers_vision_2d_generative",
                "library_opencv_4.12",
                "library_pytorch_2.9",
                "library_pillow_12.0",
            ],
            "model": "qwen3:8b-q8_0",
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.5,
            "reranker_top_n": 4,
            "confidence_cutoff": 0.2,
            "confidence_cutoff_hard": 0.1,
            "system_prompt": (
                "Combine classical CV and deep learning approaches. "
                "Reference both papers and API docs."
            ),
        },
        "3D Vision Research": {
            "description": "3D reconstruction, NeRF, and Gaussian Splatting",
            "modules": [
                "papers_3d_reconstruction_rendering",
                "book_linear_algebra_cherney",
                "library_pytorch_2.9",
            ],
            "model": "deepseek-r1:8b",
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.5,
            "reranker_top_n": 5,
            "confidence_cutoff": 0.15,
            "confidence_cutoff_hard": 0.08,
            "system_prompt": (
                "Explain 3D reconstruction and rendering with mathematical rigor. "
                "Connect theory to implementation."
            ),
        },
        "PyTorch Developer": {
            "description": "Write PyTorch code with API documentation",
            "modules": [
                "library_pytorch_2.9",
                "library_numpy_2.3",
            ],
            "model": "qwen3:8b-q8_0",
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "context_window": 8192,
            "temperature": 0.2,
            "reranker_top_n": 3,
            "confidence_cutoff": 0.25,
            "confidence_cutoff_hard": 0.1,
            "system_prompt": (
                "Write efficient, idiomatic PyTorch code. "
                "Reference API documentation for accuracy."
            ),
        },
    }


def ensure_presets_exist(presets_file):
    """
    Ensure presets file exists. If not, generate it from defaults.

    Args:
        presets_file: Path to presets.json

    Returns:
        bool: True if presets were generated, False if they already existed
    """
    import json
    import os

    if os.path.exists(presets_file):
        return False

    # Ensure directory exists
    os.makedirs(os.path.dirname(presets_file), exist_ok=True)

    # Generate presets from defaults (no model resolution needed)
    default_presets = get_default_presets()

    # Write to file with nice formatting
    with open(presets_file, "w", encoding="utf-8") as f:
        json.dump(default_presets, f, indent=2)

    return True
