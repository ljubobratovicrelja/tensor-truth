"""System and hardware detection utilities."""

import torch


def get_max_memory_gb():
    """Determine maximum available memory in GB.

    Detection order:
    - Mac (Apple Silicon): Unified memory (total system RAM)
    - Windows/Linux with CUDA: GPU VRAM
    - Fallback: CPU RAM

    Returns:
        Maximum memory in gigabytes
    """
    # Check if CUDA is available (Windows/Linux with NVIDIA GPU)
    if torch.cuda.is_available():
        try:
            _, total_bytes = torch.cuda.mem_get_info()
            return total_bytes / (1024**3)
        except Exception:
            pass

    # Check if MPS is available (Mac with Apple Silicon - unified memory)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            import psutil

            # On Apple Silicon, use total system RAM as it's unified memory
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            pass

    # Fallback to system RAM for CPU-only systems
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        # Ultimate fallback
        return 16.0
