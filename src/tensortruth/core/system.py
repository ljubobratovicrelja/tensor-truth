"""System and hardware detection utilities."""

import torch


def get_max_memory_gb():
    """
    Dynamically determines maximum available memory in GB.
    - Mac (Apple Silicon): Uses unified memory (total system RAM)
    - Windows/Linux with CUDA: Uses GPU VRAM
    - Fallback: CPU RAM
    """
    import platform

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
