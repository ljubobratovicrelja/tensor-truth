"""System and hardware detection utilities."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Memory usage information for a component."""

    name: str
    allocated_gb: float
    total_gb: Optional[float] = None
    details: Optional[str] = None

    def format_usage(self) -> str:
        """Format memory usage as string."""
        if self.total_gb is not None and self.total_gb > 0:
            pct = (self.allocated_gb / self.total_gb) * 100
            return f"{self.allocated_gb:.2f} / {self.total_gb:.2f} GB ({pct:.0f}%)"
        return f"{self.allocated_gb:.2f} GB"


def get_cuda_memory() -> Optional[MemoryInfo]:
    """Get CUDA GPU memory usage.

    Uses torch.cuda.mem_get_info() to get total GPU memory usage (matching nvidia-smi),
    not just PyTorch allocations. This captures memory used by all processes including
    Ollama/llama.cpp, other frameworks, and system overhead.

    Returns:
        MemoryInfo for CUDA device, or None if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        return None

    try:
        free, total = torch.cuda.mem_get_info()
        # Calculate used memory as total - free (matches nvidia-smi Memory-Usage)
        used_gb = (total - free) / (1024**3)
        total_gb = total / (1024**3)

        # PyTorch-specific allocations for details (useful for debugging)
        pytorch_allocated = torch.cuda.memory_allocated() / (1024**3)
        pytorch_reserved = torch.cuda.memory_reserved() / (1024**3)

        if pytorch_allocated > 0.01:  # Only show if meaningful
            details = f"PyTorch: {pytorch_allocated:.2f} GB (reserved: {pytorch_reserved:.2f} GB)"
        else:
            details = None

        return MemoryInfo(
            name="CUDA VRAM",
            allocated_gb=used_gb,
            total_gb=total_gb,
            details=details,
        )
    except Exception as e:
        logger.warning(f"Failed to get CUDA memory info: {e}")
        return None


def get_mps_memory() -> Optional[MemoryInfo]:
    """Get MPS (Apple Silicon) memory usage.

    Note: MPS uses unified memory, so we report allocated GPU memory
    against total system RAM.

    Returns:
        MemoryInfo for MPS device, or None if MPS unavailable.
    """
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None

    try:
        # Use driver_allocated_memory for actual GPU usage
        # (includes non-PyTorch allocations, more accurate for Apple Silicon)
        try:
            allocated = torch.mps.driver_allocated_memory() / (1024**3)
        except AttributeError:
            # Fallback to current_allocated_memory for older PyTorch versions
            allocated = torch.mps.current_allocated_memory() / (1024**3)

        # Get total system RAM for context (unified memory)
        try:
            import psutil

            total_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_gb = None

        return MemoryInfo(
            name="MPS (Unified)",
            allocated_gb=allocated,
            total_gb=total_gb,
            details="Apple Silicon unified memory",
        )
    except Exception as e:
        logger.warning(f"Failed to get MPS memory info: {e}")
        return None


def get_system_ram() -> Optional[MemoryInfo]:
    """Get system RAM usage.

    Returns:
        MemoryInfo for system RAM, or None if unavailable.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)

        return MemoryInfo(
            name="System RAM",
            allocated_gb=used_gb,
            total_gb=total_gb,
        )
    except ImportError:
        logger.warning("psutil not available for RAM monitoring")
        return None
    except Exception as e:
        logger.warning(f"Failed to get system RAM info: {e}")
        return None


def get_ollama_memory() -> Optional[MemoryInfo]:
    """Get Ollama model memory usage.

    Returns:
        MemoryInfo for Ollama models, or None if unavailable.
    """
    try:
        from tensortruth.core.ollama import get_running_models_detailed

        running_models = get_running_models_detailed()
        if not running_models:
            return None

        total_vram = 0
        model_names = []
        for model_info in running_models:
            vram = model_info.get("size_vram", 0)
            total_vram += vram
            model_names.append(model_info.get("name", "unknown"))

        if total_vram == 0:
            return None

        vram_gb = total_vram / (1024**3)
        details = f"Models: {', '.join(model_names)}"

        return MemoryInfo(
            name="Ollama VRAM",
            allocated_gb=vram_gb,
            details=details,
        )
    except Exception as e:
        logger.warning(f"Failed to get Ollama memory info: {e}")
        return None


def get_all_memory_info() -> List[MemoryInfo]:
    """Get comprehensive memory usage across all components.

    Returns:
        List of MemoryInfo objects for each available component.
    """
    memory_info = []

    # GPU memory (CUDA or MPS)
    cuda_mem = get_cuda_memory()
    if cuda_mem:
        memory_info.append(cuda_mem)

    mps_mem = get_mps_memory()
    if mps_mem:
        memory_info.append(mps_mem)

    # System RAM
    ram_mem = get_system_ram()
    if ram_mem:
        memory_info.append(ram_mem)

    return memory_info


def get_memory_summary() -> str:
    """Get a one-line memory summary for status display.

    Returns:
        Formatted string like "VRAM: 4.2/8.0 GB | RAM: 12.1/32.0 GB"
    """
    parts = []

    # GPU memory
    cuda_mem = get_cuda_memory()
    if cuda_mem:
        parts.append(f"VRAM: {cuda_mem.format_usage()}")
    else:
        mps_mem = get_mps_memory()
        if mps_mem:
            parts.append(f"MPS: {mps_mem.format_usage()}")

    # Ollama (if not showing CUDA, or if it adds useful info)
    if not cuda_mem:
        ollama_mem = get_ollama_memory()
        if ollama_mem:
            parts.append(f"Ollama: {ollama_mem.allocated_gb:.2f} GB")

    # System RAM
    ram_mem = get_system_ram()
    if ram_mem:
        parts.append(f"RAM: {ram_mem.format_usage()}")

    return " | ".join(parts) if parts else "Memory info unavailable"


def format_memory_report() -> List[str]:
    """Format detailed memory report for /memory command.

    Returns:
        List of markdown-formatted lines.
    """
    lines = ["### Memory Usage"]

    memory_info = get_all_memory_info()

    if not memory_info:
        lines.append("No memory information available.")
        return lines

    for mem in memory_info:
        lines.append(f"\n**{mem.name}**")
        lines.append(f"- Usage: `{mem.format_usage()}`")
        if mem.details:
            lines.append(f"- {mem.details}")

    # Add tips section
    lines.append("\n---")
    lines.append("**Tips:**")
    lines.append("- Use `/reload` to flush VRAM and restart the engine")
    lines.append("- Use `/device rag cpu` to move embedder/reranker to CPU")
    lines.append("- Use `/device llm cpu` to run LLM inference on CPU")

    return lines


def get_max_memory_gb() -> float:
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
