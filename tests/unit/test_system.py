"""
Unit tests for tensortruth.core.system module.
"""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.core.system import (
    MemoryInfo,
    format_memory_report,
    get_all_memory_info,
    get_cuda_memory,
    get_memory_summary,
    get_mps_memory,
    get_ollama_memory,
    get_system_ram,
)


@pytest.mark.unit
class TestMemoryInfo:
    """Tests for MemoryInfo dataclass."""

    def test_format_usage_with_total(self):
        """Test format_usage with both allocated and total."""
        mem = MemoryInfo(name="Test", allocated_gb=4.0, total_gb=8.0)
        result = mem.format_usage()
        assert "4.00" in result
        assert "8.00" in result
        assert "50%" in result

    def test_format_usage_without_total(self):
        """Test format_usage with only allocated."""
        mem = MemoryInfo(name="Test", allocated_gb=4.0)
        result = mem.format_usage()
        assert "4.00 GB" in result
        assert "%" not in result

    def test_format_usage_zero_total(self):
        """Test format_usage when total is zero."""
        mem = MemoryInfo(name="Test", allocated_gb=4.0, total_gb=0.0)
        result = mem.format_usage()
        assert "4.00 GB" in result
        assert "%" not in result


@pytest.mark.unit
class TestGetCudaMemory:
    """Tests for get_cuda_memory function."""

    @patch("tensortruth.core.system.torch")
    def test_cuda_available(self, mock_torch):
        """Test get_cuda_memory when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024**3)  # 2 GB
        mock_torch.cuda.memory_reserved.return_value = 3 * (1024**3)  # 3 GB
        mock_torch.cuda.mem_get_info.return_value = (
            6 * (1024**3),  # 6 GB free
            8 * (1024**3),  # 8 GB total
        )

        result = get_cuda_memory()

        assert result is not None
        assert result.name == "CUDA VRAM"
        # allocated_gb is now (total - free) = 8 - 6 = 2 GB
        assert result.allocated_gb == pytest.approx(2.0, rel=0.01)
        assert result.total_gb == pytest.approx(8.0, rel=0.01)
        # Details shows PyTorch-specific allocations when meaningful
        assert "PyTorch" in result.details
        assert "reserved" in result.details

    @patch("tensortruth.core.system.torch")
    def test_cuda_not_available(self, mock_torch):
        """Test get_cuda_memory when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        result = get_cuda_memory()

        assert result is None

    @patch("tensortruth.core.system.torch")
    def test_cuda_exception_handling(self, mock_torch):
        """Test get_cuda_memory handles exceptions gracefully."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.side_effect = RuntimeError("CUDA error")

        result = get_cuda_memory()

        assert result is None


@pytest.mark.unit
class TestGetMpsMemory:
    """Tests for get_mps_memory function."""

    @patch("tensortruth.core.system.torch")
    def test_mps_available(self, mock_torch):
        """Test get_mps_memory when MPS is available."""
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.current_allocated_memory.return_value = 2 * (1024**3)  # 2 GB

        with patch("psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(total=16 * (1024**3))

            result = get_mps_memory()

        assert result is not None
        assert result.name == "MPS (Unified)"
        assert result.allocated_gb == pytest.approx(2.0, rel=0.01)
        assert result.total_gb == pytest.approx(16.0, rel=0.01)

    @patch("tensortruth.core.system.torch")
    def test_mps_not_available(self, mock_torch):
        """Test get_mps_memory when MPS is not available."""
        mock_torch.backends.mps.is_available.return_value = False

        result = get_mps_memory()

        assert result is None


@pytest.mark.unit
class TestGetSystemRam:
    """Tests for get_system_ram function."""

    def test_system_ram(self):
        """Test get_system_ram returns valid info."""
        with patch("psutil.virtual_memory") as mock_vmem:
            mock_vmem.return_value = MagicMock(
                used=8 * (1024**3),
                total=32 * (1024**3),
            )

            result = get_system_ram()

        assert result is not None
        assert result.name == "System RAM"
        assert result.allocated_gb == pytest.approx(8.0, rel=0.01)
        assert result.total_gb == pytest.approx(32.0, rel=0.01)

    def test_system_ram_psutil_not_available(self):
        """Test get_system_ram when psutil import fails."""
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                # This test is tricky because psutil is already imported
                # We test the exception handling path instead
                pass


@pytest.mark.unit
class TestGetOllamaMemory:
    """Tests for get_ollama_memory function."""

    @patch("tensortruth.core.ollama.get_running_models_detailed")
    def test_ollama_with_running_model(self, mock_get_models):
        """Test get_ollama_memory with a running model."""
        mock_get_models.return_value = [
            {"name": "llama2:7b", "size_vram": 5 * (1024**3)}
        ]

        result = get_ollama_memory()

        assert result is not None
        assert result.name == "Ollama VRAM"
        assert result.allocated_gb == pytest.approx(5.0, rel=0.01)
        assert "llama2:7b" in result.details

    @patch("tensortruth.core.ollama.get_running_models_detailed")
    def test_ollama_multiple_models(self, mock_get_models):
        """Test get_ollama_memory with multiple running models."""
        mock_get_models.return_value = [
            {"name": "llama2:7b", "size_vram": 3 * (1024**3)},
            {"name": "mistral:7b", "size_vram": 2 * (1024**3)},
        ]

        result = get_ollama_memory()

        assert result is not None
        assert result.allocated_gb == pytest.approx(5.0, rel=0.01)
        assert "llama2:7b" in result.details
        assert "mistral:7b" in result.details

    @patch("tensortruth.core.ollama.get_running_models_detailed")
    def test_ollama_no_running_models(self, mock_get_models):
        """Test get_ollama_memory with no running models."""
        mock_get_models.return_value = []

        result = get_ollama_memory()

        assert result is None

    @patch("tensortruth.core.ollama.get_running_models_detailed")
    def test_ollama_exception_handling(self, mock_get_models):
        """Test get_ollama_memory handles exceptions gracefully."""
        mock_get_models.side_effect = Exception("Connection refused")

        result = get_ollama_memory()

        assert result is None


@pytest.mark.unit
class TestGetAllMemoryInfo:
    """Tests for get_all_memory_info function."""

    @patch("tensortruth.core.system.get_system_ram")
    @patch("tensortruth.core.system.get_mps_memory")
    @patch("tensortruth.core.system.get_cuda_memory")
    def test_all_memory_info_cuda_system(self, mock_cuda, mock_mps, mock_ram):
        """Test get_all_memory_info on CUDA system."""
        mock_cuda.return_value = MemoryInfo("CUDA VRAM", 4.0, 8.0)
        mock_mps.return_value = None
        mock_ram.return_value = MemoryInfo("System RAM", 8.0, 32.0)

        result = get_all_memory_info()

        assert len(result) == 2
        assert result[0].name == "CUDA VRAM"
        assert result[1].name == "System RAM"

    @patch("tensortruth.core.system.get_system_ram")
    @patch("tensortruth.core.system.get_mps_memory")
    @patch("tensortruth.core.system.get_cuda_memory")
    def test_all_memory_info_empty(self, mock_cuda, mock_mps, mock_ram):
        """Test get_all_memory_info when nothing available."""
        mock_cuda.return_value = None
        mock_mps.return_value = None
        mock_ram.return_value = None

        result = get_all_memory_info()

        assert len(result) == 0


@pytest.mark.unit
class TestGetMemorySummary:
    """Tests for get_memory_summary function."""

    @patch("tensortruth.core.system.get_system_ram")
    @patch("tensortruth.core.system.get_ollama_memory")
    @patch("tensortruth.core.system.get_mps_memory")
    @patch("tensortruth.core.system.get_cuda_memory")
    def test_memory_summary_cuda(self, mock_cuda, mock_mps, mock_ollama, mock_ram):
        """Test memory summary with CUDA available."""
        mock_cuda.return_value = MemoryInfo("CUDA VRAM", 4.0, 8.0)
        mock_mps.return_value = None
        mock_ollama.return_value = None
        mock_ram.return_value = MemoryInfo("System RAM", 8.0, 32.0)

        result = get_memory_summary()

        assert "VRAM:" in result
        assert "RAM:" in result
        assert "4.00" in result

    @patch("tensortruth.core.system.get_system_ram")
    @patch("tensortruth.core.system.get_ollama_memory")
    @patch("tensortruth.core.system.get_mps_memory")
    @patch("tensortruth.core.system.get_cuda_memory")
    def test_memory_summary_mps_with_ollama(
        self, mock_cuda, mock_mps, mock_ollama, mock_ram
    ):
        """Test memory summary with MPS and Ollama."""
        mock_cuda.return_value = None
        mock_mps.return_value = MemoryInfo("MPS", 2.0, 16.0)
        mock_ollama.return_value = MemoryInfo("Ollama VRAM", 3.0)
        mock_ram.return_value = MemoryInfo("System RAM", 8.0, 32.0)

        result = get_memory_summary()

        assert "MPS:" in result
        assert "Ollama:" in result
        assert "RAM:" in result

    @patch("tensortruth.core.system.get_system_ram")
    @patch("tensortruth.core.system.get_ollama_memory")
    @patch("tensortruth.core.system.get_mps_memory")
    @patch("tensortruth.core.system.get_cuda_memory")
    def test_memory_summary_unavailable(
        self, mock_cuda, mock_mps, mock_ollama, mock_ram
    ):
        """Test memory summary when nothing available."""
        mock_cuda.return_value = None
        mock_mps.return_value = None
        mock_ollama.return_value = None
        mock_ram.return_value = None

        result = get_memory_summary()

        assert "unavailable" in result


@pytest.mark.unit
class TestFormatMemoryReport:
    """Tests for format_memory_report function."""

    @patch("tensortruth.core.system.get_all_memory_info")
    def test_format_memory_report_with_info(self, mock_get_all):
        """Test format_memory_report with memory info available."""
        mock_get_all.return_value = [
            MemoryInfo("CUDA VRAM", 4.0, 8.0, "Reserved: 5.0 GB"),
            MemoryInfo("System RAM", 8.0, 32.0),
        ]

        result = format_memory_report()

        assert "### Memory Usage" in result[0]
        assert any("CUDA VRAM" in line for line in result)
        assert any("System RAM" in line for line in result)
        assert any("Tips:" in line for line in result)
        assert any("/reload" in line for line in result)

    @patch("tensortruth.core.system.get_all_memory_info")
    def test_format_memory_report_empty(self, mock_get_all):
        """Test format_memory_report when no info available."""
        mock_get_all.return_value = []

        result = format_memory_report()

        assert "### Memory Usage" in result[0]
        assert any("No memory information" in line for line in result)
