"""VRAM monitoring and estimation utilities."""

import streamlit as st
import torch


@st.cache_data(ttl=2, show_spinner=False)
def get_vram_breakdown():
    """
    Returns detailed VRAM stats:
    - total_used: What Task Manager says
    - reclaimable: What we can kill (Ollama + PyTorch)
    - baseline: What stays (OS, Browser, Display)
    """
    if not torch.cuda.is_available():
        # Heuristic for non-CUDA systems (like Mac)
        return {"total_used": 0.0, "reclaimable": 0.0, "baseline": 4.0}

    try:
        from tensortruth.core import get_running_models

        # 1. Real Hardware Usage (Everything on the card)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_used_gb = (total_bytes - free_bytes) / (1024**3)

        # 2. PyTorch Reserved (What THIS python process holds)
        torch_reserved_gb = torch.cuda.memory_reserved() / (1024**3)

        # 3. Ollama Usage (External process)
        ollama_usage_gb = 0.0
        active_models = get_running_models()
        for m in active_models:
            try:
                size_str = m.get("size_vram", "0 GB").split()[0]
                ollama_usage_gb += float(size_str)
            except Exception:
                pass

        reclaimable = torch_reserved_gb + ollama_usage_gb
        baseline = max(0.5, total_used_gb - reclaimable)

        return {
            "total_used": total_used_gb,
            "reclaimable": reclaimable,
            "baseline": baseline,
        }
    except Exception:
        return {"total_used": 0.0, "reclaimable": 0.0, "baseline": 2.5}


def estimate_vram_usage(
    model_name, num_indices, context_window, rag_device, llm_device
):
    """
    Returns (predicted_total, breakdown_dict, new_session_cost)
    """
    stats = get_vram_breakdown()
    system_baseline = stats["baseline"]

    # --- RAG COST ---
    # 1.8GB if running on GPU (CUDA) or MPS (Unified Memory counts as VRAM usage)
    if rag_device in ["cuda", "mps"]:
        rag_overhead = 1.8
    else:
        rag_overhead = 0.0  # CPU RAM

    index_overhead = num_indices * 0.15

    # --- LLM COST ---
    if llm_device == "cpu":
        llm_size = 0.0
    else:
        name = model_name.lower() if model_name else ""
        if "70b" in name:
            llm_size = 40.0
        elif "32b" in name:
            llm_size = 19.0
        elif "14b" in name:
            llm_size = 9.5
        elif "8b" in name:
            llm_size = 5.5
        elif "7b" in name:
            llm_size = 5.0
        elif "1.5b" in name:
            llm_size = 1.5
        else:
            llm_size = 6.0

    # KV Cache (Linear Approx)
    kv_cache = (context_window / 4096) * 0.8
    # KV Cache also moves to RAM if LLM is on CPU
    if llm_device == "cpu":
        kv_cache = 0.0

    new_session_cost = rag_overhead + index_overhead + llm_size + kv_cache
    predicted_total = system_baseline + new_session_cost

    return predicted_total, stats, new_session_cost


def render_vram_gauge(
    model_name, num_indices, context_window, rag_device, llm_device, max_vram_gb
):
    """Render VRAM usage gauge in Streamlit UI."""
    predicted, stats, new_cost = estimate_vram_usage(
        model_name, num_indices, context_window, rag_device, llm_device
    )
    vram_percent = min(predicted / max_vram_gb, 1.0)

    current_used = stats["total_used"]
    reclaimable = stats["reclaimable"]

    # Visual Layout
    st.markdown("##### VRAM Status")

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Load", f"{current_used:.1f} GB", delta_color="off")
    m2.metric(
        "Reclaimable",
        f"{reclaimable:.1f} GB",
        help="VRAM from Ollama/Torch that will be freed.",
        delta_color="normal",
    )
    m3.metric(
        "Predicted Peak",
        f"{predicted:.1f} GB",
        delta=(
            f"{predicted - max_vram_gb:.1f} GB" if predicted > max_vram_gb else "Safe"
        ),
        delta_color="inverse",
    )

    st.progress(vram_percent)

    if predicted > max_vram_gb:
        st.error(
            f"🛑 Configuration ({predicted:.1f} GB) exceeds limit ({max_vram_gb} GB)."
        )
    elif predicted > (max_vram_gb * 0.9):
        st.warning("⚠️ High VRAM usage predicted.")

    return predicted
