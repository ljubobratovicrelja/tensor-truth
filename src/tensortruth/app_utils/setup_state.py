"""Setup mode session state management."""

import streamlit as st

from tensortruth.app_utils.helpers import get_system_devices


def init_setup_defaults_from_config():
    """Initialize setup mode defaults from config file.

    This should be called once when entering setup mode for the first time.
    Uses config.yaml as the single source of truth for default values.
    """
    if (
        "session_parameters_initialized" in st.session_state
        and st.session_state.session_parameters_initialized
    ):
        return

    # Use cached config from session_state (loaded once in init_app_state)
    config = st.session_state.config
    system_devices = get_system_devices()

    # Knowledge base - start with empty selection
    st.session_state.setup_mods = []

    # Model selection defaults
    st.session_state.setup_model = None  # Will be set from available models
    st.session_state.setup_ctx = config.ui.default_context_window
    st.session_state.setup_temp = config.ui.default_temperature
    st.session_state.setup_max_tokens = config.ui.default_max_tokens

    # Thinking model controls
    st.session_state.setup_thinking_level = config.ui.default_thinking_level
    st.session_state.setup_thinking_top_k = config.ui.default_thinking_top_k
    st.session_state.setup_thinking_stop_sequences = (
        config.ui.default_thinking_stop_sequences
    )

    # RAG parameters
    st.session_state.setup_reranker = config.ui.default_reranker
    st.session_state.setup_top_n = config.ui.default_top_n
    st.session_state.setup_conf = config.ui.default_confidence_threshold
    st.session_state.setup_conf_cutoff_hard = config.ui.default_confidence_cutoff_hard
    st.session_state.setup_sys_prompt = ""

    # Hardware allocation - smart defaults based on available devices
    if "mps" in system_devices:
        st.session_state.setup_rag_device = "mps"
        st.session_state.setup_llm_device = "gpu"
    elif "cuda" in system_devices:
        st.session_state.setup_rag_device = config.rag.default_device
        st.session_state.setup_llm_device = "gpu"
    else:
        st.session_state.setup_rag_device = "cpu"
        st.session_state.setup_llm_device = "cpu"

    st.session_state.session_parameters_initialized = True


def build_params_from_session_state() -> dict:
    """Build parameters dict from current session_state setup values.

    Returns:
        Dict with all parameters needed for session creation or preset saving.
    """
    # Parse thinking_level - convert "disabled" to False for the engine
    thinking_level = st.session_state.setup_thinking_level
    if thinking_level == "disabled":
        thinking_level = False

    # Parse stop sequences from comma-separated string
    stop_sequences = [
        s.strip().replace("\\n", "\n")
        for s in st.session_state.setup_thinking_stop_sequences.split(",")
        if s.strip()
    ]

    return {
        "model": st.session_state.setup_model,
        "temperature": st.session_state.setup_temp,
        "context_window": st.session_state.setup_ctx,
        "max_tokens": st.session_state.setup_max_tokens,
        "thinking_level": thinking_level,
        "thinking_top_k": st.session_state.setup_thinking_top_k,
        "thinking_stop_sequences": stop_sequences,
        "system_prompt": st.session_state.setup_sys_prompt,
        "reranker_model": st.session_state.setup_reranker,
        "reranker_top_n": st.session_state.setup_top_n,
        "confidence_cutoff": st.session_state.setup_conf,
        "confidence_cutoff_hard": st.session_state.setup_conf_cutoff_hard,
        "rag_device": st.session_state.setup_rag_device,
        "llm_device": st.session_state.setup_llm_device,
    }


def get_session_params_with_defaults(session_params: dict) -> dict:
    """Get session parameters with config defaults as fallback.

    Args:
        session_params: Parameters dict from session (may be incomplete)

    Returns:
        Complete parameters dict with defaults filled in
    """
    # Use cached config from session_state (loaded once in init_app_state)
    config = st.session_state.config

    # Handle thinking_stop_sequences - may be stored as list or string
    default_stop_seq = config.ui.default_thinking_stop_sequences
    thinking_stop_sequences = session_params.get(
        "thinking_stop_sequences", default_stop_seq
    )
    if isinstance(thinking_stop_sequences, str):
        thinking_stop_sequences = [
            s.strip().replace("\\n", "\n")
            for s in thinking_stop_sequences.split(",")
            if s.strip()
        ]

    return {
        "model": session_params.get("model", "deepseek-r1:8b"),
        "temperature": session_params.get("temperature", config.ui.default_temperature),
        "context_window": session_params.get(
            "context_window", config.ui.default_context_window
        ),
        "max_tokens": session_params.get("max_tokens", config.ui.default_max_tokens),
        "thinking_level": session_params.get(
            "thinking_level", config.ui.default_thinking_level
        ),
        "thinking_top_k": session_params.get(
            "thinking_top_k", config.ui.default_thinking_top_k
        ),
        "thinking_stop_sequences": thinking_stop_sequences,
        "confidence_cutoff": session_params.get(
            "confidence_cutoff", config.ui.default_confidence_threshold
        ),
        "confidence_cutoff_hard": session_params.get(
            "confidence_cutoff_hard", config.ui.default_confidence_cutoff_hard
        ),
        "system_prompt": session_params.get("system_prompt", ""),
        "reranker_model": session_params.get(
            "reranker_model", config.ui.default_reranker
        ),
        "reranker_top_n": session_params.get("reranker_top_n", config.ui.default_top_n),
        "rag_device": session_params.get("rag_device", config.rag.default_device),
        "llm_device": session_params.get("llm_device", "gpu"),
    }
