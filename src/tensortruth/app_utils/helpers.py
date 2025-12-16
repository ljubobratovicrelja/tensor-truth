"""General helper functions for the Streamlit app."""

import asyncio
import gc
import os
import tarfile
import time
from typing import List

import aiohttp
import torch


def _download_and_extract_indexes(index_dir: str, gdrive_link: str):
    """
    Check if indexes directory is empty or missing.
    If so, download tarball from Google Drive, extract it, and clean up.
    Returns True if download was needed and successful.
    """
    # Check if indexes directory exists and has content
    needs_download = False

    if not os.path.exists(index_dir):
        needs_download = True
        os.makedirs(index_dir, exist_ok=True)
    elif not os.listdir(index_dir):
        needs_download = True

    if not needs_download:
        return False

    tarball_path = "indexes.tar"

    try:
        # Check if gdown is available
        try:
            import gdown
        except ImportError:
            raise ImportError(
                "gdown library not installed. Install with: pip install gdown"
            )

        # Download using gdown (handles Google Drive's quirks automatically)
        gdown.download(gdrive_link, tarball_path, quiet=False, fuzzy=True)

        # Extract tarball to root directory (tar already contains indexes/ folder)
        with tarfile.open(tarball_path, "r:") as tar:
            tar.extractall(path=".")

        # Clean up tarball
        os.remove(tarball_path)

        return True

    except Exception as e:
        # Clean up partial download
        if os.path.exists(tarball_path):
            os.remove(tarball_path)
        raise e


def get_random_generating_message():
    """Returns a random generating message."""

    messages = [
        "✍️ Generating response...",
        "💬 Crafting message...",
        "📝 Writing reply...",
        "🔄 Building answer...",
        "⏳ Composing...",
        "🧐 Putting words together...",
        "💡 Formulating response...",
        "🔍 Assembling output...",
        "📊 Constructing reply...",
        "✨ Creating response...",
    ]
    return messages[int(time.time()) % len(messages)]


def get_random_rag_processing_message():
    """Returns a random RAG processing message."""

    messages = [
        "🔍 Consulting the knowledge base...",
        "📚 Retrieving relevant information...",
        "🧠 Analyzing documents for context...",
        "🔎 Searching indexed data...",
        "✍️ Formulating a response based on sources...",
        "📖 Reviewing materials to assist...",
        "💡 Synthesizing information from the knowledge base...",
        "📝 Compiling insights from documents...",
        "🔗 Connecting the dots from indexed content...",
        "🧩 Piecing together relevant information...",
    ]
    return messages[int(time.time()) % len(messages)]


def download_indexes_with_ui(index_dir: str, gdrive_link: str):
    """
    Wrapper for download_and_extract_indexes that provides Streamlit UI feedback.
    """
    import streamlit as st

    try:
        with st.spinner(
            "📥 Downloading indexes from Google Drive (this may take a few minutes)..."
        ):
            success = _download_and_extract_indexes(index_dir, gdrive_link)
            if success:
                st.success("✅ Indexes downloaded and extracted successfully!")
    except ImportError as e:
        st.warning(f"⚠️ {str(e)}")
    except Exception as e:
        st.error(f"❌ Error downloading/extracting indexes: {e}")


def get_available_modules(index_dir: str):
    """Get list of available index modules."""
    if not os.path.exists(index_dir):
        return []
    return sorted(
        [d for d in os.listdir(index_dir) if os.path.isdir(os.path.join(index_dir, d))]
    )


# Cache decorator will be applied by Streamlit app if streamlit is available
try:
    import streamlit as st

    get_available_modules = st.cache_data(ttl=10)(get_available_modules)
except ImportError:
    pass


async def get_ollama_models_async():
    """Fetches list of available models from local Ollama instance (async version)."""
    try:
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["name"] for m in data["models"]]
                    return sorted(models)
    except Exception:
        pass
    return ["deepseek-r1:8b"]


async def get_ollama_ps_async():
    """Fetches running model information from Ollama (async version)."""
    try:
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("http://localhost:11434/api/ps") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
    except Exception:
        pass
    return []


def get_ollama_ps():
    """Fetches running model information from Ollama (sync wrapper)."""
    return asyncio.run(get_ollama_ps_async())


def get_ollama_models():
    """Fetches list of available models from local Ollama instance (sync wrapper)."""
    return asyncio.run(get_ollama_models_async())


# Cache decorator will be applied by Streamlit app if streamlit is available
try:
    import streamlit as st

    get_ollama_models = st.cache_data(ttl=60)(get_ollama_models)
except ImportError:
    pass


def get_system_devices():
    """Returns list of available compute devices."""
    devices = ["cpu"]
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.insert(0, "mps")
    # Check CUDA
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    return devices


def free_memory():
    """Free GPU/MPS memory by clearing caches."""
    import streamlit as st

    if "engine" in st.session_state:
        del st.session_state["engine"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ensure_engine_loaded(target_modules, target_params):
    """Ensure the RAG engine is loaded with the specified configuration."""
    import streamlit as st

    from tensortruth import load_engine_for_modules

    target_tuple = tuple(sorted(target_modules))
    param_items = sorted([(k, v) for k, v in target_params.items()])
    param_hash = frozenset(param_items)

    current_config = st.session_state.get("loaded_config")

    if current_config == (target_tuple, param_hash):
        return st.session_state.engine

    # Always show loading message for better UX
    placeholder = st.empty()
    placeholder.info(
        f"⏳ Loading Model: {target_params.get('model')} | "
        f"Pipeline: {target_params.get('rag_device')} | "
        f"LLM: {target_params.get('llm_device')}..."
    )

    if current_config is not None:
        free_memory()

    try:
        engine = load_engine_for_modules(list(target_tuple), target_params)
        st.session_state.engine = engine
        st.session_state.loaded_config = (target_tuple, param_hash)
        placeholder.empty()
        return engine
    except Exception as e:
        placeholder.error(f"Failed: {e}")
        st.stop()


def format_ollama_runtime_info() -> List[str]:
    """
    Get formatted Ollama runtime information.

    Returns:
        List of formatted strings describing running models, or empty list if unavailable.
    """
    lines = []
    try:
        running_models = get_ollama_ps()
        if running_models:
            for model_info in running_models:
                model_name = model_info.get("name", "Unknown")
                size_vram = model_info.get("size_vram", 0)
                size = model_info.get("size", 0)

                # Convert bytes to GB for readability
                size_vram_gb = size_vram / (1024**3) if size_vram else 0
                size_gb = size / (1024**3) if size else 0

                lines.append(f"**Running:** `{model_name}`")
                if size_vram_gb > 0:
                    lines.append(f"**VRAM:** `{size_vram_gb:.2f} GB`")
                if size_gb > 0:
                    lines.append(f"**Model Size:** `{size_gb:.2f} GB`")

                processor = model_info.get("details", {}).get("parameter_size", "")
                if processor:
                    lines.append(f"**Parameters:** `{processor}`")
    except Exception:
        pass

    return lines
