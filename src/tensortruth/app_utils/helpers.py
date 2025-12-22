"""General helper functions for the Streamlit app."""

import gc
import os
import tarfile
import time
from pathlib import Path
from typing import List, Optional, Union

import torch

# Constants for the Tensor Truth Indexes
HF_REPO_ID = "ljubobratovicrelja/tensor-truth-indexes"
HF_FILENAME = "indexes_v0.1.12.tar"


def get_module_display_name(
    index_dir: Union[str, Path], module_name: str
) -> tuple[str, str, str]:
    """Extract display_name and category from module's ChromaDB index.

    Args:
        index_dir: Base index directory
        module_name: Module folder name

    Returns:
        Tuple of (display_name, doc_type, category_prefix) where:
        - display_name: Human-readable name
        - doc_type: Type from metadata (book, paper, library_doc, etc.)
        - category_prefix: Formatted prefix for grouping (e.g., "📚 Books")
    """
    try:
        import re

        # Disable ChromaDB telemetry to suppress info messages
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        import chromadb

        index_path = Path(index_dir) / module_name
        client = chromadb.PersistentClient(path=index_path)

        # Try to get the collection (LlamaIndex uses 'data')
        collection = client.get_collection("data")

        # Peek at first document to get display_name and doc_type
        results = collection.peek(limit=1)
        if results["metadatas"] and len(results["metadatas"]) > 0:
            metadata = results["metadatas"][0]
            display_name = metadata.get("display_name")
            doc_type = metadata.get("doc_type", "unknown")

            if display_name:
                # Remove chapter info like "Ch.01", "Ch.1-3", etc.
                # Pattern: "Ch." followed by numbers/dashes and a separator
                display_name = re.sub(r"\s+Ch\.\s*[\d\-]+\s*-\s*", " - ", display_name)

                # Determine category prefix based on doc_type
                category_map = {
                    "book": ("📚 Books", 1),
                    "paper": ("📄 Papers", 2),
                    "library_doc": ("📦 Libraries", 3),
                }
                category_prefix, sort_order = category_map.get(
                    doc_type, ("📁 Other", 4)
                )

                return display_name, doc_type, category_prefix, sort_order
    except Exception:
        # ChromaDB read failed, use module_name as fallback
        pass

    # Fallback: use raw module_name with unknown category
    return module_name, "unknown", "📁 Other", 4


def download_and_extract_indexes(
    user_dir: Union[str, Path],
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
) -> bool:
    """
    Downloads and extracts index files from a Hugging Face repository.

    Args:
        user_dir: The local directory where the indexes should be extracted.
        repo_id: The Hugging Face repository ID (e.g., 'username/repo-name').
        filename: The specific tarball filename to download.

    Returns:
        True if the download and extraction were successful.

    Raises:
        Exception: If the download or extraction process fails.
    """
    import shutil

    from huggingface_hub import hf_hub_download

    user_dir = Path(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)

    tarball_path: Optional[Path] = None
    HF_REPO_TYPE = "dataset"

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            local_dir=user_dir,
            local_dir_use_symlinks=False,
        )

        tarball_path = Path(downloaded_file)

        # Extracting the tarball.
        with tarfile.open(tarball_path, "r:") as tar:
            tar.extractall(path=user_dir)

        return True

    except Exception as e:
        raise e
    finally:
        # Remove the tarball after successful extraction.
        if tarball_path.exists():
            tarball_path.unlink()

        # Remove Hugging Face cache directory.
        hf_cache_dir = user_dir / ".cache"
        if hf_cache_dir.exists() and hf_cache_dir.is_dir():
            shutil.rmtree(hf_cache_dir)


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


def download_indexes_with_ui(
    user_dir: Union[str, Path],
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
):
    """
    Wrapper for download_and_extract_indexes that provides Streamlit UI feedback.
    """
    import streamlit as st

    try:
        with st.spinner(
            "📥 Downloading indexes from HuggingFace Hub (this may take a few minutes)..."
        ):
            success = download_and_extract_indexes(
                user_dir, repo_id=repo_id, filename=filename
            )
            if success:
                st.success("✅ Indexes downloaded and extracted successfully!")
    except ImportError as e:
        st.warning(f"⚠️ {str(e)}")
    except Exception as e:
        st.error(f"❌ Error downloading/extracting indexes: {e}")


def get_available_modules(index_dir: Union[str, Path]):
    """Get list of available modules with categorized display names.

    Args:
        index_dir: Base index directory (str or Path)

    Returns:
        List of tuples: [(module_name, formatted_display_name), ...]
        where formatted_display_name includes category prefix for grouping
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        return []

    # Get current list of module directories
    module_dirs = sorted([d.name for d in index_dir.iterdir() if d.is_dir()])

    # For each module, get display name and category from ChromaDB metadata
    results = []
    for module_name in module_dirs:
        display_name, doc_type, category_prefix, sort_order = get_module_display_name(
            index_dir, module_name
        )
        # Format: "📚 Books › Linear Algebra - Cherney"
        formatted_name = f"{category_prefix} › {display_name}"
        results.append((module_name, formatted_name, sort_order))

    # Sort by category first (sort_order), then by display name
    results.sort(key=lambda x: (x[2], x[1]))

    # Return just module_name and formatted_name (drop sort_order)
    return [(mod, name) for mod, name, _ in results]


# Cache decorator will be applied by Streamlit app if streamlit is available
try:
    import streamlit as st

    get_available_modules = st.cache_data(ttl=10)(get_available_modules)
except ImportError:
    pass


def get_ollama_models():
    """Fetches list of available models from local Ollama instance."""
    from tensortruth.core.ollama import get_available_models

    return get_available_models()


def get_ollama_ps():
    """Fetches running model information from Ollama."""
    from tensortruth.core.ollama import get_running_models_detailed

    return get_running_models_detailed()


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


def free_memory(engine=None):
    """Free GPU/MPS memory by clearing caches.

    Args:
        engine: Optional engine reference to delete. If None, will try to clean
                up from st.session_state if streamlit is available.
    """
    # If engine provided, delete it
    if engine is not None:
        del engine

    # Also try to clean up from streamlit session_state if available
    try:
        import streamlit as st

        if "engine" in st.session_state:
            del st.session_state["engine"]
    except (ImportError, AttributeError):
        # Streamlit not available or session_state not initialized
        pass

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
