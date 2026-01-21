"""General helper functions for the Streamlit app."""

import gc
import json
import logging
import os
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from tensortruth.indexing.metadata import sanitize_model_id

logger = logging.getLogger(__name__)

# Constants for the Tensor Truth Indexes
HF_REPO_ID = "ljubobratovicrelja/tensor-truth-indexes"
HF_FILENAME = "indexes_v0.1.14.tar"
HF_MANIFEST_FILENAME = "manifest.json"

# Default embedding model for backwards compatibility
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def get_module_display_name(
    index_dir: Union[str, Path], module_name: str
) -> tuple[str, str, str, int]:
    """Extract display_name and category from module's ChromaDB index.

    Args:
        index_dir: Base index directory
        module_name: Module folder name

    Returns:
        Tuple of (display_name, doc_type, category_prefix, sort_order) where:
        - display_name: Human-readable name
        - doc_type: Type from metadata (book, paper, library_doc, etc.)
        - category_prefix: Formatted prefix for grouping (e.g., "📚 Books")
        - sort_order: Integer for sorting (1-4)
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
            doc_type = str(metadata.get("doc_type", "unknown"))

            # Prioritize group/book display names for UI (same across all items in group/book)
            # Otherwise use individual display_name (for libraries, uploaded PDFs)
            display_name = (
                metadata.get("group_display_name")  # For paper groups
                or metadata.get("book_display_name")  # For books
                or metadata.get("library_display_name")  # For libraries
                or metadata.get("display_name")  # Fallback for individual items
            )

            if display_name:
                # Ensure display_name is a string
                display_name = str(display_name)
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


def get_hf_manifest(
    repo_id: str = HF_REPO_ID,
) -> Optional[Dict[str, Any]]:
    """Fetch the index manifest from HuggingFace Hub.

    The manifest lists available embedding model indexes and their tarballs.

    Args:
        repo_id: The HuggingFace repository ID.

    Returns:
        Manifest dict if successful, None otherwise.

    Manifest schema:
        {
            "default_embedding_model": "BAAI/bge-m3",
            "available_indexes": [
                {
                    "embedding_model": "BAAI/bge-m3",
                    "embedding_model_id": "bge-m3",
                    "filename": "indexes_bge-m3_v0.2.0.tar",
                    "version": "0.2.0"
                },
                ...
            ]
        }
    """
    try:
        from huggingface_hub import hf_hub_download

        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename=HF_MANIFEST_FILENAME,
            repo_type="dataset",
        )

        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        logger.warning(f"Failed to fetch manifest from {repo_id}: {e}")
        return None


def get_available_hf_embedding_indexes(
    repo_id: str = HF_REPO_ID,
) -> List[Dict[str, Any]]:
    """Get list of available embedding indexes from HuggingFace Hub.

    Args:
        repo_id: The HuggingFace repository ID.

    Returns:
        List of available index info dicts with keys:
        - embedding_model: Full model name (e.g., "BAAI/bge-m3")
        - embedding_model_id: Sanitized ID (e.g., "bge-m3")
        - filename: Tarball filename to download
        - version: Index version string
    """
    manifest = get_hf_manifest(repo_id)
    if manifest:
        return manifest.get("available_indexes", [])

    # Fallback to legacy single-file format if manifest not available
    return [
        {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "embedding_model_id": sanitize_model_id(DEFAULT_EMBEDDING_MODEL),
            "filename": HF_FILENAME,
            "version": "legacy",
        }
    ]


def get_filename_for_embedding_model(
    embedding_model: str,
    repo_id: str = HF_REPO_ID,
) -> Optional[str]:
    """Get the tarball filename for a specific embedding model.

    Args:
        embedding_model: The embedding model name (e.g., "BAAI/bge-m3")
        repo_id: The HuggingFace repository ID.

    Returns:
        Tarball filename if available, None otherwise.
    """
    model_id = sanitize_model_id(embedding_model)
    available = get_available_hf_embedding_indexes(repo_id)

    for entry in available:
        if entry.get("embedding_model_id") == model_id:
            return entry.get("filename")

    return None


def download_and_extract_indexes(
    user_dir: Union[str, Path],
    repo_id: str = HF_REPO_ID,
    filename: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> bool:
    """Downloads and extracts index files from a Hugging Face repository.

    Supports both legacy single-file downloads and new embedding-model-specific
    downloads via manifest. Extracts to versioned path: indexes/{model_id}/

    Args:
        user_dir: The local directory where the indexes should be extracted.
        repo_id: The Hugging Face repository ID (e.g., 'username/repo-name').
        filename: The specific tarball filename to download (optional if
                  embedding_model is provided).
        embedding_model: The embedding model to download indexes for. If provided,
                        the appropriate filename will be looked up from manifest.

    Returns:
        True if the download and extraction were successful.

    Raises:
        ValueError: If neither filename nor embedding_model provided.
        Exception: If the download or extraction process fails.
    """
    import shutil

    from huggingface_hub import hf_hub_download

    # Determine filename and model_id
    model_id = sanitize_model_id(embedding_model or DEFAULT_EMBEDDING_MODEL)

    if filename is None:
        if embedding_model is not None:
            filename = get_filename_for_embedding_model(embedding_model, repo_id)
            if filename is None:
                logger.warning(
                    f"No index tarball found for embedding model: {embedding_model}. "
                    f"Falling back to default: {HF_FILENAME}"
                )
                filename = HF_FILENAME
        else:
            filename = HF_FILENAME

    user_dir = Path(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)

    tarball_path: Optional[Path] = None
    HF_REPO_TYPE = "dataset"

    try:
        logger.info(f"Downloading {filename} from {repo_id}...")
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            local_dir=user_dir,
        )

        tarball_path = Path(downloaded_file)

        # Extract to temp directory first
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Extracting {tarball_path}...")
            with tarfile.open(tarball_path, "r:") as tar:
                tar.extractall(path=temp_path)

            # Move extracted indexes to versioned path: indexes/{model_id}/
            extracted_indexes = temp_path / "indexes"
            if extracted_indexes.exists():
                target_dir = user_dir / "indexes" / model_id
                target_dir.mkdir(parents=True, exist_ok=True)

                # Move each module directory to the versioned path
                for module_dir in extracted_indexes.iterdir():
                    if module_dir.is_dir():
                        dest = target_dir / module_dir.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(module_dir), str(dest))
                        logger.info(f"  Extracted: {module_dir.name}")

                logger.info(
                    f"Index download complete: {target_dir} "
                    f"({len(list(target_dir.iterdir()))} modules)"
                )
            else:
                logger.warning("No indexes directory found in tarball")
                return False

        return True

    finally:
        # Clean up the downloaded tarball file.
        if tarball_path is not None and tarball_path.exists():
            tarball_path.unlink()

        # ... and the Hugging Face cache directory.
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
    filename: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """Wrapper for download_and_extract_indexes that provides Streamlit UI feedback.

    Args:
        user_dir: Directory to extract indexes to.
        repo_id: HuggingFace repository ID.
        filename: Specific tarball to download (optional if embedding_model provided).
        embedding_model: Embedding model to download indexes for.
    """
    import streamlit as st

    model_info = f" for {embedding_model}" if embedding_model else ""

    try:
        with st.spinner(
            f"📥 Downloading indexes{model_info} from HuggingFace Hub "
            f"(this may take a few minutes)..."
        ):
            success = download_and_extract_indexes(
                user_dir,
                repo_id=repo_id,
                filename=filename,
                embedding_model=embedding_model,
            )
            if success:
                st.success("✅ Indexes downloaded and extracted successfully!")
    except Exception as e:
        st.error(f"❌ Error downloading/extracting indexes: {e}")
        display_filename = filename or HF_FILENAME
        hf_link = (
            f"https://huggingface.co/datasets/{repo_id}/blob/main/{display_filename}"
        )
        st.info(f"Try fetching manually from: {hf_link}, and storing in: {user_dir}")


def get_available_modules(
    index_dir: Union[str, Path],
    embedding_model: Optional[str] = None,
):
    """Get list of available modules with categorized display names.

    Supports both versioned structure (indexes/{model_id}/{module}/) and
    legacy flat structure (indexes/{module}/) for backward compatibility.

    Args:
        index_dir: Base index directory (str or Path)
        embedding_model: Optional embedding model to filter by. If provided,
                        looks for modules in indexes/{model_id}/

    Returns:
        List of tuples: [(module_name, formatted_display_name), ...]
        where formatted_display_name includes category prefix for grouping
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        return []

    # Determine which directories to look in
    search_dirs = []

    if embedding_model:
        # Look in versioned structure first
        model_id = sanitize_model_id(embedding_model)
        versioned_dir = index_dir / model_id
        if versioned_dir.exists():
            search_dirs.append(versioned_dir)

    # Also check legacy flat structure for backward compatibility
    # But only include modules that are valid indexes (not model ID directories)
    for item in index_dir.iterdir():
        if item.is_dir():
            # Check if this is a valid ChromaDB index (has chroma.sqlite3)
            if (item / "chroma.sqlite3").exists():
                # This is a legacy flat structure module
                if item not in search_dirs:
                    search_dirs.append(index_dir)
                    break

    # If no embedding model specified, search all model directories
    if not embedding_model:
        for item in index_dir.iterdir():
            if item.is_dir() and not (item / "chroma.sqlite3").exists():
                # This looks like a model ID directory
                if any(
                    (sub / "chroma.sqlite3").exists()
                    for sub in item.iterdir()
                    if sub.is_dir()
                ):
                    search_dirs.append(item)

    # Collect all modules from search directories
    results = []
    seen_modules = set()

    for search_dir in search_dirs:
        module_dirs = sorted([d.name for d in search_dir.iterdir() if d.is_dir()])

        for module_name in module_dirs:
            # Skip if not a valid index
            module_path = search_dir / module_name
            if not (module_path / "chroma.sqlite3").exists():
                continue

            # Skip duplicates
            if module_name in seen_modules:
                continue
            seen_modules.add(module_name)

            display_name, doc_type, category_prefix, sort_order = (
                get_module_display_name(search_dir, module_name)
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


def _clear_retriever_cache(engine: Any) -> None:
    """Clear LRU cache from engine's retriever to release GPU tensor references.

    Args:
        engine: Chat engine that may contain a retriever with cached data.
    """
    if engine is None:
        return

    try:
        # CondensePlusContextChatEngine stores retriever in _retriever attribute
        retriever = getattr(engine, "_retriever", None)
        if retriever is not None and hasattr(retriever, "clear_cache"):
            retriever.clear_cache()
    except Exception as e:
        logger.warning(f"Failed to clear retriever cache: {e}")


def free_memory(engine=None):
    """Free GPU/MPS memory by clearing caches.

    Args:
        engine: Optional engine reference to delete. If None, will try to clean
                up from st.session_state if streamlit is available.
    """
    # Clear retriever LRU cache before deleting engine to release GPU tensors
    if engine is not None:
        _clear_retriever_cache(engine)
        del engine

    # Also try to clean up from streamlit session_state if available
    try:
        import streamlit as st

        if "engine" in st.session_state:
            _clear_retriever_cache(st.session_state["engine"])
            del st.session_state["engine"]

        # Reset loaded_config to force engine reload on next use
        if "loaded_config" in st.session_state:
            st.session_state.loaded_config = None
    except (ImportError, AttributeError):
        # Streamlit not available or session_state not initialized
        pass

    # Clear LlamaIndex Settings embedding model (~1-2GB VRAM)
    # Note: Access _embed_model directly to avoid auto-initialization of default model
    try:
        from llama_index.core import Settings

        if getattr(Settings, "_embed_model", None) is not None:
            Settings._embed_model = None
    except (ImportError, AttributeError):
        pass

    # Clear MARKER_CONVERTER GPU models (~2-4GB VRAM)
    try:
        from tensortruth.utils.pdf import clear_marker_converter

        clear_marker_converter()
    except ImportError as e:
        # Optional dependency not available
        logger.debug(f"Could not import clear_marker_converter: {e}")
    except Exception as e:
        # Unexpected error during VRAM cleanup
        logger.warning(f"Error while clearing marker converter: {e}", exc_info=True)

    gc.collect()

    # Synchronize CUDA before clearing cache to ensure all operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
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
