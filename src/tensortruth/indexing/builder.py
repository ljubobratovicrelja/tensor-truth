"""Core vector index building functionality.

This module provides the core business logic for building hierarchical
vector indexes from documentation with metadata extraction.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..app_utils.config_schema import DEFAULT_EMBEDDING_MODEL_CONFIGS, TensorTruthConfig
from ..core.types import DocumentType
from ..utils.metadata import (
    extract_arxiv_metadata_from_config,
    extract_book_chapter_metadata,
    extract_library_metadata_from_config,
    extract_library_module_metadata,
    get_book_metadata_from_config,
    get_document_type_from_config,
)
from .metadata import sanitize_model_id, write_index_metadata

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

logger = logging.getLogger(__name__)


def _create_embed_model(
    model_name: str,
    device: str,
) -> HuggingFaceEmbedding:
    """Create HuggingFace embedding model with config-driven settings.

    Loads model-specific configuration from config.yaml (or built-in defaults)
    and applies optimizations like batch size, dtype, flash attention, etc.

    Args:
        model_name: HuggingFace model path (e.g., "BAAI/bge-m3")
        device: Device to run on ("cpu", "cuda", "mps")

    Returns:
        Configured HuggingFaceEmbedding instance
    """
    # Try to load config, fall back to built-in defaults
    try:
        from ..app_utils.config import load_config

        config = load_config()
        model_config = config.rag.get_embedding_model_config(model_name)
    except Exception:
        # Fall back to built-in defaults if config loading fails
        from ..app_utils.config_schema import (
            DEFAULT_EMBEDDING_MODEL_CONFIG,
            EmbeddingModelConfig,
        )

        if model_name in DEFAULT_EMBEDDING_MODEL_CONFIGS:
            model_config = EmbeddingModelConfig(
                **DEFAULT_EMBEDDING_MODEL_CONFIGS[model_name]
            )
        else:
            model_config = DEFAULT_EMBEDDING_MODEL_CONFIG

    # Determine batch size based on device
    batch_size = (
        model_config.batch_size_cuda
        if device == "cuda"
        else model_config.batch_size_cpu
    )

    # Build model_kwargs
    model_kwargs: Dict[str, Any] = {"trust_remote_code": model_config.trust_remote_code}

    # Add torch_dtype if specified
    if model_config.torch_dtype:
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if model_config.torch_dtype in dtype_map:
            model_kwargs["torch_dtype"] = dtype_map[model_config.torch_dtype]

    # Try to enable flash attention if configured
    if model_config.flash_attention:
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled for embedding model")
        except ImportError:
            logger.debug("Flash Attention not available, using default attention")

    # Build tokenizer_kwargs
    tokenizer_kwargs = None
    if model_config.padding_side:
        tokenizer_kwargs = {"padding_side": model_config.padding_side}

    logger.info(
        f"Creating embedding model: {model_name} "
        f"(batch_size={batch_size}, dtype={model_config.torch_dtype or 'default'})"
    )

    return HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        embed_batch_size=batch_size,
    )


def extract_metadata(
    module_name: str,
    doc_type: DocumentType,
    sources_config: Dict,
    documents: List[Document],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Extract metadata for a list of documents in a module.

    This function extracts document-specific metadata based on the document
    type and injects it into the document objects for use during indexing.

    Args:
        module_name: Name of the module (e.g., "pytorch", "dl_foundations")
        doc_type: Type of documents being processed (BOOK, LIBRARY, or PAPERS)
        sources_config: Loaded sources.json configuration
        documents: List of LlamaIndex Document objects to process
        progress_callback: Optional callback function(current, total) for progress updates

    Raises:
        ValueError: If document type is unknown or metadata extraction fails critically
    """
    # Prepare root metadata - book info for all chunks, or library for all doc modules etc.
    root_metadata = None

    if doc_type == DocumentType.BOOK:
        logger.info("Using book metadata extraction with caching.")
        root_metadata = get_book_metadata_from_config(module_name, sources_config)
    elif doc_type == DocumentType.LIBRARY:
        logger.info("Using library metadata extraction.")
        root_metadata = extract_library_metadata_from_config(
            module_name, sources_config
        )

    # Extract metadata for each document
    for i, doc in enumerate(documents):
        file_path = Path(doc.metadata.get("file_path", ""))
        try:
            # Extract metadata based on document type
            if doc_type == DocumentType.BOOK:
                # Book extraction with caching
                if not root_metadata:
                    raise ValueError(f"Missing root metadata for book: {module_name}")
                metadata = extract_book_chapter_metadata(
                    file_path,
                    root_metadata,
                )
            elif doc_type == DocumentType.LIBRARY:
                # Library module extraction, handles per-module URL and display name.
                if not root_metadata:
                    raise ValueError(
                        f"Missing root metadata for library: {module_name}"
                    )
                metadata = extract_library_module_metadata(file_path, root_metadata)
            elif doc_type == DocumentType.PAPERS:
                # Per paper display name, authors, URL etc.
                maybe_metadata = extract_arxiv_metadata_from_config(
                    file_path, module_name, sources_config
                )
                if not maybe_metadata:
                    raise ValueError(f"Missing metadata for paper: {file_path}")
                metadata = maybe_metadata
            else:
                raise ValueError(f"Unknown document type: {doc_type}")

            # Inject only essential metadata fields to avoid chunk size issues
            # (LlamaIndex includes metadata in chunk context)
            essential_fields = [
                "title",
                "formatted_authors",
                "display_name",
                "authors",
                "source_url",
                "doc_type",
                "group_display_name",  # For paper groups UI display
                "book_display_name",  # For book UI display (same across all chapters)
                "library_display_name",  # For library UI display (same across all modules)
            ]

            for field in essential_fields:
                if field in metadata:
                    doc.metadata[field] = metadata[field]

            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(documents):
                logger.info(f"  Processed {i + 1}/{len(documents)} documents...")
                if progress_callback:
                    progress_callback(i + 1, len(documents))

        except Exception as e:
            logger.warning(f"Failed to extract metadata for {file_path.name}: {e}")
            # Continue with default metadata

    logger.info(f"Metadata extraction complete for {len(documents)} documents")


def build_module(
    module_name: str,
    library_docs_dir: str,
    indexes_dir: str,
    sources_config: Dict,
    extensions: List[str] | None = None,
    chunk_sizes: List[int] | None = None,
    device: Optional[str] = None,
    embedding_model: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> bool:
    """Build vector index for a documentation module.

    This is the core indexing function that:
    1. Loads documents from the source directory
    2. Extracts metadata based on document type
    3. Parses documents into hierarchical chunks
    4. Embeds chunks and creates vector index
    5. Persists the index to disk with metadata

    Args:
        module_name: Name of module (e.g., "pytorch", "dl_foundations")
        library_docs_dir: Base directory containing library documentation
        indexes_dir: Base directory for vector indexes
        sources_config: Loaded sources.json configuration
        extensions: List of file extensions to include (default: [".md", ".html", ".pdf"])
        chunk_sizes: Hierarchical chunk sizes for document parsing (default: [2048, 512, 256])
        device: Device for embedding ("cpu", "cuda", "mps"). Auto-detected if None.
        embedding_model: HuggingFace embedding model path (default: BAAI/bge-m3)
        progress_callback: Optional callback function(stage, current, total) for progress updates

    Returns:
        True if build succeeded, False if build failed or was skipped

    Raises:
        ValueError: If module configuration is invalid
    """
    if extensions is None:
        extensions = [".md", ".html", ".pdf"]
    if chunk_sizes is None:
        chunk_sizes = [2048, 512, 256]
    if embedding_model is None:
        embedding_model = DEFAULT_EMBEDDING_MODEL

    # Get document type from config
    doc_type = get_document_type_from_config(module_name, sources_config)
    logger.info(f"Module '{module_name}' document type: {doc_type}")

    module_dir_name = f"{doc_type.value}_{module_name}"

    source_dir = os.path.join(library_docs_dir, module_dir_name)

    # Use embedding model subdirectory for versioned indexes
    model_id = sanitize_model_id(embedding_model)
    persist_dir = os.path.join(indexes_dir, model_id, module_dir_name)

    logger.info(f"\n--- BUILDING MODULE: {module_name} ---")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: {persist_dir}")

    # Validate source directory
    if not os.path.exists(source_dir):
        logger.error(f"Source directory missing: {source_dir}")
        return False

    # Remove old index if it exists
    if os.path.exists(persist_dir):
        logger.info(f"Removing old index at {persist_dir}...")
        shutil.rmtree(persist_dir)

    # Load documents
    try:
        documents = SimpleDirectoryReader(
            source_dir,
            recursive=True,
            required_exts=extensions,
            exclude_hidden=False,
        ).load_data()
    except Exception as e:
        logger.error(f"Failed to load documents from {source_dir}: {e}")
        return False

    logger.info(f"Loaded {len(documents)} documents.")

    if len(documents) == 0:
        logger.warning(f"No documents found in {source_dir}. Skipping module.")
        return False

    # Extract metadata
    if progress_callback:
        progress_callback("metadata", 0, len(documents))

    extract_metadata(
        module_name,
        doc_type,
        sources_config,
        documents,
        progress_callback=lambda curr, total: (
            progress_callback("metadata", curr, total) if progress_callback else None
        ),
    )

    # Parse documents into hierarchical nodes
    if progress_callback:
        progress_callback("parsing", 0, len(documents))

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    logger.info(f"Parsed {len(nodes)} nodes ({len(leaf_nodes)} leaves).")

    # Create isolated vector database
    db = chromadb.PersistentClient(path=persist_dir)
    collection = db.get_or_create_collection("data")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Build index and persist
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)

    # Detect device if not provided
    if device is None:
        device = TensorTruthConfig._detect_default_device()

    logger.info(f"Embedding with {embedding_model} on {device.upper()}...")

    if progress_callback:
        progress_callback("embedding", 0, len(leaf_nodes))

    # Create embedding model with config-driven settings
    embed_model = _create_embed_model(embedding_model, device)

    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    storage_context.persist(persist_dir=persist_dir)

    # Write index metadata for future compatibility checks
    write_index_metadata(
        index_dir=Path(persist_dir),
        embedding_model=embedding_model,
        chunk_sizes=chunk_sizes,
    )

    logger.info(f"Module '{module_name}' built successfully with {embedding_model}!")

    return True
