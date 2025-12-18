"""Document metadata extraction utilities.

This module provides functions to extract rich metadata from documents
for enhanced citation display in the RAG pipeline.

Extraction strategy:
1. Explicit metadata first (YAML headers, PDF metadata)
2. LLM-based extraction as fallback
3. Graceful degradation to filename if all else fails
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


# ============================================================================
# Explicit Metadata Extraction
# ============================================================================


def extract_yaml_header_metadata(content: str) -> Optional[Dict[str, Any]]:
    """Extract metadata from YAML-like header in markdown files.

    Looks for headers in format:
        # Title: Some Title
        # Authors: Author1, Author2
        # Year: 2023
        # ArXiv ID: 1234.5678

    Args:
        content: Document text content

    Returns:
        Dictionary with extracted metadata or None if no header found
    """
    lines = content.split("\n")
    metadata = {}

    # Only check first 20 lines for header
    for line in lines[:20]:
        line = line.strip()

        # Match pattern: # Key: Value
        match = re.match(r"^#\s*([^:]+):\s*(.+)$", line)
        if match:
            key = match.group(1).strip().lower()
            value = match.group(2).strip()

            # Map common keys
            if key == "title":
                metadata["title"] = value
            elif key in ["author", "authors"]:
                metadata["authors"] = value
            elif key == "year":
                metadata["year"] = value
            elif key in ["arxiv id", "arxiv_id"]:
                metadata["arxiv_id"] = value

    # Return None if no metadata found
    if not metadata:
        return None

    logger.info(f"Extracted YAML metadata: {metadata}")
    return metadata


def extract_pdf_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from PDF file info dict.

    Uses PyMuPDF to read PDF metadata (Title, Author, Subject, etc.)

    Args:
        file_path: Path to PDF file

    Returns:
        Dictionary with extracted metadata or None if extraction fails
    """
    try:
        import pymupdf

        doc = pymupdf.open(str(file_path))
        pdf_metadata = doc.metadata

        if not pdf_metadata:
            return None

        metadata = {}

        # Extract title
        if pdf_metadata.get("title"):
            metadata["title"] = pdf_metadata["title"]

        # Extract author
        if pdf_metadata.get("author"):
            metadata["authors"] = pdf_metadata["author"]

        # Extract year from creation date if available
        if pdf_metadata.get("creationDate"):
            date_match = re.search(r"D:(\d{4})", pdf_metadata["creationDate"])
            if date_match:
                metadata["year"] = date_match.group(1)

        doc.close()

        if not metadata:
            return None

        logger.info(f"Extracted PDF metadata: {metadata}")
        return metadata

    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata: {e}")
        return None


def extract_explicit_metadata(
    doc: Document, file_path: Path
) -> Optional[Dict[str, Any]]:
    """Extract metadata from explicit sources (YAML headers, PDF metadata).

    Tries YAML header first (for markdown), then PDF metadata.

    Args:
        doc: LlamaIndex Document object
        file_path: Path to source file

    Returns:
        Dictionary with extracted metadata or None if no explicit metadata found
    """
    # Try YAML header extraction (for markdown files)
    if file_path.suffix.lower() in [".md", ".markdown"]:
        yaml_metadata = extract_yaml_header_metadata(doc.text)
        if yaml_metadata:
            return yaml_metadata

    # Try PDF metadata extraction
    if file_path.suffix.lower() == ".pdf":
        pdf_metadata = extract_pdf_metadata(file_path)
        if pdf_metadata:
            return pdf_metadata

    return None


# ============================================================================
# LLM-based Metadata Extraction
# ============================================================================


def extract_metadata_with_llm(
    doc: Document,
    file_path: Path,
    ollama_url: str,
    model: str = "qwen2.5:0.5b",
    max_chars: int = 2000,
) -> Dict[str, Any]:
    """Use LLM to extract title and authors from document content.

    Args:
        doc: LlamaIndex Document object
        file_path: Path to source file
        ollama_url: Ollama API base URL
        model: Ollama model to use for extraction
        max_chars: Maximum characters to send to LLM

    Returns:
        Dictionary with extracted metadata (may have None values)
    """
    # Extract first N characters
    excerpt = doc.text[:max_chars]

    # Build prompt
    prompt = f"""You are a document metadata extractor. Extract the title and \
authors from the following document excerpt.

Rules:
1. Title: The main title of the document (paper, book chapter, article)
2. Authors: All authors, comma-separated. If more than 3 authors, use "FirstAuthor et al."
3. Return ONLY valid JSON with no additional text
4. If you cannot find a clear title, use null
5. If you cannot find authors, use null

Return format (JSON only):
{{
  "title": "string or null",
  "authors": "string or null"
}}

Document excerpt:
---
{excerpt}
---

JSON response:"""

    try:
        # Call Ollama API
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()

        # Parse response
        result = response.json()
        llm_output = result.get("response", "").strip()

        # Parse JSON from LLM output
        metadata = _parse_llm_json_response(llm_output)

        if metadata.get("title"):
            logger.info(
                f"LLM extracted metadata: title='{metadata['title']}', "
                f"authors='{metadata.get('authors')}'"
            )
        else:
            logger.warning(
                f"LLM extraction returned no title for {file_path.name}. "
                f"Raw LLM output: {llm_output[:200]}"
            )
        return metadata

    except Exception as e:
        logger.warning(f"LLM metadata extraction failed for {file_path.name}: {e}")
        return {"title": None, "authors": None}


def _parse_llm_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with error handling.

    Args:
        response: Raw LLM response text

    Returns:
        Dictionary with title and authors (may be None)
    """
    try:
        # Try direct JSON parse
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON block with regex
        match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Fallback: return empty metadata
    logger.warning(f"Failed to parse LLM JSON response: {response[:100]}...")
    return {"title": None, "authors": None}


# ============================================================================
# Helper Functions
# ============================================================================


def get_source_url_for_arxiv(arxiv_id: str) -> str:
    """Generate ArXiv URL from ID.

    Handles both old (hep-th/9901001) and new (1234.5678) formats.

    Args:
        arxiv_id: ArXiv paper ID

    Returns:
        Full ArXiv URL
    """
    return f"https://arxiv.org/abs/{arxiv_id}"


def get_library_info_from_config(
    module_name: str, sources_config: Dict
) -> Optional[Dict]:
    """Get library information from config for a module.

    Args:
        module_name: Module directory name (e.g., "pytorch_2.9")
        sources_config: Contents of config/sources.json

    Returns:
        Dictionary with title, version, doc_root or None if not found
    """
    libraries = sources_config.get("libraries", {})

    # Try direct match first
    if module_name in libraries:
        return libraries[module_name]

    # Try matching by library name (extract from module_name)
    # Pattern: libraryname_version (e.g., pytorch_2.9 -> pytorch)
    lib_name = module_name.rsplit("_", 1)[0] if "_" in module_name else module_name

    if lib_name in libraries:
        lib_config = libraries[lib_name]
        # Add the actual module name for reference
        return {**lib_config, "module_name": module_name}

    return None


def get_paper_collection_info_from_config(
    module_name: str, sources_config: Dict
) -> Optional[Dict]:
    """Get paper collection information from config for a module.

    Args:
        module_name: Module directory name (e.g., "dl_foundations")
        sources_config: Contents of config/sources.json

    Returns:
        Dictionary with display_name, description or None if not found
    """
    papers = sources_config.get("papers", {})

    if module_name in papers:
        return papers[module_name]

    return None


def get_source_url_for_library(module_name: str, sources_config: Dict) -> Optional[str]:
    """Get base documentation URL for a library module.

    Args:
        module_name: Name of library/module (e.g., "pytorch_2.9")
        sources_config: Contents of config/sources.json

    Returns:
        Base documentation URL or None if not found
    """
    lib_info = get_library_info_from_config(module_name, sources_config)
    return lib_info.get("doc_root") if lib_info else None


def format_authors(authors: Union[str, List, None]) -> Optional[str]:
    """Format authors for display.

    Converts to "LastName et al." format if more than 3 authors.

    Args:
        authors: Author string or list

    Returns:
        Formatted author string or None
    """
    if not authors:
        return None

    # Handle list input
    if isinstance(authors, list):
        authors = ", ".join(authors)

    # Count authors (split by comma or "and")
    author_list = re.split(r",|\band\b", authors)
    author_list = [a.strip() for a in author_list if a.strip()]

    if len(author_list) == 0:
        return None
    elif len(author_list) == 1:
        return author_list[0]
    elif len(author_list) <= 3:
        return ", ".join(author_list)
    else:
        # Extract first author's last name
        first_author = author_list[0]
        # Try to get last name (assumes "First Last" or "Last, First" format)
        if "," in first_author:
            last_name = first_author.split(",")[0].strip()
        else:
            parts = first_author.split()
            last_name = parts[-1] if parts else first_author

        return f"{last_name} et al."


def create_display_name(title: Optional[str], authors: Optional[str] = None) -> str:
    """Create pretty display name for citation.

    Format: "Title - Authors" or "Title" if no authors.

    Args:
        title: Document title
        authors: Author string (already formatted)

    Returns:
        Display name for UI
    """
    if not title:
        return "Unknown Document"

    if authors:
        return f"{title} - {authors}"
    else:
        return title


def classify_document_type(file_path: Path, module_name: Optional[str] = None) -> str:
    """Classify document type based on path and module.

    Args:
        file_path: Path to document file
        module_name: Name of module being indexed

    Returns:
        Document type: "paper", "book", "library_doc", "uploaded_pdf"
    """
    filename = file_path.stem.lower()

    # Check if it's an uploaded PDF (has pdf_UUID pattern)
    if re.match(r"^pdf_[a-f0-9]+", filename):
        return "uploaded_pdf"

    # Check module name prefix first (most reliable indicator)
    if module_name:
        if module_name.startswith("book_"):
            return "book"
        # Paper collections have specific naming patterns
        if any(
            module_name.startswith(prefix)
            for prefix in ["dl_foundations", "vision_", "3d_reconstruction"]
        ):
            return "paper"

    # Check if it's a book (common book keywords in path)
    if any(
        keyword in str(file_path).lower()
        for keyword in ["_books", "textbook", "manual", "guide"]
    ):
        return "book"

    # Check if it's a paper (common paper indicators)
    if any(
        keyword in filename
        for keyword in ["arxiv", "paper", "proceedings", "conference", "journal"]
    ):
        return "paper"

    # Check if it's library documentation (has module name in path and not a paper/book)
    if module_name and module_name.lower() in str(file_path).lower():
        return "library_doc"

    # Default to paper
    return "paper"


# ============================================================================
# Main Extraction Orchestrator
# ============================================================================


def extract_document_metadata(
    doc: Document,
    file_path: Path,
    module_name: Optional[str] = None,
    sources_config: Optional[Dict] = None,
    ollama_url: Optional[str] = None,
    use_llm_fallback: bool = True,
) -> Dict[str, Any]:
    """Extract comprehensive metadata from a document.

    This is the main entry point for metadata extraction.
    Tries explicit extraction first, falls back to LLM if enabled.

    Args:
        doc: LlamaIndex Document object
        file_path: Path to source file
        module_name: Name of module being indexed (for classification)
        sources_config: Contents of config/sources.json
        ollama_url: Ollama API URL (required if use_llm_fallback=True)
        use_llm_fallback: Whether to use LLM if explicit extraction fails

    Returns:
        Dictionary with metadata fields:
        - display_name: Pretty title for UI
        - authors: Formatted author string
        - source_url: URL to original source
        - doc_type: Document classification
        - arxiv_id: ArXiv ID if applicable
        - year: Publication year if available
    """
    metadata = {}

    # Step 1: Try explicit extraction
    explicit_metadata = extract_explicit_metadata(doc, file_path)

    if explicit_metadata:
        metadata.update(explicit_metadata)
        logger.info(f"Using explicit metadata for {file_path.name}")
    elif use_llm_fallback and ollama_url:
        # Step 2: Fallback to LLM extraction
        logger.info(f"No explicit metadata found, using LLM for {file_path.name}")
        llm_metadata = extract_metadata_with_llm(doc, file_path, ollama_url)
        metadata.update(llm_metadata)

    # Step 3: For library docs and paper collections, override with config info if available
    if sources_config and module_name:
        # Check if it's a paper collection first (always override for collections)
        paper_info = get_paper_collection_info_from_config(module_name, sources_config)
        if paper_info and paper_info.get("display_name"):
            # Override individual paper title with collection name
            metadata["title"] = paper_info["display_name"]
            # Clear authors for paper collections (we don't want individual paper authors)
            metadata["authors"] = None
        elif not metadata.get("title"):
            # For library docs, use config if no title extracted
            lib_info = get_library_info_from_config(module_name, sources_config)
            if lib_info:
                # Use library name and version from config
                lib_name = (
                    module_name.rsplit("_", 1)[0] if "_" in module_name else module_name
                )
                version = lib_info.get("version", "")

                # Create title: "Library Version" (e.g., "PyTorch 2.9")
                title_parts = [lib_name.replace("_", " ").replace("-", " ").title()]
                if version:
                    title_parts.append(version)

                metadata["title"] = " ".join(title_parts)
                metadata["source_url"] = lib_info.get("doc_root")

    # Step 4: Generate derived fields
    title = metadata.get("title")
    authors = metadata.get("authors")

    # Format authors
    if authors:
        formatted_authors = format_authors(authors)
        metadata["authors"] = formatted_authors

    # Create display name
    if title:
        display_name = create_display_name(title, metadata.get("authors"))
    else:
        # Fallback to filename without extension
        logger.warning(
            f"No title found for {file_path.name}, using filename as display name. "
            f"Metadata extraction may have failed."
        )
        display_name = file_path.stem.replace("_", " ")

    metadata["display_name"] = display_name

    # Generate source URL if not already set
    if not metadata.get("source_url"):
        source_url = None
        if "arxiv_id" in metadata:
            source_url = get_source_url_for_arxiv(metadata["arxiv_id"])
        elif sources_config and module_name:
            source_url = get_source_url_for_library(module_name, sources_config)
        metadata["source_url"] = source_url

    # Classify document type
    doc_type = classify_document_type(file_path, module_name)
    metadata["doc_type"] = doc_type

    # Only log for non-library docs (library docs are handled via config, no need for per-file logs)
    if doc_type != "library_doc":
        logger.info(
            f"Final metadata for {file_path.name}: "
            f"display_name={metadata['display_name']}, "
            f"doc_type={doc_type}"
        )

    return metadata
