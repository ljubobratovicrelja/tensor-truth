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

    NOTE: This only extracts year from creation date. Title and authors
    are NOT extracted from embedded metadata because they are often
    incorrect (e.g., publisher names instead of authors, journal names
    in titles). Use LLM extraction for title/authors instead.

    Args:
        file_path: Path to PDF file

    Returns:
        Dictionary with extracted metadata (only year) or None if extraction fails
    """
    try:
        import pymupdf

        doc = pymupdf.open(str(file_path))
        pdf_metadata = doc.metadata

        if not pdf_metadata:
            return None

        metadata = {}

        # Only extract year from creation date (reliable)
        # Do NOT extract title/authors from embedded metadata (often wrong)
        if pdf_metadata.get("creationDate"):
            date_match = re.search(r"D:(\d{4})", pdf_metadata["creationDate"])
            if date_match:
                metadata["year"] = date_match.group(1)
                logger.info(f"Extracted year from PDF metadata: {metadata['year']}")

        doc.close()

        if not metadata:
            return None

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
    model: str = "qwen2.5-coder:7b",
    max_chars: int = 3000,
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

CRITICAL RULES:
1. Title: Extract the COMPLETE main paper/article/chapter title
   - Titles may span multiple lines - combine them into one string
   - Ignore journal names that appear before the title
   - Example: If you see "IEEE Transactions..." followed by
     "Three-Dimensional Location\nof Circular Features",
     the title is "Three-Dimensional Location of Circular Features"
2. Authors: Extract ONLY the individual person names who WROTE the document
   - DO NOT extract journal names (e.g., "IEEE Transactions", "Nature")
   - DO NOT extract publisher names (e.g., "Springer", "ACM", "IEEE")
   - DO NOT extract conference names (e.g., "CVPR", "NeurIPS")
   - DO NOT extract institution names (e.g., "MIT", "Stanford")
   - Authors are PEOPLE with first and last names (e.g., "John Smith, Jane Doe")
   - Ignore titles like "Member, IEEE" or "Fellow, IEEE" - those are not author names
3. If more than 4 authors, list all of them (do not use "et al." - we'll format that later)
4. Return ONLY valid JSON with no additional text
5. If you cannot find a clear title or authors, use null

Examples of CORRECT extraction:
- Title spanning lines: "Three-Dimensional Location Estimation of Circular
  Features for Machine Vision"
- Authors: "Reza Safaee-Rad, Ivo Tchoukanov, Kenneth Carless Smith,
  Bensiyon Benhabib"

Examples of INCORRECT author extraction (DO NOT DO THIS):
- "IEEE Transactions on Robotics" ✗ (this is a journal)
- "Springer-Verlag" ✗ (this is a publisher)
- "Member, IEEE" ✗ (this is a title, not an author)

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
    # Remove markdown code blocks if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        # Remove opening ```json or ``` and closing ```
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]  # Remove first line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove last line
        cleaned = "\n".join(lines)

    try:
        # Try direct JSON parse
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object with balanced braces
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
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


def extract_arxiv_id_from_filename(filename: str) -> Optional[str]:
    """Extract ArXiv ID from filename.

    Handles formats:
    - 1512.03385.pdf -> 1512.03385
    - 2103.00020.md -> 2103.00020

    Args:
        filename: Filename (with or without extension)

    Returns:
        ArXiv ID or None if not an ArXiv file
    """
    # Remove extension
    stem = Path(filename).stem

    # Match pattern: YYMM.NNNNN (standard ArXiv ID format)
    match = re.match(r"^(\d{4}\.\d{5})$", stem)
    if match:
        return match.group(1)

    return None


def get_arxiv_metadata_from_config(
    arxiv_id: str, module_name: str, sources_config: Dict
) -> Optional[Dict[str, Any]]:
    """Get ArXiv paper metadata from sources.json.

    Args:
        arxiv_id: ArXiv paper ID (e.g., "1512.03385")
        module_name: Module/category name (e.g., "dl_foundations")
        sources_config: Loaded sources.json config

    Returns:
        Dict with title, authors, year, source_url or None if not found
    """
    if not sources_config:
        return None

    papers = sources_config.get("papers", {})
    category = papers.get(module_name)

    if not category or "items" not in category:
        return None

    # Look up by arxiv ID key
    item = category["items"].get(arxiv_id)

    if not item:
        return None

    return {
        "title": item.get("title"),
        "authors": item.get("authors"),
        "year": item.get("year"),
        "source_url": item.get("url"),
        "arxiv_id": arxiv_id,
    }


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

    Format: "Title, Authors" or "Title" if no authors.

    Args:
        title: Document title
        authors: Author string (already formatted)

    Returns:
        Display name for UI
    """
    if not title:
        return "Unknown Document"

    if authors:
        return f"{title}, {authors}"
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


def _extract_arxiv_metadata(
    file_path: Path, module_name: str, sources_config: Dict
) -> Optional[Dict[str, Any]]:
    """Extract metadata for ArXiv papers from sources.json.

    Returns:
        Metadata dict if ArXiv paper found in config, None otherwise
    """
    arxiv_id = extract_arxiv_id_from_filename(file_path.name)
    if not arxiv_id:
        return None

    metadata = get_arxiv_metadata_from_config(arxiv_id, module_name, sources_config)
    if metadata:
        logger.info(f"Using ArXiv metadata from sources.json for {file_path.name}")
    return metadata


def _extract_explicit_or_llm_metadata(
    doc: Document,
    file_path: Path,
    module_name: Optional[str],
    sources_config: Optional[Dict],
    ollama_url: Optional[str],
    use_llm_fallback: bool,
) -> Dict[str, Any]:
    """Extract metadata using explicit sources (YAML/PDF) or LLM fallback.

    Returns:
        Metadata dict (may be incomplete)
    """
    metadata = {}

    # Try explicit extraction first
    explicit_metadata = extract_explicit_metadata(doc, file_path)

    # Check if we have complete metadata from explicit sources
    has_complete = (
        explicit_metadata
        and explicit_metadata.get("title")
        and explicit_metadata.get("authors")
    )

    if has_complete:
        metadata.update(explicit_metadata)
        logger.info(f"Using explicit metadata for {file_path.name}")
        return metadata

    # Check if this is a paper collection (will be overridden anyway)
    is_paper_collection = False
    if sources_config and module_name:
        paper_info = get_paper_collection_info_from_config(module_name, sources_config)
        is_paper_collection = bool(paper_info and paper_info.get("display_name"))

    # Use LLM fallback if enabled and not a paper collection
    if use_llm_fallback and ollama_url and not is_paper_collection:
        logger.info(f"Using LLM extraction for {file_path.name}")
        llm_metadata = extract_metadata_with_llm(doc, file_path, ollama_url)
        # Merge: keep year from explicit if available
        if explicit_metadata:
            metadata.update(explicit_metadata)
        metadata.update(llm_metadata)
    elif explicit_metadata:
        # Just use what we got from explicit extraction (e.g., year only)
        metadata.update(explicit_metadata)

    return metadata


def _apply_config_overrides(
    metadata: Dict[str, Any],
    module_name: Optional[str],
    sources_config: Optional[Dict],
    is_arxiv: bool,
) -> None:
    """Apply config-based overrides for library docs and non-ArXiv paper collections.

    Modifies metadata dict in-place.
    """
    if not sources_config or not module_name or is_arxiv:
        return

    # Check for paper collection override
    paper_info = get_paper_collection_info_from_config(module_name, sources_config)
    if paper_info and paper_info.get("display_name"):
        # Override with collection name for non-ArXiv papers
        metadata["title"] = paper_info["display_name"]
        metadata["authors"] = None
        return

    # Check for library doc info
    if not metadata.get("title"):
        lib_info = get_library_info_from_config(module_name, sources_config)
        if lib_info:
            lib_name = (
                module_name.rsplit("_", 1)[0] if "_" in module_name else module_name
            )
            version = lib_info.get("version", "")
            title_parts = [lib_name.replace("_", " ").replace("-", " ").title()]
            if version:
                title_parts.append(version)
            metadata["title"] = " ".join(title_parts)
            metadata["source_url"] = lib_info.get("doc_root")


def _finalize_metadata(
    metadata: Dict[str, Any],
    file_path: Path,
    module_name: Optional[str],
    sources_config: Optional[Dict],
) -> None:
    """Generate derived fields (display_name, source_url, doc_type).

    Modifies metadata dict in-place.
    """
    # Format authors
    if metadata.get("authors"):
        metadata["authors"] = format_authors(metadata["authors"])

    # Create display name
    title = metadata.get("title")
    if title:
        metadata["display_name"] = create_display_name(title, metadata.get("authors"))
    else:
        logger.warning(
            f"No title found for {file_path.name}, using filename as display name"
        )
        metadata["display_name"] = file_path.stem.replace("_", " ")

    # Generate source URL if not set
    if not metadata.get("source_url"):
        source_url = None
        if "arxiv_id" in metadata:
            source_url = get_source_url_for_arxiv(metadata["arxiv_id"])
        elif sources_config and module_name:
            source_url = get_source_url_for_library(module_name, sources_config)
        metadata["source_url"] = source_url

    # Classify document type
    metadata["doc_type"] = classify_document_type(file_path, module_name)

    # Log result (except for library docs to reduce noise)
    if metadata["doc_type"] != "library_doc":
        logger.info(
            f"Final metadata for {file_path.name}: "
            f"display_name={metadata['display_name']}, "
            f"doc_type={metadata['doc_type']}"
        )


def extract_document_metadata(
    doc: Document,
    file_path: Path,
    module_name: Optional[str] = None,
    sources_config: Optional[Dict] = None,
    ollama_url: Optional[str] = None,
    use_llm_fallback: bool = True,
) -> Dict[str, Any]:
    """Extract comprehensive metadata from a document.

    Extraction strategy:
    1. ArXiv papers: Look up in sources.json (fastest, most reliable)
    2. Other docs: Try explicit extraction (YAML/PDF), fallback to LLM if needed
    3. Apply config overrides for library docs and paper collections
    4. Generate derived fields (display_name, source_url, doc_type)

    Args:
        doc: LlamaIndex Document object
        file_path: Path to source file
        module_name: Name of module being indexed
        sources_config: Contents of config/sources.json
        ollama_url: Ollama API URL (for LLM fallback)
        use_llm_fallback: Whether to use LLM if explicit extraction fails

    Returns:
        Dictionary with metadata fields:
        - display_name: Pretty title for UI
        - authors: Formatted author string
        - source_url: URL to original source
        - doc_type: Document classification
        - arxiv_id: ArXiv ID if applicable
        - year: Publication year if applicable
    """
    # Step 1: Try ArXiv metadata from sources.json (ArXiv papers only)
    if sources_config and module_name:
        metadata = _extract_arxiv_metadata(file_path, module_name, sources_config)
        is_arxiv = metadata is not None
    else:
        metadata = None
        is_arxiv = False

    # Step 2: If not ArXiv, extract using explicit sources or LLM
    if not metadata:
        metadata = _extract_explicit_or_llm_metadata(
            doc, file_path, module_name, sources_config, ollama_url, use_llm_fallback
        )

    # Step 3: Apply config overrides (library docs, non-ArXiv paper collections)
    _apply_config_overrides(metadata, module_name, sources_config, is_arxiv)

    # Step 4: Finalize (format authors, create display_name, classify doc_type)
    _finalize_metadata(metadata, file_path, module_name, sources_config)

    return metadata
