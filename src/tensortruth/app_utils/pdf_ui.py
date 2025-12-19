"""PDF upload and session document UI utilities."""

from datetime import datetime

import streamlit as st

from tensortruth.app_utils import save_sessions
from tensortruth.app_utils.paths import get_session_dir
from tensortruth.pdf_handler import PDFHandler
from tensortruth.session_index import SessionIndexBuilder


def process_pdf_upload(uploaded_file, session_id: str, sessions_file: str):
    """Handle PDF upload: save, convert, index, update session.

    Args:
        uploaded_file: Streamlit UploadedFile object
        session_id: Current session ID
        sessions_file: Path to sessions JSON file
    """
    session = st.session_state.chat_data["sessions"][session_id]

    # Initialize pdf_documents list if not present
    if "pdf_documents" not in session:
        session["pdf_documents"] = []

    # Check for duplicate by filename
    for doc in session["pdf_documents"]:
        if doc["filename"] == uploaded_file.name:
            return  # Skip if already exists

    # Create PDF handler
    handler = PDFHandler(get_session_dir(session_id))

    # Save PDF and get metadata
    with st.spinner(f"Uploading {uploaded_file.name}..."):
        pdf_metadata = handler.upload_pdf(uploaded_file)

    # Add to session with "processing" status
    session["pdf_documents"].append(
        {
            "id": pdf_metadata["id"],
            "filename": uploaded_file.name,
            "uploaded_at": str(datetime.now()),
            "file_size": pdf_metadata["file_size"],
            "page_count": pdf_metadata.get("page_count", 0),
            "status": "processing",
            "error_message": None,
        }
    )
    save_sessions(sessions_file)

    # Auto-detect: use direct PDF indexing if text is extractable, else convert to markdown
    try:
        from tensortruth.utils.pdf import pdf_has_extractable_text

        # Check if PDF has extractable text
        has_text = pdf_has_extractable_text(pdf_metadata["path"])

        if has_text:
            # Fast path: index PDF directly without markdown conversion
            st.info(
                f"📄 {uploaded_file.name} has extractable text - using fast indexing"
            )

            metadata_cache = session.get("pdf_metadata_cache", {})
            builder = SessionIndexBuilder(session_id, metadata_cache=metadata_cache)

            with st.spinner("Indexing document..."):
                pdf_files = handler.get_all_pdf_files()
                builder.build_index_from_pdfs(pdf_files)
        else:
            # Slow path: scanned PDF or image-heavy, use marker for better quality
            st.warning(
                f"📸 {uploaded_file.name} appears to be scanned - "
                f"using high-quality OCR (this may take a while)"
            )

            with st.spinner(f"Converting {uploaded_file.name} with OCR..."):
                _ = handler.convert_pdf_to_markdown(pdf_metadata["path"])

            metadata_cache = session.get("pdf_metadata_cache", {})
            builder = SessionIndexBuilder(session_id, metadata_cache=metadata_cache)

            with st.spinner("Indexing document..."):
                markdown_files = handler.get_all_markdown_files()
                builder.build_index(markdown_files)

        # Save updated metadata cache
        session["pdf_metadata_cache"] = builder.get_metadata_cache()

        # Update PDF document status with extracted metadata
        for doc in session["pdf_documents"]:
            if doc["id"] == pdf_metadata["id"]:
                doc["status"] = "indexed"

                # Add extracted metadata from updated cache
                pdf_meta = session["pdf_metadata_cache"].get(pdf_metadata["id"], {})
                if pdf_meta:
                    doc["display_name"] = pdf_meta.get("display_name")
                    doc["authors"] = pdf_meta.get("authors")
                    doc["source_url"] = pdf_meta.get("source_url")
                    doc["metadata_extracted_at"] = str(datetime.now())

        session["has_temp_index"] = True
        save_sessions(sessions_file)

        # Invalidate engine cache
        st.session_state.loaded_config = None

        st.success(f"✅ {uploaded_file.name} indexed successfully!")

    except Exception as e:
        # Update status to "error"
        for doc in session["pdf_documents"]:
            if doc["id"] == pdf_metadata["id"]:
                doc["status"] = "error"
                doc["error_message"] = str(e)
        save_sessions(sessions_file)
        st.error(f"Failed to process PDF: {str(e)}")


def delete_pdf_from_session(pdf_id: str, session_id: str, sessions_file: str):
    """Delete PDF and rebuild index.

    Args:
        pdf_id: PDF document ID to delete
        session_id: Current session ID
        sessions_file: Path to sessions JSON file
    """
    session = st.session_state.chat_data["sessions"][session_id]

    # Remove from session data
    pdf_docs = session.get("pdf_documents", [])
    session["pdf_documents"] = [doc for doc in pdf_docs if doc["id"] != pdf_id]

    # Delete PDF and markdown
    handler = PDFHandler(get_session_dir(session_id))
    handler.delete_pdf(pdf_id)

    # Rebuild index if any PDFs remain, otherwise delete index
    metadata_cache = session.get("pdf_metadata_cache", {})
    builder = SessionIndexBuilder(session_id, metadata_cache=metadata_cache)

    if session["pdf_documents"]:
        try:
            # Try PDF-first approach (check if we have PDFs in pdfs dir)
            pdf_files = handler.get_all_pdf_files()
            markdown_files = handler.get_all_markdown_files()

            if pdf_files:
                # Use direct PDF indexing
                builder.build_index_from_pdfs(pdf_files)
                session["pdf_metadata_cache"] = builder.get_metadata_cache()
            elif markdown_files:
                # Fall back to markdown indexing
                builder.build_index(markdown_files)
                session["pdf_metadata_cache"] = builder.get_metadata_cache()
            else:
                # No files found, delete index
                builder.delete_index()
                session["has_temp_index"] = False
        except Exception as e:
            st.warning(f"Failed to rebuild index: {e}")
            builder.delete_index()
            session["has_temp_index"] = False
    else:
        builder.delete_index()
        session["has_temp_index"] = False

    save_sessions(sessions_file)

    # Invalidate engine cache
    st.session_state.loaded_config = None


def render_pdf_documents_section(session_id: str, sessions_file: str):
    """Render the PDF documents section in sidebar.

    Args:
        session_id: Current session ID
        sessions_file: Path to sessions JSON file
    """
    session = st.session_state.chat_data["sessions"][session_id]

    st.markdown("**📄 Session Documents**")
    pdf_docs = session.get("pdf_documents", [])

    if pdf_docs:
        for doc in pdf_docs:
            status_icons = {
                "uploading": "⏳",
                "processing": "🔄",
                "indexed": "✅",
                "error": "❌",
            }
            status_icon = status_icons.get(doc["status"], "❓")

            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.caption(f"{status_icon} {doc['filename']}")
                if doc.get("error_message"):
                    st.caption(f"⚠️ {doc['error_message']}", help="Error details")
            with col2:
                if st.button("🗑️", key=f"del_pdf_{doc['id']}", help="Remove PDF"):
                    delete_pdf_from_session(doc["id"], session_id, sessions_file)
                    st.rerun()
    else:
        st.caption("No documents uploaded")

    # Upload widget
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="collapsed",
        help="Upload a PDF document to query alongside knowledge bases",
    )

    # Track processed files to avoid reprocessing on rerun
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = set()

    if uploaded_file:
        # Create unique identifier for this file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        # Only process if not already processed
        if file_id not in st.session_state.processed_pdfs:
            process_pdf_upload(uploaded_file, session_id, sessions_file)
            st.session_state.processed_pdfs.add(file_id)
            st.rerun()
