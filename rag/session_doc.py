"""
rag/session_doc.py

ChromaDB ingestion + retrieval for Tool 3 — Supporting Document (session-scoped).
Uses Docling chunking and MiniLM embeddings via rag.vector.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rag.chunk4 import chunk_multiple_documents
from rag.vector import build_collection, query_collection

_BASE_DIR    = Path(__file__).resolve().parent.parent
_CHROMA_BASE = _BASE_DIR / "chroma_data"

def _get_db_path(session_id: str) -> str:
    return str(_CHROMA_BASE / f"session_doc_{session_id}")

def ingest_session_documents(file_paths: list, session_id: str) -> int:
    """
    Parse MULTIPLE uploaded supporting documents and store all chunks into ChromaDB.
    """
    chunks = chunk_multiple_documents(file_paths, session_id=session_id)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded files.")

    db_path = _get_db_path(session_id)
    col_name = f"session_doc_{session_id}"
    
    build_collection(col_name, db_path, chunks)
    return len(chunks)


def search_session_document(query: str, session_id: str, top_k: int = 5) -> str:
    """
    Search the uploaded supporting document's ChromaDB collection for this session.
    """
    db_path = _get_db_path(session_id)
    col_name = f"session_doc_{session_id}"

    try:
        if not Path(db_path).exists():
            return (
                f"[Session: {session_id}] No supporting document data found. "
                "The document may not have been ingested yet."
            )

        results = query_collection(
            collection_name=col_name,
            db_path=db_path,
            query_text=query,
            n_results=top_k
        )

        if not results:
            return "No relevant content found in the uploaded document for this query."

        parts = []
        for r in results:
            meta = r["metadata"]
            score = r["score"]
            doc = r["content"]

            source = meta.get("meta_source_file", "Unknown")
            sheet = meta.get("meta_sheet_name", "")
            chunk_type = meta.get("chunk_type", "")
            headings = meta.get("meta_headings", "")

            header = f"**Source:** {source}"
            if sheet:
                header += f" | **Sheet:** {sheet}"
            if headings and headings != "[]":
                header += f" | **Headings:** {headings}"
            if chunk_type:
                header += f" | **Type:** {chunk_type}"
            header += f" | **Relevance:** {score}"

            parts.append(f"{header}\n\n{doc}")

        return "\n\n---\n\n".join(parts)

    except Exception as exc:
        return f"[ERROR] Session document search failed: {exc}"


def delete_doc_collection(session_id: str) -> bool:
    """
    Delete the ChromaDB collection for a session completely from disk.
    """
    try:
        db_path = _get_db_path(session_id)
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
        return True
    except Exception:
        return False
