"""
rag/new_rfp.py

ChromaDB ingestion + retrieval for Tool 2 — New RFP (session-scoped).
Uses Docling chunking and MiniLM embeddings via rag.vector.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rag.chunk4 import chunk_multiple_documents
from rag.vector import build_collection, query_collection
import json # Make sure to add this import at the top of your file!

_BASE_DIR    = Path(__file__).resolve().parent.parent
_CHROMA_BASE = _BASE_DIR / "chroma_data"

def _get_db_path(session_id: str) -> str:
    return str(_CHROMA_BASE / f"new_rfp_{session_id}")

def ingest_new_rfps(file_paths: list, session_id: str) -> int:
    """
    Parse MULTIPLE uploaded RFP files using Docling, chunk them, and store into ChromaDB.
    """
    chunks = chunk_multiple_documents(file_paths, session_id=session_id)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded files.")
    

    # 2. NEW CODE: Print all extracted chunks to the terminal
    # ============================================================
    print("\n" + "="*60)
    print(f"📄 EXTRACTED {len(chunks)} CHUNKS FOR SESSION: {session_id}")
    print("="*60)
    
    for i, chunk in enumerate(chunks):
        text_content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        print(f"\n--- CHUNK {i + 1} | Source: {metadata.get('source_file', 'Unknown')} ---")
        print(text_content)
        
    print("="*60 + "\n")
    # 
    
    txt_filename = f"chunks_session_{session_id}.txt"
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                metadata = chunk.get("metadata", {})
                f.write(f"--- CHUNK {i + 1} | Source: {metadata.get('source_file', 'Unknown')} ---\n")
                f.write(chunk.get("content", ""))
                f.write("\n\n")
        print(f"💾 Successfully saved all chunks to {txt_filename}\n")
    except Exception as e:
        print(f"⚠️ Failed to save chunks to TXT: {e}")


    db_path = _get_db_path(session_id)
    col_name = f"new_rfp_{session_id}"
    
    build_collection(col_name, db_path, chunks)
    return len(chunks)


def get_new_rfp_context(query: str, session_id: str, top_k: int = 20) -> str:
    """
    Search the uploaded RFP's ChromaDB collection for this session.
    """
    db_path = _get_db_path(session_id)
    col_name = f"new_rfp_{session_id}"

    try:
        if not Path(db_path).exists():
            return (
                f"[Session: {session_id}] No RFP data found. "
                "The RFP document may not have been ingested yet."
            )

        results = query_collection(
            collection_name=col_name,
            db_path=db_path,
            query_text=query,
            n_results=top_k
        )

        if not results:
            return "No relevant clauses found in the uploaded RFP for this query."

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
        return f"[ERROR] New RFP context retrieval failed: {exc}"


def delete_rfp_collection(session_id: str) -> bool:
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
