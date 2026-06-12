# """
# src/rag/historical_db.py

# ChromaDB retrieval for Tool 1 — Permanent Historical RFP Knowledge Base.

# Usage in tools.py:
#   from src.rag.historical_db import search_historical_rfp as _hist_fn
#   result = _hist_fn(query=query, top_k=10)
# """

# from __future__ import annotations

# import os
# from pathlib import Path

# import chromadb
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# from dotenv import load_dotenv

# load_dotenv()

# # ---------------------------------------------------------------------------
# # Paths and constants
# # ---------------------------------------------------------------------------

# _BASE_DIR   = Path(__file__).resolve().parent.parent.parent
# _CHROMA_DIR = str(_BASE_DIR / "chroma_data" / "historical_rfps")
# _COLLECTION = "historical_rfps"

# # ---------------------------------------------------------------------------
# # Lazy-loaded ChromaDB client and collection
# # ---------------------------------------------------------------------------

# _client: chromadb.PersistentClient | None = None
# _collection = None


# def _get_collection():
#     global _client, _collection
#     if _collection is None:
#         _client = chromadb.PersistentClient(path=_CHROMA_DIR)
#         _collection = _client.get_or_create_collection(
#             name=_COLLECTION,
#             embedding_function=OpenAIEmbeddingFunction(
#                 api_key=os.getenv("OPENAI_API_KEY", ""),
#                 model_name="text-embedding-3-small",
#             ),
#             metadata={"hnsw:space": "cosine"},
#         )
#     return _collection


# # ---------------------------------------------------------------------------
# # Public retrieval function (called by tools.py)
# # ---------------------------------------------------------------------------

# def search_historical_rfp(query: str, top_k: int = 20) -> str:
#     """
#     Search the permanent historical RFP ChromaDB collection.

#     Args:
#         query:  Focused search query (key terms only — 10-30 words ideal).
#         top_k:  Number of chunks to retrieve.

#     Returns:
#         Formatted string of matched historical RFP chunks with client, section,
#         and relevance score.  Returns an error string on failure.
#     """
#     try:
#         collection = _get_collection()
#         total = collection.count()

#         if total == 0:
#             return (
#                 "[SAMPLE DB EMPTY] Run scripts/create_sample_dbs.py first "
#                 "to populate the historical RFP knowledge base."
#             )

#         results = collection.query(
#             query_texts=[query],
#             n_results=min(top_k, total),
#             include=["documents", "metadatas", "distances"],
#         )

#         docs      = results["documents"][0]
#         metas     = results["metadatas"][0]
#         distances = results["distances"][0]

#         if not docs:
#             return "No matching historical RFP data found for this query."

#         parts = []
#         for doc, meta, dist in zip(docs, metas, distances):
#             similarity = round(1.0 - dist, 3)          # cosine distance → similarity
#             client_name = meta.get("client", "Unknown")
#             section     = meta.get("section", "")
#             industry    = meta.get("industry", "")

#             header = f"**Client:** {client_name}"
#             if section:
#                 header += f" | **Section:** {section}"
#             if industry:
#                 header += f" | **Industry:** {industry}"
#             header += f" | **Relevance:** {similarity}"

#             parts.append(f"{header}\n\n{doc}")

#         return "\n\n---\n\n".join(parts)

#     except Exception as exc:
#         return f"[ERROR] Historical RFP search failed: {exc}"




"""
rag/historical_db.py — Historical RFP ChromaDB retrieval using MiniLM.
"""
from __future__ import annotations
from pathlib import Path
from rag.vector import query_collection

_BASE_DIR   = Path(__file__).resolve().parent.parent
_CHROMA_DIR = str(_BASE_DIR / "chroma_data" / "historical_rfps")
_COLLECTION = "historical_rfps"

def search_historical_rfp(query: str, top_k: int = 20) -> str:
    try:
        results = query_collection(
            collection_name=_COLLECTION,
            db_path=_CHROMA_DIR,
            query_text=query,
            n_results=top_k
        )
        if not results:
            return "No matching historical RFP data found for this query."

        parts = []
        for r in results:
            meta = r["metadata"]
            source  = meta.get("source_file", "Unknown")
            sheet   = meta.get("sheet_name", "")
            heading = meta.get("meta_headings", "")
            header  = f"**Source:** {source}"
            if sheet:   header += f" | **Sheet:** {sheet}"
            if heading and heading != "[]": header += f" | **Section:** {heading}"
            header += f" | **Relevance:** {r['score']}"
            parts.append(f"{header}\n\n{r['content']}")

        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        return f"[ERROR] Historical RFP search failed: {exc}"
