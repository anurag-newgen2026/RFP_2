"""
vector.py – ChromaDB Vector Store Builder & Query Interface
===========================================================
Reads  4chunked_rfp_data.json  (produced by chunk4.py) and ingests every
chunk into a persistent ChromaDB collection using a local sentence-
transformer embedding model (no API key required).

Usage
-----
  Build:  python vector.py build
  Query:  python vector.py query "your question here"
  Stats:  python vector.py stats

Collection layout
-----------------
  document  → chunk content text
  id        → "chunk_<chunk_index>"
  metadata  → flat dict of all fields ChromaDB supports
                (str / int / float / bool – lists serialised to JSON strings)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ChromaDB's built-in ONNX embedding model (all-MiniLM-L6-v2, ~45 MB, no API key)
# Uses chromadb's own download path – avoids HuggingFace xet issues on Windows

INGEST_BATCH     = 100   # chunks per upsert call
SKIP_MIN_CHARS   = 10    # skip chunks shorter than this (e.g. "SUB-TOTAL")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB requires all metadata values to be str | int | float | bool.
    Converts the nested chunk record into a flat dict, serialising lists
    as JSON strings and dropping None values.

    Fields mapped:
      Root level   : chunk_type, content_length, word_count, created_at
      metadata.*   : source_file, doc_format, chunk_index, sheet_name,
                     table_header, table_chunk_index, page_range,
                     section_depth
      metadata.*   : headings, subheadings, captions, element_types,
                     page_numbers  (→ JSON strings)
    """
    flat: Dict[str, Any] = {}

    # ── root-level scalar fields ──────────────────────────────────
    for key in ("chunk_type", "content_length", "word_count", "created_at"):
        val = chunk.get(key)
        if val is not None:
            flat[key] = val

    # ── metadata sub-dict ─────────────────────────────────────────
    meta = chunk.get("metadata", {})
    for key, val in meta.items():
        if val is None:
            continue
        if isinstance(val, list):
            # Serialise lists → JSON string (searchable via string contains)
            flat[f"meta_{key}"] = json.dumps(val, ensure_ascii=False)
        elif isinstance(val, (str, int, float, bool)):
            flat[f"meta_{key}"] = val
        else:
            flat[f"meta_{key}"] = str(val)

    return flat


def _get_collection(client: chromadb.PersistentClient, collection_name: str):
    """
    Return (or create) the ChromaDB collection.
    Uses DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX Runtime),
    which downloads reliably on Windows without HuggingFace symlink issues.
    """
    ef = DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD  – ingest List of Dicts → ChromaDB
# ═══════════════════════════════════════════════════════════════════════════════

def build_collection(collection_name: str, db_path: str, all_chunks: List[Dict[str, Any]]) -> None:
    """
    Upsert chunks into the ChromaDB collection at the specified path.
    Already-existing chunks are updated (idempotent).
    """
    # Skip very short / empty chunks (they add noise but carry no meaning)
    chunks = [
        c for c in all_chunks
        if len(c.get("content", "").strip()) >= SKIP_MIN_CHARS
    ]
    
    Path(db_path).mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=db_path)
    collection = _get_collection(client, collection_name)

    ids       : List[str]            = []
    documents : List[str]            = []
    metadatas : List[Dict[str, Any]] = []

    for chunk in chunks:
        idx = chunk.get("metadata", {}).get("chunk_index", 0)
        src = chunk.get("metadata", {}).get("source_file", "unknown")
        import hashlib
        # Create a unique ID combining filename and index
        raw_id = f"{src}_{idx}".encode("utf-8")
        chunk_id = f"chunk_{hashlib.md5(raw_id).hexdigest()[:16]}"
        ids.append(chunk_id)
        documents.append(chunk.get("content", "").strip())
        metadatas.append(_flatten_metadata(chunk))

    for i in range(0, len(documents), INGEST_BATCH):
        end = i + INGEST_BATCH
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY  – semantic search over the collection
# ═══════════════════════════════════════════════════════════════════════════════

def query_collection(
    collection_name: str,
    db_path:        str,
    query_text:     str,
    n_results:      int             = 5,
    source_filter:  Optional[str]   = None,   # e.g. "RFP Credit Decision Engine.pdf"
    format_filter:  Optional[str]   = None,   # e.g. ".xlsx"
    type_filter:    Optional[str]   = None,   # e.g. "table"
    sheet_filter:   Optional[str]   = None,   # e.g. "Sheet1"
) -> List[Dict[str, Any]]:
    """
    Semantic search.  Accepts optional metadata filters that narrow the
    result set before re-ranking by embedding similarity.

    Returns a list of result dicts with keys:
      id, score, content, metadata
    """
    if not Path(db_path).exists():
        return []
    client     = chromadb.PersistentClient(path=db_path)
    collection = _get_collection(client, collection_name)

    # Build ChromaDB `where` clause  (AND of all supplied filters)
    where_parts: List[Dict[str, Any]] = []
    if source_filter:
        where_parts.append({"meta_source_file": {"$eq": source_filter}})
    if format_filter:
        where_parts.append({"meta_doc_format":  {"$eq": format_filter}})
    if type_filter:
        where_parts.append({"chunk_type":       {"$eq": type_filter}})
    if sheet_filter:
        where_parts.append({"meta_sheet_name":  {"$eq": sheet_filter}})

    where: Optional[Dict] = None
    if len(where_parts) == 1:
        where = where_parts[0]
    elif len(where_parts) > 1:
        where = {"$and": where_parts}

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results.get("ids"):
        for rid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "id":       rid,
                "score":    round(1 - dist, 4),   # cosine similarity
                "content":  doc,
                "metadata": meta,
            })
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════════

def stats(collection_name: str, db_path: str) -> None:
    """Print collection statistics."""
    client     = chromadb.PersistentClient(path=db_path)
    collection = _get_collection(client, collection_name)

    count = collection.count()
    print(f"\nCollection : {COLLECTION_NAME}")
    print(f"Path       : {Path(CHROMA_PATH).resolve()}")
    print(f"Documents  : {count}")

    if count > 0:
        # Sample the first 10 to show metadata fields
        sample = collection.peek(10)
        print(f"\nSample document IDs : {sample['ids']}")
        if sample.get("metadatas"):
            keys = set()
            for m in sample["metadatas"]:
                keys.update(m.keys())
            print(f"Metadata fields     : {sorted(keys)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _print_results(results: List[Dict[str, Any]]) -> None:
    sep = "-" * 72
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"\n{sep}")
        print(f"  Result #{i}  |  score={r['score']:.4f}  |  id={r['id']}")
        print(f"  Source : {m.get('meta_source_file', '')}  "
              f"[{m.get('meta_doc_format', '')}]")
        if m.get("meta_sheet_name"):
            print(f"  Sheet  : {m['meta_sheet_name']}")
        print(f"  Type   : {m.get('chunk_type', '')}  |  "
              f"pages={m.get('meta_page_numbers', '')}")
        if m.get("meta_headings") and m["meta_headings"] != "[]":
            print(f"  Heading: {m['meta_headings']}")
        print(f"  Content: {r['content'][:300]}{'…' if len(r['content']) > 300 else ''}")
    print(f"\n{sep}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "build":
        json_src = sys.argv[2] if len(sys.argv) > 2 else "5_chunked_rfp_data.json"
        path = Path(json_src)
        if not path.exists():
            print(f"ERROR: JSON file not found: {path.resolve()}")
            sys.exit(1)
        import json
        with open(path, encoding="utf-8") as f:
            all_chunks = json.load(f)
        build_collection("rfp_knowledge_base", "./chroma_data/historical_rfps", all_chunks)

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: python vector.py query \"your question\"")
            sys.exit(1)
        q_text  = sys.argv[2]
        n       = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        results = query_collection("rfp_knowledge_base", "./chroma_data/historical_rfps", q_text, n_results=n)
        print(f"\nQuery: {q_text!r}  |  top {n} results")
        _print_results(results)

    elif cmd == "stats":
        stats("rfp_knowledge_base", "./chroma_data/historical_rfps")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: build | query <text> [n] | stats")
        sys.exit(1)