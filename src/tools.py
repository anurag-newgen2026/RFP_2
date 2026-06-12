"""
tools.py — Tool definitions for the RFP Intelligence Agent.

Tools (active):
  1. search_historical_rfp   — Permanent historical RFP knowledge base (ChromaDB)
  2. get_new_rfp_context     — Newly uploaded RFP (session-based ChromaDB)
  3. search_session_document — Uploaded supporting doc (session-based ChromaDB)
  4. get_product_context     — Newgen product catalog (JSON file)
  5. tavily_search           — Live web search (Normal + Deep Mode via planner)

Session management:
  set_session_id(sid)  — Called by app.py on file upload
  get_session_id()     — Used internally by session-based tools (2 & 3)

When friends deliver their real ingestion code:
  - Replace scripts/create_sample_dbs.py ingestion logic with their code
  - The src/rag/*.py retrieval modules DO NOT need to change (same signatures)
"""

import json
import os
from pathlib import Path
from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from langchain.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent
_PRODUCTS_PATH = _BASE_DIR / "data" / "newgen_products.json"

# ---------------------------------------------------------------------------
# Session ID management
# Tools 2 and 3 are session-scoped — they only work after a file is uploaded.
# app.py calls set_session_id() when the user uploads a file.
# ---------------------------------------------------------------------------

_current_session_id: str = ""


def set_session_id(sid: str) -> None:
    """Set the active session ID. Called from app.py on every file upload."""
    global _current_session_id
    _current_session_id = sid


def get_session_id() -> str:
    """Return the active session ID."""
    return _current_session_id


# ---------------------------------------------------------------------------
# Tavily client (lazy init)
# ---------------------------------------------------------------------------

_tavily_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set in .env")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# ---------------------------------------------------------------------------
# Helper: fetch a webpage and convert to markdown
# ---------------------------------------------------------------------------

def _fetch_page(url: str, timeout: float = 10.0) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return markdownify(resp.text)
    except Exception as exc:
        return f"[Error fetching {url}: {exc}]"


# ---------------------------------------------------------------------------
# Tool 1 — search_historical_rfp
# Permanent KB: past RFPs Newgen has already responded to.
# Currently uses existing FAISS search as placeholder.
# Replace with ChromaDB function when friend delivers it.
# ---------------------------------------------------------------------------

# ChromaDB retrieval for historical RFPs (permanent KB)
# When friends deliver real ingestion code: only scripts/create_sample_dbs.py changes.
from rag.historical_db import search_historical_rfp as _hist_fn

# ChromaDB retrieval for session-based vector stores
from rag.new_rfp import get_new_rfp_context as _new_rfp_fn
from rag.session_doc import search_session_document as _doc_fn


@tool
def search_historical_rfp(query: str) -> str:
    """
    Searches the permanent historical RFP knowledge base for past approved responses.

    Use this tool when the user asks about:
    - Past RFP answers or what Newgen said in previous projects
    - A specific historical client (SBI, HUDCO, Al Hilal, KFH, Adani, LIC, etc.)
    - Compliance, integration, or feature answers from past submissions
    - Cross-referencing a new requirement with what Newgen has committed to before
    - General knowledge about Newgen's RFP response history

    Returns: Relevant clause-answer pairs with source citations (client, section, score).
    """
    try:
        result = _hist_fn(query=query, top_k=20)
        if not result:
            return "No matching historical RFP data found for this query."
        return result
    except Exception as exc:
        return f"[ERROR] Historical RFP search failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 2 — get_new_rfp_context
# Session-scoped: searches a new RFP uploaded in this session.
# Ingestion (parsing + storing) is triggered by app.py at upload time.
# This tool ONLY does retrieval.
# ---------------------------------------------------------------------------

@tool
def get_new_rfp_context(query: str) -> str:
    """
    Searches the new RFP document uploaded by the user in this session.

    Use this tool when:
    - The user has uploaded a new RFP and asks about its requirements or clauses
    - The user asks to draft responses for the uploaded RFP
    - The user asks to compare the uploaded RFP with historical responses
    - The user wants clarification questions generated for the uploaded RFP
    - The user asks what a specific section of the uploaded RFP says

    Returns: Relevant clauses and requirements from the uploaded RFP,
             with clause numbers, section labels, and page references.

    Important: Only works if the user has uploaded a New RFP in this session.
               Returns a guidance message if no RFP has been uploaded yet.
    """
    session_id = get_session_id()
    if not session_id:
        return (
            "No new RFP has been uploaded in this session. "
            "Please use the Upload Panel on the left to upload an RFP document first, "
            "then ask your question."
        )
    try:
        return _new_rfp_fn(query=query, session_id=session_id)
    except Exception as exc:
        return f"[ERROR] New RFP context retrieval failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 3 — search_session_document
# Session-scoped: searches a supporting document uploaded in this session.
# (PPT, case study, architecture diagram, PDF brochure, etc.)
# ---------------------------------------------------------------------------

@tool
def search_session_document(query: str) -> str:
    """
    Searches a supporting document uploaded in this session.

    Supporting documents include: PowerPoint presentations, case studies,
    architecture diagrams, product brochures, or any reference PDF/DOCX.

    Use this tool when:
    - The user uploads a PPT or PDF and asks about its content
    - The user asks whether any uploaded slide covers a specific RFP requirement
    - The user wants to map uploaded artifacts to RFP clauses (artifact alignment)
    - The user asks which parts of their uploaded document are relevant to a topic
    - The user asks to find gaps — clauses not covered by any uploaded artifact

    Returns: Matching sections or slides from the uploaded document,
             with file name, slide/page reference, and match score.

    Important: Only works if the user has uploaded a Supporting Document in this session.
               Returns a guidance message if no document has been uploaded yet.
    """
    session_id = get_session_id()
    if not session_id:
        return (
            "No supporting document has been uploaded in this session. "
            "Please use the Upload Panel on the left to upload a PPT, PDF, or "
            "case study first, then ask your question."
        )
    try:
        return _doc_fn(query=query, session_id=session_id)
    except Exception as exc:
        return f"[ERROR] Session document search failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 4 — get_product_context (unchanged)
# ---------------------------------------------------------------------------

@tool
def get_product_context() -> str:
    """
    Load and return the complete Newgen Software product portfolio data.

    Returns the full content of newgen_products.json as a formatted string (~24 KB).
    This includes all 8 Newgen modules:
    - NewgenONE Platform
    - NewgenONE Content ORB
    - NewgenONE Digital Process Automation Platform
    - NewgenONE Document Management System
    - Intelligent Document Processing (IDP)
    - NewgenONE Marvin (AI layer)
    - NewgenONE Integration Ecosystem

    Each module includes: core_problems, key_capabilities, features, use_cases, industries, keywords.

    Use this tool whenever the user asks about Newgen's existing products, capabilities,
    what Newgen already covers, or when comparing RFP needs against Newgen offerings.
    """
    try:
        with open(_PRODUCTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        return f"[ERROR] Products file not found at: {_PRODUCTS_PATH}"
    except Exception as exc:
        return f"[ERROR] Failed to load product data: {exc}"


# ---------------------------------------------------------------------------
# Tool 5 — tavily_search (Deep Mode only, unchanged)
# ---------------------------------------------------------------------------

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"],
        InjectedToolArg,
    ] = "general",
) -> str:
    """
    Search the web for live information on a given query using Tavily.

    Fetches real-time results and returns the full webpage content as markdown.
    Use this tool when you need:
    - Current industry trends and market data
    - Competitor product information
    - Emerging technology landscape
    - Any information that cannot be found in the internal RFP or product data

    In Deep Search mode, this tool MUST be called to enrich internal analysis with
    live market intelligence.

    Args:
        query: The search query string to execute.
        max_results: Number of results to return (default: 3).
        topic: Search topic filter — 'general', 'news', or 'finance' (default: 'general').

    Returns:
        Formatted string with search results including page titles, URLs, and full content.
    """
    try:
        client = _get_tavily_client()
        results = client.search(query, max_results=max_results, topic=topic)

        parts = []
        for item in results.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "No title")
            content = _fetch_page(url)
            parts.append(f"## {title}\n**URL:** {url}\n\n{content}\n---")

        if not parts:
            return f"No results found for query: '{query}'"

        return (
            f"Found {len(parts)} result(s) for '{query}':\n\n"
            + "\n".join(parts)
        )
    except Exception as exc:
        return f"[ERROR] Tavily search failed: {exc}"


# ---------------------------------------------------------------------------
# Exported tool lists per mode
# ---------------------------------------------------------------------------

# Normal Mode: 5 tools (internal knowledge + web search via planner when needed)
NORMAL_TOOLS = [
    search_historical_rfp,
    get_new_rfp_context,
    search_session_document,
    get_product_context,
    tavily_search,          # Planner selects this only for external-info queries
]

# Deep Search Mode: 5 tools (adds live web search)
DEEP_TOOLS = [
    search_historical_rfp,
    get_new_rfp_context,
    search_session_document,
    get_product_context,
    tavily_search,
]
