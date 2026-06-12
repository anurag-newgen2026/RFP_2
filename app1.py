"""
app.py — Gradio UI for the RFP Intelligence Agent (Gradio 6.15.x)

Ingestion flow:
  - Upload & Ingest button → handle_upload() generator (2-step: spinner → ✓ Ingested)
  - Real ChromaDB ingestion via src/rag/new_rfp.ingest_new_rfp()
                           and src/rag/session_doc.ingest_session_document()
  - × remove chip → deletes ChromaDB collection for that session
  - Gradio loading icon/spinner fully suppressed (show_progress=False + CSS + MutationObserver)
"""

import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import httpx
import json
import shutil

# ---------------------------------------------------------------------------
# Cleanup orphaned temporary vector DBs on startup
# ---------------------------------------------------------------------------
def _cleanup_orphaned_dbs():
    chroma_base = Path("chroma_data")
    if not chroma_base.exists():
        return
    now = time.time()
    for p in chroma_base.iterdir():
        if p.is_dir() and (p.name.startswith("new_rfp_") or p.name.startswith("session_doc_")):
            # If older than 24 hours (86400 seconds)
            if now - p.stat().st_mtime > 86400:
                try:
                    shutil.rmtree(p)
                    print(f"Cleaned up orphaned DB: {p.name}")
                except Exception as e:
                    print(f"Failed to clean {p.name}: {e}")

_cleanup_orphaned_dbs()


# ---------------------------------------------------------------------------
# Session context builder (moved from agent_router.py)
# ---------------------------------------------------------------------------

def _build_session_context(chips_data: list | None) -> str:
    """
    Build a dynamic session context block from the list of uploaded file chips.
    Only includes chips with status='ingested'.
    """
    if not chips_data:
        return ""

    rfp_files = [
        c for c in chips_data
        if c.get("type") == "📄 New RFP" and c.get("status") == "ingested"
    ]
    doc_files = [
        c for c in chips_data
        if c.get("type") != "📄 New RFP" and c.get("status") == "ingested"
    ]

    if not rfp_files and not doc_files:
        return ""

    lines = [
        "",
        "## SESSION CONTEXT",
        "The following documents have been uploaded and are available for retrieval:",
    ]
    for f in rfp_files:
        lines.append(f'- New RFP: "{f["filename"]}" — use `get_new_rfp_context` to access it')
    for f in doc_files:
        lines.append(f'- Supporting Document: "{f["filename"]}" — use `search_session_document` to access it')
    lines += [
        "",
        "When the user asks any question that could relate to an uploaded document",
        '(including vague requests like "help me answer this", "summarise it",',
        '"what does it say", "draft a response"), automatically use the appropriate',
        "retrieval tool — do not ask the user to re-specify the document.",
    ]
    return "\n".join(lines)

# Historical RFP knowledge base now uses persistent ChromaDB (src/rag/historical_db.py).
# No startup initialization needed — ChromaDB loads lazily on first query.
# To rebuild the sample data: python scripts/create_sample_dbs.py

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; }

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Suppress ALL Gradio loading overlays (⇢ icon + timer + blue border)
   Gradio 6.15.x renders .icon-wrap and .eta-bar inside component wrappers.
   MutationObserver in JS handles any icons that slip through CSS.
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.icon-wrap,
.icon-wrap * { display: none !important; }
.eta-bar,
.eta-bar * { display: none !important; }
.progress-text { display: none !important; }
.progress-bar-wrap { display: none !important; }
.file-preview-holder { display: none !important; }
/* Remove the blue "generating" border Gradio adds to output containers */
.wrap.generating { border-color: #21262d !important; box-shadow: none !important; }
.wrap.pending    { border-color: #21262d !important; box-shadow: none !important; }
/* SVG-based icons Gradio 6 may add */
.wrap.generating > svg,
.wrap.pending > svg { display: none !important; }
.wrap.generating > .loader,
.wrap.pending > .loader { display: none !important; }

/* ━━━━ Chip spinner (our own animation) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
@keyframes chipSpin { to { transform: rotate(360deg); } }

/* ━━━━ CSS-hidden textbox (stays in DOM so JS can find it) ━━━━━━━━━━━━━━━━ */
.remove-chip-hidden {
    position: absolute !important;
    width: 1px !important; height: 1px !important;
    overflow: hidden !important; opacity: 0 !important;
    pointer-events: none !important; z-index: -1 !important;
}

/* ━━━━ Header ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.rfp-header {
    background: linear-gradient(135deg,#1a1f2e 0%,#0d1117 100%);
    border:1px solid #21262d; border-radius:12px;
    padding:20px 28px; margin-bottom:16px;
    display:flex; align-items:center; gap:14px;
}
.rfp-header-logo { font-size:34px; line-height:1; }
.rfp-header-text h1 { font-size:20px; font-weight:700; color:#58a6ff; margin:0 0 4px; }
.rfp-header-text p  { font-size:12px; color:#8b949e; margin:0; }
.panel-label {
    font-size:11px; font-weight:600; letter-spacing:.08em;
    text-transform:uppercase; color:#8b949e; margin-bottom:6px; padding-left:2px;
}

/* ━━━━ Chatbot ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.chatbot-wrap { background:#161b22 !important; border:1px solid #21262d !important; border-radius:10px !important; }

/* ━━━━ Input area ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.input-area {
    background:#161b22; border:1px solid #30363d;
    border-radius:14px; padding:10px 12px 8px;
    margin-top:8px; transition:border-color .2s;
}
.input-area:focus-within { border-color:#58a6ff !important; box-shadow:0 0 0 3px rgba(88,166,255,.08); }
.query-box textarea {
    background:transparent !important; border:none !important;
    color:#e6edf3 !important; font-size:14px !important;
    resize:none !important; box-shadow:none !important; padding:0 !important;
}
.query-box textarea:focus { border:none !important; box-shadow:none !important; }
.query-box textarea::placeholder { color:#484f58 !important; }

/* ━━━━ Toolbar buttons ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.input-toolbar { margin-top:8px; padding-top:8px; border-top:1px solid #21262d; }
.plus-btn button {
    width:34px !important; height:34px !important; min-width:34px !important;
    border-radius:50% !important; background:#21262d !important;
    border:1.5px solid #30363d !important; color:#8b949e !important;
    font-size:20px !important; font-weight:300 !important; padding:0 !important;
    line-height:1 !important; transition:all .2s !important;
}
.plus-btn button:hover { background:#30363d !important; border-color:#58a6ff !important; color:#58a6ff !important; }
.send-btn button {
    background:linear-gradient(135deg,#238636,#2ea043) !important;
    border:none !important; color:#fff !important; border-radius:8px !important;
    font-weight:600 !important; height:36px !important; min-width:100px !important;
    transition:all .2s !important;
}
.send-btn button:hover { filter:brightness(1.15) !important; transform:translateY(-1px) !important; }
.clear-btn button {
    background:#21262d !important; border:1px solid #30363d !important;
    color:#8b949e !important; border-radius:8px !important;
    height:36px !important; min-width:80px !important; transition:all .2s !important;
}
.clear-btn button:hover { border-color:#58a6ff !important; color:#58a6ff !important; }

/* ━━━━ Upload popup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.upload-popup {
    background:#1c2128; border:1px solid #30363d; border-radius:14px;
    padding:16px; margin-top:8px;
    animation:popIn .18s cubic-bezier(.34,1.56,.64,1);
}
@keyframes popIn {
    from { opacity:0; transform:translateY(6px) scale(.98); }
    to   { opacity:1; transform:translateY(0)  scale(1); }
}
.popup-title {
    font-size:12px; font-weight:600; color:#8b949e;
    letter-spacing:.06em; text-transform:uppercase; margin-bottom:12px;
}

/* ━━━━ Type selector cards ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.type-card button {
    background:#0d1117 !important; border:1.5px solid #30363d !important;
    border-radius:10px !important; color:#e6edf3 !important;
    padding:14px 12px !important; font-size:13px !important; font-weight:500 !important;
    height:auto !important; min-height:80px !important;
    white-space:pre-wrap !important; line-height:1.5 !important;
    transition:all .18s !important; text-align:center !important; width:100% !important;
}
.type-card button:hover {
    border-color:#58a6ff !important; background:#141d2e !important;
    transform:translateY(-2px) !important; box-shadow:0 4px 20px rgba(88,166,255,.12) !important;
}

/* ━━━━ File section ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.file-section { border-top:1px solid #21262d; margin-top:12px; padding-top:12px; }
.type-label-text { font-size:12px; margin-bottom:10px; font-weight:500; }
.ingest-btn button {
    background:linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    border:none !important; color:#fff !important; border-radius:8px !important;
    font-weight:600 !important; height:38px !important; width:100% !important;
    margin-top:10px !important; transition:all .2s !important;
}
.ingest-btn button:hover { filter:brightness(1.15) !important; transform:translateY(-1px) !important; }
.popup-error { font-size:12px; color:#f85149; margin-top:8px; min-height:16px; }

/* ━━━━ Chips row container ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
#chips-row-container { border:none !important; outline:none !important; box-shadow:none !important; }
#chips-row-container.generating,
#chips-row-container.pending { border:none !important; box-shadow:none !important; }
#chips-row-container .icon-wrap { display:none !important; }

/* ━━━━ Activity panel ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.activity-panel {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:14px; height:640px; overflow-y:auto; font-size:13px;
    scrollbar-width:thin; scrollbar-color:#30363d transparent;
}
.activity-panel::-webkit-scrollbar { width:5px; }
.activity-panel::-webkit-scrollbar-thumb { background:#30363d; border-radius:4px; }

/* ━━━━ Event cards ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.section-hdr {
    font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    color:#484f58; border-bottom:1px solid #21262d; padding-bottom:5px; margin:10px 0 8px;
}
.section-hdr:first-child { margin-top:0; }
.ev { padding:7px 11px; border-radius:7px; margin-bottom:5px; animation:fadeIn .25s ease; line-height:1.5; }
@keyframes fadeIn { from{opacity:0;transform:translateY(3px)} to{opacity:1;transform:translateY(0)} }
.ev-plan     { background:#1a2035; border-left:3px solid #58a6ff; }
.ev-plan h4  { color:#58a6ff; margin:0 0 5px; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
.ev-plan ol  { margin:0; padding-left:18px; color:#c9d1d9; }
.ev-plan li  { margin-bottom:2px; font-size:12px; }
.ev-subagent { background:#1e1a2e; border-left:3px solid #c084fc; color:#d8b4fe; font-size:12px; }
.ev-subagent b { color:#c084fc; }
.ev-web  { background:#1a2535; border-left:3px solid #38bdf8; color:#7dd3fc; font-size:12px; }
.ev-web b { color:#38bdf8; }
.ev-web code { background:#0f2233; padding:1px 5px; border-radius:4px; }
.ev-tool { background:#1c2520; border-left:3px solid #3fb950; color:#85e89d; font-size:12px; display:flex; justify-content:space-between; align-items:flex-start; }
.ev-tool-left b { color:#3fb950; }
.ev-tool-left code { background:#0f1f10; padding:1px 5px; border-radius:4px; }
.timing { font-size:10px; color:#3fb950; background:#0f1f10; padding:1px 7px; border-radius:10px; white-space:nowrap; flex-shrink:0; margin-left:8px; }
.timing.pending { color:#484f58; background:#1a1a1a; }
.ev-result  { background:#1a1a1a; border-left:3px solid #484f58; color:#8b949e; font-size:11px; }
.ev-result b { color:#6e7681; }
.ev-result code { background:#0d1117; padding:1px 5px; border-radius:4px; }
.ev-reasoning { background:#1c2535; border-left:3px solid #f97316; color:#c9d1d9; font-size:12px; }
.ev-reasoning b { color:#f97316; }
.ev-done { background:#1c2d1c; border-left:3px solid #3fb950; color:#3fb950; font-weight:600; font-size:12px; }
.ev-idle { color:#484f58; font-size:12px; text-align:center; padding:24px; }
.source-card { background:#111827; border:1px solid #1e2a38; border-radius:7px; padding:7px 11px; margin-bottom:5px; font-size:12px; transition:border-color .2s; }
.source-card:hover { border-color:#38bdf8; }
.source-num   { color:#38bdf8; font-weight:700; margin-right:6px; }
.source-title { color:#c9d1d9; font-weight:500; }
.source-domain { color:#484f58; font-size:11px; }
.source-link { color:#58a6ff; font-size:11px; word-break:break-all; }
.source-link:hover { color:#79c0ff; }
.ev-url { background:#111827; border-left:3px solid #1e3a5f; color:#8b949e; font-size:11px; padding:5px 10px; border-radius:5px; margin-bottom:4px; }
.ev-url a { color:#58a6ff; text-decoration:none; }
.ev-url a:hover { text-decoration:underline; }
.status-bar { display:flex; align-items:center; gap:8px; padding:6px 12px; border-radius:6px; background:#161b22; border:1px solid #21262d; font-size:12px; color:#8b949e; margin-bottom:8px; }
.status-dot { width:8px; height:8px; border-radius:50%; background:#484f58; flex-shrink:0; }
.status-dot.active { background:#3fb950; animation:pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
"""

# ---------------------------------------------------------------------------
# JavaScript: removeChip() + MutationObserver to kill Gradio loading icons
# ---------------------------------------------------------------------------

JS_INJECT = """
<script>
(function () {

  /* ── 1. removeChip: JS → Gradio Python via CSS-hidden textbox ──────────── */
  function removeChip(chipId) {
    var container = document.getElementById('remove-chip-id');
    if (!container) { console.warn('[removeChip] #remove-chip-id not found'); return; }
    var ta = container.querySelector('textarea');
    if (!ta) { console.warn('[removeChip] textarea not found'); return; }
    /* Use native property setter so Svelte/React binding picks it up */
    try {
      var desc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
      desc.set.call(ta, chipId);
    } catch (e) {
      ta.value = chipId;
    }
    ta.dispatchEvent(new InputEvent('input',  { bubbles: true, cancelable: true }));
    ta.dispatchEvent(new Event('change', { bubbles: true }));
  }
  window.removeChip = removeChip;

  /* ── 2. MutationObserver: hide Gradio loading icons as they appear ──────── */
  var HIDE_SELECTORS = [
    '.upload-popup',
    '#chips-row-container',
    '.file-section'
  ];

  function killIcons(root) {
    if (!root) return;
    root.querySelectorAll('.icon-wrap, .eta-bar').forEach(function (el) {
      el.style.cssText += ';display:none!important;';
    });
  }

  function attachObserver(el) {
    if (!el || el._gradioKillAttached) return;
    el._gradioKillAttached = true;
    new MutationObserver(function () { killIcons(el); })
        .observe(el, { childList: true, subtree: true });
    killIcons(el);
  }

  function initAll() {
    HIDE_SELECTORS.forEach(function (sel) {
      document.querySelectorAll(sel).forEach(attachObserver);
    });
  }

  /* Run immediately and again after DOM is ready */
  initAll();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  }
  /* Also run a few times early in case Gradio mounts components late */
  [200, 600, 1200, 2500].forEach(function (ms) {
    setTimeout(initAll, ms);
  });

})();
</script>
"""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _domain(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url

def _extract_urls_from_tavily(text):
    return re.findall(r'\*\*URL:\*\*\s*(https?://\S+)', text)

def _extract_sources_from_answer(answer):
    sources = []
    for m in re.finditer(r'\[(\d+)\]\s+([^:\n]+?):\s*(https?://\S+)', answer):
        sources.append({"num": m.group(1), "title": m.group(2).strip(), "url": m.group(3).strip()})
    return sources

def _fmt_result(tool_name, raw):
    kb = round(len(raw.encode()) / 1024, 1)
    if tool_name == "search_historical_rfp":   return f"✓ {kb} KB of historical RFP data"
    if tool_name == "get_new_rfp_context":      return f"✓ {kb} KB from uploaded RFP"
    if tool_name == "search_session_document":  return f"✓ {kb} KB from uploaded document"
    if tool_name == "get_product_context":      return f"✓ {kb} KB of Newgen product portfolio"
    if tool_name == "tavily_search":
        return f"✓ {len(_extract_urls_from_tavily(raw))} web page(s) fetched"
    return (raw[:160] + " …") if len(raw) > 160 else raw


# ---------------------------------------------------------------------------
# Chip rendering  (state-driven)
# ---------------------------------------------------------------------------
# Each chip dict:  {"id": str, "filename": str, "type": str,
#                   "session_id": str, "status": "ingesting"|"ingested"|"error"}

def _render_all_chips(chips: list) -> str:
    """
    Render file chips as HTML.

    Status visuals:
      ingesting → ChatGPT-style circular spinner + orange "⟳ Ingesting…" (no × yet)
      ingested  → coloured icon + green "✓ Ingested" + × remove button
      error     → coloured icon + red "✗ Failed"   + × remove button
    """
    if not chips:
        return ""

    parts = []
    for chip in chips:
        status = chip.get("status", "ingested")
        cid    = chip["id"]
        fname  = chip["filename"]
        short  = fname if len(fname) <= 22 else fname[:19] + "…"
        is_rfp = chip["type"] == "📄 New RFP"
        label  = "New RFP" if is_rfp else "Supporting Doc"
        bg     = "#1d4ed8" if is_rfp else "#dc2626"
        icon   = "📄" if is_rfp else "📎"
        sid    = chip["session_id"]

        card_style = (
            "position:relative;display:inline-flex;align-items:center;gap:10px;"
            "background:#1f2937;border:1px solid #374151;border-radius:12px;"
            "padding:10px 14px 10px 10px;margin:4px;max-width:260px;vertical-align:top;"
        )

        if status == "ingesting":
            # ── Spinner (no × button — not ingested yet) ──────────────────────
            spinner = (
                "width:38px;height:38px;border-radius:50%;"
                "border:3px solid #374151;border-top-color:#3b82f6;"
                "flex-shrink:0;animation:chipSpin 0.9s linear infinite;"
            )
            parts.append(
                f'<div style="{card_style}">'
                f'<div style="{spinner}"></div>'
                f'<div style="min-width:0;">'
                f'<div style="font-size:12px;font-weight:600;color:#e6edf3;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;">{short}</div>'
                f'<div style="font-size:10px;color:#6b7280;margin-top:1px;">{label} · #{sid}</div>'
                f'<div style="font-size:10px;color:#f97316;margin-top:3px;font-weight:500;">⟳ Ingesting…</div>'
                f'</div></div>'
            )
        else:
            # ── Ingested / Error (with × button) ──────────────────────────────
            badge_color = "#3fb950" if status == "ingested" else "#f85149"
            badge_text  = "✓ Ingested" if status == "ingested" else "✗ Failed"

            rm_btn = (
                f'<button onclick="removeChip(\'{cid}\')" title="Remove file"'
                f' style="position:absolute;top:-7px;right:-7px;width:20px;height:20px;'
                f'border-radius:50%;background:#374151;border:1.5px solid #4b5563;'
                f'color:#9ca3af;font-size:12px;cursor:pointer;'
                f'display:flex;align-items:center;justify-content:center;'
                f'line-height:1;padding:0;transition:all .15s;z-index:10;"'
                f' onmouseover="this.style.background=\'#ef4444\';'
                f'this.style.borderColor=\'#ef4444\';this.style.color=\'#fff\'"'
                f' onmouseout="this.style.background=\'#374151\';'
                f'this.style.borderColor=\'#4b5563\';this.style.color=\'#9ca3af\'">×</button>'
            )
            parts.append(
                f'<div style="{card_style}">'
                + rm_btn
                + f'<div style="width:38px;height:38px;border-radius:8px;background:{bg};'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:18px;flex-shrink:0;">{icon}</div>'
                f'<div style="min-width:0;">'
                f'<div style="font-size:12px;font-weight:600;color:#e6edf3;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;">{short}</div>'
                f'<div style="font-size:10px;color:#6b7280;margin-top:1px;">{label} · #{sid}</div>'
                f'<div style="font-size:10px;color:{badge_color};margin-top:3px;font-weight:500;">{badge_text}</div>'
                f'</div></div>'
            )

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:0;padding:4px 0 2px;">'
        + "".join(parts)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Upload handler  (generator → 2-step: spinner first, ✓ Ingested second)
# ---------------------------------------------------------------------------

def handle_upload(files, upload_type: str, session_id: str, chips_data: list):
    """
    Generator function — yields twice:
      1st yield → close popup, show spinner chip(s)     (immediate)
      2nd yield → per-file: mark ingested/error          (after ingestion)

    9 output slots matching the wiring below.
    """
    ALLOWED = {".pdf", ".docx", ".pptx", ".xlsx", ".txt"}

    # Normalise to list
    if   files is None:           files_list = []
    elif isinstance(files, list): files_list = files
    else:                         files_list = [files]

    def _err(msg):
        return (
            session_id, True,
            gr.update(visible=True), gr.update(visible=False),
            chips_data, _render_all_chips(chips_data), msg,
            gr.update(value=None), gr.update(visible=True),
        )

    if not files_list:
        yield _err('<span class="popup-error">⚠️ No file selected.</span>')
        return

    if not session_id:
        session_id = str(uuid.uuid4())[:8].upper()

    # Validate
    valid_files, skipped = [], []
    for f in files_list:
        if Path(f.name).suffix.lower() in ALLOWED:
            valid_files.append(f)
        else:
            skipped.append(Path(f.name).name)

    if not valid_files:
        yield _err(
            f'<span class="popup-error">❌ All files skipped: {", ".join(skipped)}</span>'
        )
        return

    warn = (
        f'<span class="popup-error" style="color:#f97316">⚠️ Skipped: {", ".join(skipped)}</span>'
        if skipped else ""
    )

    # Build pending (spinner) entries
    pending = [
        {
            "id":         str(uuid.uuid4())[:8].upper(),
            "filename":   Path(f.name).name,
            "type":       upload_type,
            "session_id": session_id,
            "status":     "ingesting",
        }
        for f in valid_files
    ]
    all_chips = list(chips_data) + pending

    # ── First yield: close popup, show spinners ───────────────────────────
    yield (
        session_id, False,
        gr.update(visible=False),   # close popup
        gr.update(visible=True),    # show chips row
        all_chips,
        _render_all_chips(all_chips),
        warn,
        gr.update(value=None),      # reset file picker
        gr.update(visible=False),   # hide file section
    )

    # ── Ingest each file into ChromaDB, update chip as it completes ────────
    done_chips = list(chips_data)

    for i, (f, entry) in enumerate(zip(valid_files, pending)):
        try:
            with open(f.name, "rb") as file_data:
                # Send multipart POST request to FastAPI backend
                files_payload = [("files", (Path(f.name).name, file_data, "application/octet-stream"))]
                data_payload = {"session_id": session_id, "upload_type": upload_type}
                resp = httpx.post(
                    "http://localhost:8000/upload",
                    data=data_payload,
                    files=files_payload,
                    timeout=600.0
                )
                resp.raise_for_status()

            done_chips.append({**entry, "status": "ingested"})
        except Exception as exc:
            print(f"[Ingest ERROR] {entry['filename']}: {exc}")
            done_chips.append({**entry, "status": "error"})

        # Remaining files still show spinner
        remaining_spinners = pending[i + 1:]
        display = done_chips + remaining_spinners

        yield (
            session_id, False,
            gr.update(visible=False),
            gr.update(visible=True),
            display,
            _render_all_chips(display),
            warn,
            gr.update(),
            gr.update(),
        )

    # Final clean state (all done, no spinners)
    yield (
        session_id, False,
        gr.update(visible=False),
        gr.update(visible=True),
        done_chips,
        _render_all_chips(done_chips),
        warn,
        gr.update(),
        gr.update(),
    )


# ---------------------------------------------------------------------------
# Remove-chip handler  (JS → hidden textbox → this function)
# ---------------------------------------------------------------------------

def remove_chip_handler(chip_id: str, chips_data: list, session_id: str):
    """
    Remove a chip by ID.
    Returns: new chips_data, new HTML, chips_row visibility, clear textbox.
    """
    chip_id = (chip_id or "").strip()
    if not chip_id:
        return chips_data, _render_all_chips(chips_data), gr.update(visible=bool(chips_data)), ""

    # ── Delete from ChromaDB when user removes a chip ────────────────────────
    removed = next((c for c in chips_data if c["id"] == chip_id), None)
    if removed:
        try:
            httpx.delete(
                f"http://localhost:8000/collection/{removed['session_id']}",
                params={"upload_type": removed["type"]},
                timeout=10.0
            )
        except Exception as exc:
            print(f"[Remove chip] Backend deletion error: {exc}")
    # ─────────────────────────────────────────────────────────────────────────

    new_chips = [c for c in chips_data if c["id"] != chip_id]
    html      = _render_all_chips(new_chips)
    visible   = bool(new_chips)

    return new_chips, html, gr.update(visible=visible), ""


# ---------------------------------------------------------------------------
# Popup toggle / type-select helpers
# ---------------------------------------------------------------------------

def toggle_upload_panel(is_open: bool):
    new_state = not is_open
    return new_state, gr.update(visible=new_state)

def select_type_rfp():
    return (
        "📄 New RFP",
        gr.update(visible=True),
        '<span class="type-label-text" style="color:#3fb950">📄 New RFP selected — choose files below</span>',
    )

def select_type_doc():
    return (
        "📎 Supporting Document",
        gr.update(visible=True),
        '<span class="type-label-text" style="color:#c084fc">📎 Supporting Document selected — choose files below</span>',
    )


# ---------------------------------------------------------------------------
# Activity HTML builder
# ---------------------------------------------------------------------------

def _build_activity_html(steps, url_visits, sources, running=False) -> str:
    if not steps:
        dot = "🟢" if running else "⬜"
        return (
            f'<div class="ev-idle">{dot} Waiting for agent activity…</div>'
        )

    parts = ['<div class="section-hdr">🔄 Agent Steps</div>']
    for ev, elapsed in steps:
        etype = ev[0]
        if etype == "plan":
            items = "".join(f"<li>{t}</li>" for t in ev[1])
            parts.append(f'<div class="ev ev-plan"><h4>📋 Research Plan</h4><ol>{items}</ol></div>')
        elif etype == "subagent":
            parts.append(f'<div class="ev ev-subagent">🤖 <b>Sub-Agent:</b> {ev[1]}</div>')
        elif etype == "web_search":
            parts.append(f'<div class="ev ev-web">🌐 <b>Searching:</b> <code>{ev[1]}</code></div>')
        elif etype == "refined_query":
            # Show the LLM-refined query used for a dependent tool
            # Appears between the Research Plan and the tool call card
            tool_label = ev[1]
            refined    = ev[2]
            parts.append(
                f'<div class="ev ev-reasoning">'
                f'🔍 <b>Refined query</b> for <code>{tool_label}</code>:'
                f'<br><span style="color:#c9d1d9;font-style:italic;padding-left:12px">'
                f'&ldquo;{refined}&rdquo;</span></div>'
            )
        elif etype == "tool_call":
            if ev[1] == "tavily_search": continue   # shown via refined_query card instead
            parts.append(
                f'<div class="ev ev-tool"><span class="ev-tool-left">🔧 <b>Tool:</b>'
                f' <code>{ev[1]}</code></span>'
                f'<span class="timing pending">timing…</span></div>'
            )
        elif etype == "tool_result":
            t = f'<span class="timing">{elapsed:.2f}s</span>' if elapsed else ""
            parts.append(
                f'<div class="ev ev-result">📄 <b>Result</b> <code>{ev[1]}</code>:'
                f' {_fmt_result(ev[1], ev[2])} {t}</div>'
            )
        elif etype == "tavily_result":
            t = f'<span class="timing">{elapsed:.2f}s</span>' if elapsed else ""
            parts.append(
                f'<div class="ev ev-result">📄 <b>Result</b>'
                f' <code>tavily_search</code>: {ev[1]} {t}</div>'
            )
        elif etype == "reasoning":
            parts.append(f'<div class="ev ev-reasoning">💡 <b>Reasoning:</b> {ev[1]}</div>')
        elif etype == "final_answer":
            parts.append('<div class="ev ev-done">✅ Answer generated</div>')

    if url_visits:
        parts.append('<div class="section-hdr">🌐 Sites Searched</div>')
        seen = set()
        for url in url_visits:
            if url in seen: continue
            seen.add(url)
            dom = _domain(url)
            short_url = url if len(url) <= 60 else url[:57] + "…"
            parts.append(
                f'<div class="ev-url"><a href="{url}" target="_blank">🔗 {dom}</a>'
                f'<br><span style="color:#30363d">{short_url}</span></div>'
            )

    if sources:
        parts.append('<div class="section-hdr">📚 Sources</div>')
        for s in sources:
            dom = _domain(s["url"])
            parts.append(
                f'<div class="source-card">'
                f'<span class="source-num">[{s["num"]}]</span>'
                f'<span class="source-title">{s["title"]}</span><br>'
                f'<span class="source-domain">{dom}</span> · '
                f'<a class="source-link" href="{s["url"]}" target="_blank">'
                f'{s["url"][:70]}{"…" if len(s["url"])>70 else ""}</a></div>'
            )

    return "\n".join(parts)


def _status_html(running: bool) -> str:
    dot = "status-dot active" if running else "status-dot"
    return (
        f'<div class="status-bar">'
        f'<span class="{dot}"></span>'
        f'<span>{"Processing…" if running else "Ready"}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Streaming chat handler
# ---------------------------------------------------------------------------

def respond(message: str, history: list, session_id: str,
            chips_data: list | None = None):
    if not message or not message.strip():
        yield history, _build_activity_html([], [], []), _status_html(False)
        return

    history = history + [{"role": "user", "content": message}]
    yield history, _build_activity_html([], [], [], running=True), _status_html(True)

    steps, url_visits, sources, final_answer = [], [], [], ""
    pending: dict = defaultdict(deque)

    ctx = _build_session_context(chips_data)

    try:
        with httpx.stream(
            "POST",
            "http://localhost:8000/chat",
            json={"message": message, "session_id": session_id, "session_context": ctx},
            timeout=120.0
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                
                now   = time.perf_counter()
                etype = event[0]
                if etype == "tool_call":
                    pending[event[1]].append(now)
                    steps.append((event, None))
                elif etype == "tool_result":
                    name, raw = event[1], event[2]
                    elapsed   = now - pending[name].popleft() if pending[name] else None
                    if name == "tavily_search":
                        urls = _extract_urls_from_tavily(raw)
                        url_visits.extend(urls)
                        steps.append((("tavily_result", f"✓ {len(urls)} web page(s) fetched"), elapsed))
                    else:
                        steps.append((event, elapsed))
                elif etype == "web_search":
                    pending["tavily_search"].append(now)
                    steps.append((event, None))
                elif etype == "final_answer":
                    final_answer = event[1]
                    sources      = _extract_sources_from_answer(final_answer)
                    steps.append((event, None))
                else:
                    steps.append((event, None))

                still = etype != "final_answer"
                yield history, _build_activity_html(steps, url_visits, sources, still), _status_html(still)
    except Exception as e:
        steps.append((("reasoning", f"Error connecting to backend API: {e}"), None))
        yield history, _build_activity_html(steps, url_visits, sources, False), _status_html(False)

    bot = final_answer or "⚠️ No response. Check your API keys."
    history = history + [{"role": "assistant", "content": bot}]
    yield history, _build_activity_html(steps, url_visits, sources, False), _status_html(False)


def clear_all():
    return [], _build_activity_html([], [], []), _status_html(False)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER = """
<div class="rfp-header">
  <div class="rfp-header-logo">🧠</div>
  <div class="rfp-header-text">
    <h1>RFP Intelligence Agent — Newgen Software</h1>
    <p>Historical RFP Search · New RFP Ingestion · Artifact Alignment · AI-Powered Analysis</p>
  </div>
</div>
"""

EXAMPLES = [
    "What did Newgen respond to Al Hilal's collections management RFP?",
    "Summarise the key requirements from the uploaded RFP",
    "Draft a response for the loan origination clause in the uploaded RFP",
    "Which slides in the uploaded PPT cover the SWIFT integration requirement?",
    "What capability gaps exist between the uploaded RFP and Newgen's products?",
    "Generate clarification questions for the uploaded RFP",
    "Which Newgen products best cover the uploaded RFP requirements?",
]

# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="RFP Intelligence Agent — Newgen") as demo:

    # JS injection (removeChip + MutationObserver)
    gr.HTML(JS_INJECT)
    gr.HTML(HEADER)

    # ── Persistent state ───────────────────────────────────────────────────
    session_state     = gr.State("")
    upload_panel_open = gr.State(False)
    upload_type_state = gr.State("📄 New RFP")
    chips_data_state  = gr.State([])   # list of chip dicts

    # CSS-hidden textbox stays in the DOM so JS removeChip() can find it
    # DO NOT set visible=False — hidden via CSS class instead
    remove_chip_id = gr.Textbox(
        value="", visible=True,
        elem_id="remove-chip-id",
        elem_classes=["remove-chip-hidden"],
    )

    with gr.Row(equal_height=False):

        # ── LEFT: Chat column ────────────────────────────────────────────────
        with gr.Column(scale=6):
            gr.HTML('<div class="panel-label">💬 Chat</div>')

            chatbot = gr.Chatbot(
                value=[], height=420,
                show_label=False,
                elem_classes=["chatbot-wrap"],
                avatar_images=(None, "🧠"),
            )

            # ── Upload popup (above input bar, hidden by default) ────────────
            with gr.Column(
                visible=False,
                elem_classes=["upload-popup"],
                show_progress=False,       # ← prevents loading icon on this column
            ) as upload_popup_col:

                gr.HTML('<div class="popup-title">📎 Upload Document</div>')

                with gr.Row():
                    rfp_type_btn = gr.Button(
                        "📄  New RFP\n\nClient's requirements document",
                        elem_classes=["type-card"],
                    )
                    doc_type_btn = gr.Button(
                        "📎  Supporting Document\n\nPPT · case study · diagram",
                        elem_classes=["type-card"],
                    )

                with gr.Column(
                    visible=False,
                    elem_classes=["file-section"],
                    show_progress=False,   # ← prevents loading icon on this column
                ) as file_section_col:

                    type_label_display = gr.HTML("", show_progress="hidden")
                    file_upload = gr.File(
                        label="Select one or more files",
                        file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".txt"],
                        file_count="multiple",
                    )
                    upload_ingest_btn = gr.Button(
                        "⬆  Upload & Ingest", elem_classes=["ingest-btn"]
                    )
                    popup_error_html = gr.HTML(
                        '<span class="popup-error"></span>',
                        show_progress="hidden",
                    )

            # ── File chips row ───────────────────────────────────────────────
            with gr.Row(
                visible=False,
                elem_id="chips-row-container",
                show_progress=False,       # ← prevents loading icon on this row
            ) as file_chips_row:
                file_chip_html = gr.HTML("", show_progress="hidden")

            # ── Input area ───────────────────────────────────────────────────
            with gr.Column(elem_classes=["input-area"]):
                query_box = gr.Textbox(
                    placeholder="Ask about RFP requirements, draft responses, artifact alignment…",
                    show_label=False, lines=2, max_lines=6,
                    elem_classes=["query-box"],
                )
                with gr.Row(equal_height=True, elem_classes=["input-toolbar"]):
                    plus_btn  = gr.Button("+", elem_classes=["plus-btn"], scale=0, min_width=36)
                    gr.HTML('<span style="flex:1"></span>')
                    send_btn  = gr.Button("Send ➤", elem_classes=["send-btn"],
                                          variant="primary", scale=0, min_width=110)
                    clear_btn = gr.Button("🗑 Clear", elem_classes=["clear-btn"],
                                          scale=0, min_width=90)

            gr.Examples(examples=EXAMPLES, inputs=query_box, label="💡 Example queries")

        # ── RIGHT: Activity column ───────────────────────────────────────────
        with gr.Column(scale=4):
            gr.HTML('<div class="panel-label">⚡ Live Agent Activity</div>')
            status_box   = gr.HTML(value=_status_html(False))
            activity_box = gr.HTML(
                value=_build_activity_html([], [], []),
                elem_classes=["activity-panel"],
            )

    # ── Event wiring ──────────────────────────────────────────────────────────

    # 1. "+" toggles popup
    plus_btn.click(
        fn=toggle_upload_panel,
        inputs=[upload_panel_open],
        outputs=[upload_panel_open, upload_popup_col],
        queue=False,
    )

    # 2. "New RFP" type card
    rfp_type_btn.click(
        fn=select_type_rfp,
        outputs=[upload_type_state, file_section_col, type_label_display],
        queue=False,
    )

    # 3. "Supporting Document" type card
    doc_type_btn.click(
        fn=select_type_doc,
        outputs=[upload_type_state, file_section_col, type_label_display],
        queue=False,
    )

    # 4. Upload & Ingest — generator yields spinner then ✓ Ingested
    upload_ingest_btn.click(
        fn=handle_upload,
        inputs=[file_upload, upload_type_state, session_state, chips_data_state],
        outputs=[
            session_state,
            upload_panel_open,
            upload_popup_col,
            file_chips_row,
            chips_data_state,
            file_chip_html,
            popup_error_html,
            file_upload,
            file_section_col,
        ],
        queue=True,        # generator needs queue=True to stream intermediate yields
    )

    # 5. × remove chip — JS sets hidden textbox value → .change() fires Python
    remove_chip_id.change(
        fn=remove_chip_handler,
        inputs=[remove_chip_id, chips_data_state, session_state],
        outputs=[chips_data_state, file_chip_html, file_chips_row, remove_chip_id],
        queue=False,
    )

    # 6. Chat send
    def _submit(msg, hist, sid, chips):
        yield from respond(msg, hist, sid, chips_data=chips)

    send_btn.click(
        fn=_submit,
        inputs=[query_box, chatbot, session_state, chips_data_state],
        outputs=[chatbot, activity_box, status_box],
        queue=True,
    ).then(fn=lambda: "", outputs=query_box)

    query_box.submit(
        fn=_submit,
        inputs=[query_box, chatbot, session_state, chips_data_state],
        outputs=[chatbot, activity_box, status_box],
        queue=True,
    ).then(fn=lambda: "", outputs=query_box)

    # 7. Clear
    clear_btn.click(fn=clear_all, outputs=[chatbot, activity_box, status_box])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.queue(max_size=5).launch(
        server_name="localhost",
        #server_port=7860,
        share=True,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(),
    )
