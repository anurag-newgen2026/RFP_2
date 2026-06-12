"""
RFP Intelligence Agent — LangGraph StateGraph (Approach 2: Supervisor Node)

Architecture:
  AgentState (TypedDict) — single shared state for all nodes

  Nodes:
    planner_node    — LLM CALL #1: plan_query() → writes plan + strategy to State
    supervisor_node — PURE CODE: BFS check, sets next_task in State
    tool_node       — PURE CODE: calls ChromaDB/JSON/Web tool, writes results to State
    redefiner_node  — LLM CALL: refines vague sub-query using dep results
    synthesizer_node— LLM CALL (final): synthesizes all results into answer

  Routing (conditional edges from supervisor):
    "tool"        → tool_node        (single ready task, query available)
    "redefiner"   → redefiner_node   (ready task has deps, no refined query yet)
    "synthesizer" → synthesizer_node (all tasks complete)
    [Send(...)]   → tool_node × N    (multiple ready tasks — parallel dispatch)

  Strategy behaviour:
    "none"       — no retrieval needed → goes straight to synthesizer
    "parallel"   — all tasks independent → Send all at once → synthesizer
    "sequential" — BFS one-at-a-time, dep tasks → redefiner → tool → loop
    "mixed"      — BFS, parallel where ready, sequential for dep tasks

Exposes:
  stream_normal_agent(user_query, session_id, session_context) → Generator[tuple]
  run_normal_agent(user_query, session_id, session_context)    → str
"""

import operator
import os
from typing import Annotated, Generator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from src.prompts import REFINER_PROMPT, SYNTHESIZER_PROMPT
from src.query_planner import plan_query
from src.tools import NORMAL_TOOLS, set_session_id

load_dotenv()


# ---------------------------------------------------------------------------
# Tool lookup map — built once at import time
# ---------------------------------------------------------------------------

_TOOL_MAP = {t.name: t for t in NORMAL_TOOLS}


# ---------------------------------------------------------------------------
# AgentState
# Reducers handle concurrent writes from parallel tool_node dispatches.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ── Input (set once at start) ─────────────────────────────────────────
    user_query:      str
    session_id:      str
    session_context: str

    # ── Plan (written by planner_node, read by all) ───────────────────────
    plan:            dict   # ExecutionPlan.model_dump() — tasks, strategy, needs_retrieval
    strategy:        str    # "sequential" | "parallel" | "mixed" | "none"

    # ── Accumulating results (reducers allow parallel writes) ─────────────
    completed:       Annotated[set, lambda a, b: a | b]           # union
    task_results:    Annotated[dict, lambda a, b: {**a, **b}]     # merge
    task_tool:       Annotated[dict, lambda a, b: {**a, **b}]     # merge
    refined_queries: Annotated[dict, lambda a, b: {**a, **b}]     # merge
    events:          Annotated[list, operator.add]                 # append

    # ── Routing hint (set by supervisor, consumed by tool/redefiner) ──────
    next_task:       dict | None

    # ── Final output ─────────────────────────────────────────────────────
    final_answer:    str


# ---------------------------------------------------------------------------
# Helper: call a tool by name
# ---------------------------------------------------------------------------

def _call_tool(tool_name: str, query: str) -> str:
    """
    Invoke a tool by name.
    get_product_context takes no args — all others take {"query": ...}.
    """
    tool = _TOOL_MAP.get(tool_name)
    if not tool:
        return f"[Error: tool '{tool_name}' not found]"
    try:
        if tool_name == "get_product_context":
            return str(tool.invoke({}))
        return str(tool.invoke({"query": query}))
    except Exception as exc:
        return f"[Error calling {tool_name}: {exc}]"


# ---------------------------------------------------------------------------
# Helper: refine a sub-query using dependency results  (Redefiner logic)
# ---------------------------------------------------------------------------

def _refine_query(intent: str, dep_results: list[dict], user_query: str, execution_plan: dict,
                  tool_name: str = "") -> str:
    """
    Convert a planner intent string into a concrete query using dep results.

    For internal ChromaDB tools: generates 10-30 key-term dump for vector similarity.
    For tavily_search: generates a natural-language web search query.
    """
    context = "\n\n".join(
        f"[{r['tool']} result]:\n{r['result']}"
        for r in dep_results
    )

    # if tool_name == "tavily_search":
    #     style_instruction = (
    #         "Write a natural-language web search query (like a full sentence you "
    #         "would type into Google). It should be specific to the topics, "
    #         "technologies, standards, or product names extracted from the results "
    #         "above. Include the current year (2025) and relevant context words like "
    #         "'pricing', 'regulations', 'standards', 'market', 'alternatives', etc. "
    #         "Reply with ONLY the search query string, nothing else."
    #     )
    # else:
    #     style_instruction = (
    #         "Write a precise semantic search query for a vector database.\n\n"
    #         "STEP 1 — Analyse the retrieved content above and identify:\n"
    #         "  - The core domain or industry (e.g., banking, procurement, insurance)\n"
    #         "  - Specific technology names and integration types mentioned\n"
    #         "  - Specific compliance standards, regulations, or certifications\n"
    #         "  - Specific product features, modules, or capabilities requested\n"
    #         "  - Specific SLAs, numeric thresholds, or performance targets\n\n"
    #         "STEP 2 — Construct the query using this format:\n"
    #         "  '[Clear retrieval goal] for [domain]: [specific terms separated by commas]'\n\n"
    #         "  The retrieval goal must describe WHAT you are searching for. Examples:\n"
    #         "  - 'Newgen past RFP responses for B2B procurement marketplace:'\n"
    #         "  - 'RFP technical requirements for banking digital transformation:'\n"
    #         "  - 'Product capabilities covering document workflow automation:'\n\n"
    #         "STRICT RULES:\n"
    #         "  - DO NOT use vague words'\n"
    #         "  - Every term in the query must be a specific, searchable concept\n\n"
    #     )
    
    # prompt = (
    #     f'User question: "{user_query}"\n\n'
    #     f"Results from previous retrieval steps:\n{context}\n\n"
    #     f'Next retrieval intent: "{intent}"\n\n'
    #     + style_instruction
    # )
    # # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # return llm.invoke(prompt).content.strip()


    # 2. Inject the variables into the new prompt
    prompt_text = REFINER_PROMPT.format(
        user_query=user_query,
        execution_plan=str(execution_plan),
        context=context if context else "No previous results.",
        intent=intent,
        tool_name=tool_name
    )
    # 3. Call the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # We pass it as a SystemMessage to enforce strong instructions
    messages = [
        SystemMessage(content=prompt_text)
    ]
    
    return llm.invoke(messages).content.strip()


# ---------------------------------------------------------------------------
# Node 1: planner_node   (LLM CALL #1)
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """
    Calls plan_query() to produce an ExecutionPlan.
    Writes plan, strategy, and the initial "plan" display event to State.
    """
    plan = plan_query(
        user_query=state["user_query"],
        session_context=state.get("session_context", ""),
    )

    if plan is None:
        # Planning failed — treat as no-retrieval and answer directly
        return {
            "plan":     {"needs_retrieval": False, "strategy": "none", "tasks": []},
            "strategy": "none",
            "events":   [],
        }

    # Build plan display items for the Gradio activity panel
    strategy_label = plan.strategy.upper()
    plan_items = []
    for t in sorted(plan.tasks, key=lambda x: x.id):
        dep_str = (
            f" ← after Task {t.depends_on}" if t.depends_on else " [independent]"
        )
        plan_items.append(
            f"[{strategy_label}] Task {t.id}: {t.tool}{dep_str} — {t.sub_query}"
        )

    return {
        "plan":     plan.model_dump(),
        "strategy": plan.strategy,
        "events":   [("plan", plan_items)] if plan_items else [],
    }


# ---------------------------------------------------------------------------
# Node 2: supervisor_node   (PURE CODE — no LLM)
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """
    Pure Python check.
    Sets next_task in State so tool_node and redefiner_node know what to process.
    The actual routing decision (which node to go to) is in routing_function.
    """
    tasks     = state["plan"].get("tasks", [])
    completed = state.get("completed", set())
    refined   = state.get("refined_queries", {})

    all_ids = {t["id"] for t in tasks}
    if completed >= all_ids:
        return {"next_task": None}

    #find all tasks whose dependencies are satisfied and not yet done
    ready = [
        t for t in tasks
        if t["id"] not in completed
        and all(dep in completed for dep in t["depends_on"])
    ]

    if not ready:
        return {"next_task": None}

    # For sequential/mixed: prioritise tasks that need refinement first
    for t in ready:
        if t["depends_on"] and t["id"] not in refined:
            return {"next_task": t}

    return {"next_task": ready[0] if ready else None}


# ---------------------------------------------------------------------------
# Routing function (conditional edge from supervisor)
# ---------------------------------------------------------------------------

def routing_function(state: AgentState) -> str | list:
    """
    Returns the next node name (str) or a list of Send objects for parallel dispatch.

    Decision logic:
      - all done           → "synthesizer"
      - no retrieval       → "synthesizer"
      - parallel/none      → [Send("tool", ...)] for all ready tasks simultaneously
      - sequential/mixed   → "redefiner" if next task needs refinement
                           → "tool" for single task
                           → [Send("tool", ...)] if multiple tasks are ready
    """
    tasks     = state["plan"].get("tasks", [])
    completed = state.get("completed", set())
    strategy  = state.get("strategy", "none")
    refined   = state.get("refined_queries", {})

    all_ids = {t["id"] for t in tasks}

    # Terminate conditions
    if not tasks or not state["plan"].get("needs_retrieval", True):
        return "synthesizer"
    if completed >= all_ids:
        return "synthesizer"

    # Find all currently ready tasks (deps satisfied, not yet completed)
    ready = [
        t for t in tasks
        if t["id"] not in completed
        and all(dep in completed for dep in t["depends_on"])
    ]

    if not ready:
        return "synthesizer"

    # ── Parallel / None ───────────────────────────────────────────────────
    if strategy in ("parallel", "none"):
        if len(ready) == 1:
            return "tool"
        # Dispatch all tasks simultaneously via Send
        return [Send("tool", {**state, "next_task": t}) for t in ready]

    # ── Sequential / Mixed ────────────────────────────────────────────────
    # Check if next_task (set by supervisor) needs refinement first
    for t in ready:
        if t["depends_on"] and t["id"] not in refined:
            return "redefiner"

    # All ready tasks have their queries — dispatch simultaneously
    if len(ready) == 1:
        return "tool"
    return [Send("tool", {**state, "next_task": t}) for t in ready]


# ---------------------------------------------------------------------------
# Node 3: tool_node   (PURE CODE — no LLM)
# ---------------------------------------------------------------------------

def tool_node(state: AgentState) -> dict:
    """
    Direct tool call. Uses refined_query if available, else sub_query.
    Returns task_results, task_tool, completed, and two events.
    """
    task      = state["next_task"]
    task_id   = task["id"]
    tool_name = task["tool"]
    query     = state.get("refined_queries", {}).get(task_id, task["sub_query"])

    result  = _call_tool(tool_name, query)
    preview = result[:3000] + " …" if len(result) > 3000 else result

    return {
        "task_results":  {task_id: result},
        "task_tool":     {task_id: tool_name},
        "completed":     {task_id},
        "events": [
            ("tool_call",   tool_name, {"query": query}),
            ("tool_result", tool_name, preview),
        ],
    }


# ---------------------------------------------------------------------------
# Node 4: redefiner_node   (LLM CALL)
# ---------------------------------------------------------------------------

def redefiner_node(state: AgentState) -> dict:
    """
    Refines a vague sub-query using the actual results from dependency tasks.
    Produces a precise, vector-search-ready query and writes it to refined_queries.
    """
    task = state["next_task"]
    dep_data = [
        {
            "tool":   state.get("task_tool", {}).get(did, "unknown"),
            "result": state.get("task_results", {}).get(did, ""),
        }
        for did in task["depends_on"]
    ]
    refined = _refine_query(
        task["sub_query"], dep_data,
        state["user_query"], tool_name=task["tool"],
        execution_plan=state["plan"],
    )
    return {
        "refined_queries": {task["id"]: refined},
        "events":          [("refined_query", task["tool"], refined)],
    }


# ---------------------------------------------------------------------------
# Node 5: synthesizer_node   (LLM CALL — final)
# ---------------------------------------------------------------------------

def synthesizer_node(state: AgentState) -> dict:
    """
    Reads all task_results from State and calls LLM once to write the final answer.
    If no retrieval was performed (strategy="none"), answers the query directly.
    """
    tasks        = state["plan"].get("tasks", [])
    task_results = state.get("task_results", {})
    task_tool    = state.get("task_tool", {})
    session_ctx  = state.get("session_context", "")

    system_prompt = SYNTHESIZER_PROMPT + ("\n" + session_ctx if session_ctx else "")
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)

    if not tasks or not task_results:
        # No retrieval — answer directly from LLM knowledge
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"]),
        ]
    else:
        ordered = sorted(tasks, key=lambda t: t["id"])
        context_block = "\n\n".join(
            f"[{task_tool.get(t['id'], t['tool'])} result — Task {t['id']}]:\n"
            f"{task_results.get(t['id'], 'No result')}"
            for t in ordered
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"User question: {state['user_query']}\n\n"
                f"Retrieved information:\n{context_block}\n\n"
                "Using ALL the retrieved information above, provide a comprehensive, "
                "well-structured answer."
            )),
        ]

    response = llm.invoke(messages)
    answer   = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "final_answer": answer,
        "events":       [("final_answer", answer)],
    }


# ---------------------------------------------------------------------------
# Build and compile the LangGraph StateGraph
# ---------------------------------------------------------------------------

def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner",     planner_node)
    graph.add_node("supervisor",  supervisor_node)
    graph.add_node("tool",        tool_node)
    graph.add_node("redefiner",   redefiner_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")
    #Fixed edge 
    graph.add_edge("planner",   "supervisor")
    graph.add_edge("tool",      "supervisor")     # loop back after each tool call
    graph.add_edge("redefiner", "supervisor")     # loop back after refinement

    #Conditional edges - routing function to decide next step based on supervisor output
    graph.add_conditional_edges(
        "supervisor",
        routing_function,
        {
            "tool":        "tool",
            "redefiner":   "redefiner",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)
    return graph.compile()


_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Streaming entry point — yields event tuples for Gradio activity panel
# ---------------------------------------------------------------------------

def stream_normal_agent(
    user_query: str,
    session_id: str = "",
    session_context: str = "",
) -> Generator[tuple, None, None]:
    """
    Runs the RFP agent StateGraph and yields structured event tuples.

    Event tuple formats (consumed by app.py):
      ("plan",          items: list[str])          ← Research Plan display
      ("tool_call",     tool_name: str, args: dict)
      ("tool_result",   tool_name: str, preview: str)
      ("refined_query", tool_name: str, query: str) ← Redefiner output
      ("final_answer",  text: str)
    """
    set_session_id(session_id)

    init_state: AgentState = {
        "user_query":      user_query,
        "session_id":      session_id,
        "session_context": session_context,
        "plan":            {},
        "strategy":        "",
        "completed":       set(),
        "task_results":    {},
        "task_tool":       {},
        "refined_queries": {},
        "next_task":       None,
        "events":          [],
        "final_answer":    "",
    }

    # stream_mode="values" yields full state snapshots after each node.
    # We compare event counts to yield only newly added events per snapshot.
    prev_count = 0
    for snapshot in _GRAPH.stream(init_state, stream_mode="values"):
        events = snapshot.get("events", [])
        for ev in events[prev_count:]:
            yield ev
        prev_count = len(events)


# ---------------------------------------------------------------------------
# Blocking call — returns final answer string
# ---------------------------------------------------------------------------

def run_normal_agent(
    user_query: str,
    session_id: str = "",
    session_context: str = "",
) -> str:
    """Run agent and return the final answer string (blocking)."""
    final = ""
    for event in stream_normal_agent(
        user_query, session_id=session_id, session_context=session_context
    ):
        if event[0] == "final_answer":
            final = event[1]
    return final or "I was unable to generate a response. Please try again."
