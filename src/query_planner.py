"""
src/query_planner.py

Query Planning layer — Planner Node (LLM CALL #1 in StateGraph).

Flow:
  1. planner_node() in normal_agent.py calls plan_query()
  2. Returns an ExecutionPlan (Pydantic model, guaranteed valid structure)
  3. planner_node writes plan.model_dump() + strategy to AgentState
  4. supervisor_node reads AgentState directly — no system prompt injection needed
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.prompts import QUERY_PLANNER_PROMPT

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic models — guarantee structured output from LLM
# ---------------------------------------------------------------------------

class RetrievalTask(BaseModel):
    id: int = Field(
        description=(
            "Task number starting from 1. Tasks with lower IDs that others "
            "depend on should have smaller numbers."
        )
    )
    tool: Literal[
        "search_historical_rfp",
        "get_new_rfp_context",
        "search_session_document",
        "get_product_context",
        "tavily_search",
    ] = Field(description="The tool to call for this retrieval task.")
    sub_query: str = Field(
        description=(
            "A focused query for this tool's specific retrieval goal only. "
            "10-30 words of KEY TERMS (technology names, requirement names, standards). "
            "For sequential tasks: describe intent — the ReAct agent refines this "
            "at execution time using actual results from dependency tasks. "
            "For parallel tasks: must be complete and specific right now."
        )
    )
    depends_on: list[int] = Field(
        default=[],
        description=(
            "IDs of tasks that MUST complete before this task. "
            "Empty list = task can start immediately (runs in parallel with "
            "other empty-depends_on tasks). "
            "Non-empty = task waits for those results."
        ),
    )
    reason: str = Field(
        description=(
            "One sentence: why this tool was chosen and why it has these dependencies."
        )
    )


class ExecutionPlan(BaseModel):
    needs_retrieval: bool = Field(
        description=(
            "False ONLY if the query can be answered entirely from general LLM "
            "knowledge without any tool calls."
        )
    )
    strategy: Literal["parallel", "sequential", "mixed", "none"] = Field(
        description=(
            "parallel: all tasks independent (all depends_on are empty). "
            "sequential: each task depends on the previous. "
            "mixed: Task 1 sequential first, then Tasks 2+ fan out in parallel. "
            "none: no retrieval needed."
        )
    )
    tasks: list[RetrievalTask] = Field(
        default=[],
        description=(
            "Ordered list of retrieval tasks. Tasks with empty depends_on can "
            "run simultaneously. Tasks with non-empty depends_on wait for their "
            "dependencies to complete."
        ),
    )


# ---------------------------------------------------------------------------
# Planner LLM (lighter/faster model for planning — not reasoning)
# ---------------------------------------------------------------------------

def _get_planner_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    ).with_structured_output(ExecutionPlan)


# ---------------------------------------------------------------------------
# Public function — called by normal_agent.py
# ---------------------------------------------------------------------------

def plan_query(user_query: str, session_context: str = "") -> ExecutionPlan | None:
    """
    Analyze a user query and produce a structured ExecutionPlan.

    Args:
        user_query:      The user's original message.
        session_context: The current session context string (e.g., uploaded files,
                         active session ID). Injected into the planner prompt so it
                         knows which session-based tools are available.

    Returns:
        ExecutionPlan if successful.
        None if the LLM call fails (caller should fall back to normal ReAct behaviour).
    """
    try:
        prompt = QUERY_PLANNER_PROMPT.format(
            user_query=user_query,
            session_context=session_context if session_context else "No files uploaded this session.",
        )
        llm = _get_planner_llm()
        plan: ExecutionPlan = llm.invoke(prompt)
        return plan
    except Exception as exc:
        # Fail silently — normal_agent.py falls back to unplanned ReAct
        print(f"[QueryPlanner] Planning failed, falling back to standard ReAct: {exc}")
        return None


