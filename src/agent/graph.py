from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from src.agent.schemas import ExtractedData


# ---------- JSON safety ----------
def _json_safe(obj):
    """Recursively convert NaN/inf to None so payload becomes valid JSON."""
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ---------- OpenAI helper ----------
def _openai_client(api_key: str):
    from openai import OpenAI

    # robust for flaky networks
    return OpenAI(api_key=api_key, timeout=60, max_retries=5)


EXTRACT_SYSTEM = """You are a data extraction + validation agent.

Your job: convert messy text into STRICT JSON that matches this schema:

{
  "employees": [
    {
      "user_id": int,
      "name": string,
      "age": int|null,
      "email": string|null,
      "salary": number|null,
      "join_date": "YYYY-MM-DD"|null,
      "department": one of ["Artificial Intelligence","AI/ML","Machine Learning","Data Science"],
      "performance_score": number|null (0..10),
      "location": string|null,
      "job_title": string|null
    }
  ],
  "rejected": [
    { "raw_record": string, "reasons": [string, ...] }
  ]
}

CRITICAL RULES (NO HALLUCINATION):
- NEVER invent user_id. If user_id is missing/uncertain, DO NOT guess.
  Put that record into "rejected" with reason "missing user_id".
- NEVER guess values from vague text like "maybe", "around", "probably", "approx".
  Use null for uncertain optional fields.
- If a record cannot be made schema-valid WITHOUT guessing required fields, reject it.
- Do not fabricate emails or domains. If email is invalid -> null (or reject only if required, but email is optional here).

Normalization rules:
- Output JSON ONLY, no markdown.
- If a field is missing, set it to null (not empty string).
- Normalize department values:
  AI/ai/Artificial Intelligence -> "Artificial Intelligence"
  AI/ML -> "AI/ML"
  ML/Machine Learning -> "Machine Learning"
  DataScience/Data science -> "Data Science"
- Convert word numbers (e.g., "twenty nine") to integers when clear.
- Convert dates to ISO YYYY-MM-DD if possible, else null.
- Salary: remove $ and commas; if missing, null.
- performance_score must be 0..10; if value is out of range or unclear -> null.
"""

CORRECT_SYSTEM = """You are a self-correcting data validation agent.

You will be given:
- the previous JSON you produced
- a validation error message describing why it failed

Fix the JSON to satisfy the schema.

CRITICAL RULES (NO HALLUCINATION):
- NEVER invent user_id. If user_id is missing/uncertain, reject the record instead of guessing.
- NEVER guess uncertain values (maybe/around/probably). Use null for optional fields.
- Prefer moving problematic records to "rejected" with clear reasons rather than fabricating data.

Rules:
- Output JSON ONLY.
- Keep valid records in "employees".
- Put non-fixable records in "rejected" with reasons.
- Use null for missing fields (not empty strings).
"""


# ---------- LangGraph State ----------
class AgentState(TypedDict):
    raw_text: str
    attempt: int
    max_attempts: int
    last_json_text: str
    validation_error: str
    result: Optional[Dict[str, Any]]
    log: List[Dict[str, Any]]


def _llm_extract(state: AgentState, api_key: str, model: str) -> AgentState:
    client = _openai_client(api_key)
    payload = {"raw_text": state["raw_text"]}

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0,
        max_output_tokens=1400,
    )
    out = (resp.output_text or "").strip()

    state["last_json_text"] = out
    state["log"].append({"step": "extract", "attempt": state["attempt"], "output": out[:2000]})
    return state


def _validate(state: AgentState) -> AgentState:
    try:
        data = ExtractedData.model_validate_json(state["last_json_text"])
        state["result"] = _json_safe(data.model_dump())
        state["validation_error"] = ""
        state["log"].append({"step": "validate", "attempt": state["attempt"], "status": "pass"})
    except ValidationError as e:
        state["result"] = None
        state["validation_error"] = str(e)
        state["log"].append(
            {
                "step": "validate",
                "attempt": state["attempt"],
                "status": "fail",
                "error": state["validation_error"][:2000],
            }
        )
    return state


def _llm_correct(state: AgentState, api_key: str, model: str) -> AgentState:
    client = _openai_client(api_key)
    payload = {
        "previous_json": state["last_json_text"],
        "validation_error": state["validation_error"],
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": CORRECT_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0,
        max_output_tokens=1400,
    )
    out = (resp.output_text or "").strip()

    state["last_json_text"] = out
    state["log"].append({"step": "correct", "attempt": state["attempt"], "output": out[:2000]})
    return state


def _should_retry(state: AgentState) -> str:
    if state["result"] is not None:
        return "finalize"
    if state["attempt"] >= state["max_attempts"]:
        return "finalize"
    return "retry"


def build_graph(api_key: str, model: str):
    g = StateGraph(AgentState)

    g.add_node("extract", lambda s: _llm_extract(s, api_key, model))
    g.add_node("validate", _validate)
    g.add_node("correct", lambda s: _llm_correct(s, api_key, model))

    g.set_entry_point("extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges(
        "validate",
        _should_retry,
        {"retry": "correct", "finalize": END},
    )

    # after correcting, increment attempt then validate again
    def inc_attempt(state: AgentState) -> AgentState:
        state["attempt"] += 1
        return state

    g.add_node("inc_attempt", inc_attempt)
    g.add_edge("correct", "inc_attempt")
    g.add_edge("inc_attempt", "validate")

    return g.compile()


def run_agent(raw_text: str, api_key: str, model: str = "gpt-4.1-mini", max_attempts: int = 3):
    graph = build_graph(api_key, model)
    init: AgentState = {
        "raw_text": raw_text,
        "attempt": 1,
        "max_attempts": max_attempts,
        "last_json_text": "",
        "validation_error": "",
        "result": None,
        "log": [],
    }
    final_state = graph.invoke(init)
    return final_state
