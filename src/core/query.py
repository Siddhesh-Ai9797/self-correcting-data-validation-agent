from __future__ import annotations

from typing import Any, List
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import math


class FilterSpec(BaseModel):
    column: str
    op: str = Field(..., description="one of: eq, neq, contains, in, gte, lte")
    value: Any

class QuerySpec(BaseModel):
    select: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    distinct: bool = True
    limit: int = 50

SYSTEM_PROMPT = """You are a data query planner.
You receive:
- user_question
- available_columns
- sample_values (small)
Return ONLY valid JSON for QuerySpec with:
- select: columns needed
- filters: list of {column, op, value}
- distinct: true/false
- limit: integer <= 200

Rules:
- Prefer deterministic, simple filters.
- If the user asks for a department like "Artificial Intelligence", filter Department equals that exact string.
- If user says AI department, treat it as Department in ["Artificial Intelligence","AI/ML"] unless user explicitly excludes AI/ML.
- Do NOT invent columns.
"""
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

def _openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def plan_query_with_llm(user_question: str, df: pd.DataFrame, api_key: str, model: str = "gpt-4.1-mini") -> QuerySpec:
    cols = list(df.columns)
    sample_df = df.head(8).copy()
    sample_df = sample_df.where(pd.notna(sample_df), None)
    sample = _json_safe(sample_df.to_dict(orient="records"))



    client = _openai_client(api_key)
    import json  # add at top of file if not present

    payload = {
    "user_question": user_question,
    "available_columns": cols,
    "sample_values": sample,
}

    resp = client.responses.create(
    model=model,
    input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ],
    temperature=0,
    max_output_tokens=600,
)


    text = resp.output_text
    try:
        spec = QuerySpec.model_validate_json(text)
        spec.limit = max(1, min(int(spec.limit), 200))
        return spec
    except ValidationError as e:
        raise ValueError(f"Could not parse model output as QuerySpec JSON. Raw output:\\n{text}\\n\\nError:\\n{e}")

def execute_query(spec: QuerySpec, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for f in spec.filters:
        col = f.column
        if col not in out.columns:
            continue
        op = f.op
        val = f.value

        if op == "eq":
            if val is None:
               out = out[out[col].isna()]
            else:
               out = out[out[col] == val]
        elif op == "neq":
            if val is None:
               out = out[out[col].notna()]
            else:
               out = out[out[col] != val]
        elif op == "contains":
            out = out[out[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op == "in":
            if not isinstance(val, list):
                val = [val]
            out = out[out[col].isin(val)]
        elif op == "gte":
            out = out[pd.to_numeric(out[col], errors="coerce") >= float(val)]
        elif op == "lte":
            out = out[pd.to_numeric(out[col], errors="coerce") <= float(val)]

    if spec.select:
        safe_select = [c for c in spec.select if c in out.columns]
        out = out[safe_select]

    if spec.distinct:
        out = out.drop_duplicates()

    return out.head(spec.limit)

def summarize_results_with_llm(user_question: str, result_df: pd.DataFrame, api_key: str, model: str = "gpt-4.1-mini") -> str:
    client = _openai_client(api_key)
    safe_df = result_df.copy().where(pd.notna(result_df), None)
    preview = _json_safe(safe_df.to_dict(orient="records"))


    import json  # add at top if not present

    payload = {"question": user_question, "results": preview}

    resp = client.responses.create(
    model=model,
    input=[
        {"role": "system", "content": "You are a helpful analyst. Summarize results concisely and accurately."},
        {"role": "user", "content": json.dumps(payload)},
    ],
    temperature=0.2,
    max_output_tokens=500,
)

    return resp.output_text
