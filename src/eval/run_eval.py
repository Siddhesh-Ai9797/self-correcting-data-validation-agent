from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.cleaning import clean_dataframe
from src.core.query import QuerySpec, FilterSpec, execute_query, plan_query_with_llm


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    details: str


def _load_benchmarks(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _spec_from_dict(d: Dict[str, Any]) -> QuerySpec:
    filters = [FilterSpec(**f) for f in d.get("filters", [])]
    return QuerySpec(
        select=d.get("select", []),
        filters=filters,
        distinct=bool(d.get("distinct", True)),
        limit=int(d.get("limit", 50)),
    )


def _check_expected(result_df: pd.DataFrame, expected: Dict[str, Any]) -> Tuple[bool, str]:
    et = expected.get("type")
    if et == "set_equals":
        col = expected["column"]
        want = set(expected["values"])
        if col not in result_df.columns:
            return False, f"Missing expected column '{col}'. Columns: {list(result_df.columns)}"
        got = set([x for x in result_df[col].dropna().astype(str).tolist()])
        missing = want - got
        extra = got - want
        if missing or extra:
            return False, f"Set mismatch. Missing={sorted(missing)} Extra={sorted(extra)}"
        return True, "OK"
    if et == "row_count_gte":
        min_rows = int(expected["min_rows"])
        n = len(result_df)
        return (n >= min_rows), f"Rows={n}, expected >= {min_rows}"
    if et == "row_count_equals":
        want = int(expected["rows"])
        n = len(result_df)
        return (n == want), f"Rows={n}, expected == {want}"
    return False, f"Unknown expected.type '{et}'"


def run(args: argparse.Namespace) -> int:
    bench = _load_benchmarks(Path(args.benchmarks))
    df_raw = pd.read_csv(args.csv)
    df, report = clean_dataframe(df_raw)

    results: List[CaseResult] = []

    for case in bench["cases"]:
        cid = case["id"]
        mode = case.get("mode", "spec")
        expected = case["expected"]

        try:
            if mode == "spec":
                spec = _spec_from_dict(case["spec"])
            elif mode == "llm":
                if not args.api_key and not os.getenv("OPENAI_API_KEY"):
                    results.append(CaseResult(cid, False, "No API key for LLM mode"))
                    continue
                api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
                spec = plan_query_with_llm(case["question"], df, api_key=api_key, model=args.model)
            else:
                results.append(CaseResult(cid, False, f"Unknown mode '{mode}'"))
                continue

            out = execute_query(spec, df)
            ok, details = _check_expected(out, expected)
            results.append(CaseResult(cid, ok, details))
        except Exception as e:
            results.append(CaseResult(cid, False, f"Exception: {e}"))

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n=== Cleaning report ===")
    print({"rows": report.rows, "fixes": report.fixes, "warnings": report.warnings})

    print("\n=== Benchmark results ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.case_id}: {r.details}")

    print(f"\nSummary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run benchmark evaluation for the AI Data Validation Agent")
    p.add_argument("--csv", required=True, help="Path to CSV dataset")
    p.add_argument("--benchmarks", default="src/eval/benchmarks.json", help="Path to benchmarks.json")
    p.add_argument("--api-key", default="", help="OpenAI API key (optional; only needed for llm-mode cases)")
    p.add_argument("--model", default="gpt-4.1-mini", help="Model for llm-mode cases")
    raise SystemExit(run(p.parse_args()))
