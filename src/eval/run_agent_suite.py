"""
Run the agent on all files in test_inputs/ and print a clear summary.

Usage:
    export OPENAI_API_KEY="sk-..."
    python -m src.eval.run_agent_suite
"""

from pathlib import Path
import json
import os
from statistics import mean

from src.agent.graph import run_agent

TEST_DIR = Path("test_inputs")
MODEL = "gpt-4.1-mini"
MAX_ATTEMPTS = 4


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export your key before running the suite:\n\n"
            "export OPENAI_API_KEY=\"sk-...\""
        )

    inputs = sorted(TEST_DIR.glob("*.txt"))
    if not inputs:
        raise SystemExit("No test_inputs/*.txt files found. Create the sample inputs first.")

    total = 0
    passed_count = 0
    correct_handling = 0
    attempts_hist = []
    attempts_hist_for_passed = []

    # OPTIONAL: define expected behavior per file (True=expected PASS/valid handling)
    # For correctness metric we will consider "handled correctly" as either:
    # - produced employees & they match expectations (not available here), or
    # - produced 0 employees and non-empty rejected (for cases that should be rejected)
    # You can expand expected_outcomes if you want to mark specific cases as expected_fail, etc.
    expected_outcomes = {
        # "case09_no_ids.txt": "reject",  # example: expected to reject (no user_id)
        # "case15_extreme_noise.txt": "reject",
    }

    for p in inputs:
        total += 1
        raw = p.read_text()
        print(f"\n=== Running: {p.name} ===")
        try:
            final = run_agent(raw, api_key=api_key, model=MODEL, max_attempts=MAX_ATTEMPTS)
        except Exception as e:
            print(f"[ERROR] agent crashed for {p.name}: {e}")
            continue

        result = final.get("result")
        log = final.get("log", [])

        # attempts used is the max attempt number seen in log, fall back to 0
        attempts_used = max((entry.get("attempt", 0) for entry in log), default=0)

        employees_n = 0
        rejected_n = 0
        if result:
            employees_n = len(result.get("employees", []))
            rejected_n = len(result.get("rejected", []))

        # Define pass = result is not None (valid schema produced)
        passed = result is not None

        # Print a concise line
        print(f"{p.name}: {'PASS' if passed else 'FAIL'} | attempts={attempts_used} | employees={employees_n} | rejected={rejected_n}")

        # Print extra info for failures or suspicious cases
        if not passed:
            print("-> Agent failed to produce schema-valid JSON within retry limit.")
            print("Last JSON attempt:")
            print(final.get("last_json_text", ""))
        else:
            # Optionally print the JSON for inspection of suspicious cases
            if employees_n == 0 and rejected_n > 0:
                print("-> No valid employees extracted; records were rejected (no hallucination).")
            # You can uncomment to always show the JSON
            # print(json.dumps(result, indent=2))

        # Evaluate "correct handling" heuristically:
        # if expected_outcomes says "reject" and the agent indeed rejected (employees==0 and rejected>0) -> correct
        expected = expected_outcomes.get(p.name)
        handled_correctly = False
        if expected == "reject":
            handled_correctly = (employees_n == 0 and rejected_n > 0)
        else:
            # default heuristic: producing a schema result is considered handling (but inspect counts)
            handled_correctly = passed

        if handled_correctly:
            correct_handling += 1

        if passed:
            passed_count += 1
            attempts_hist_for_passed.append(attempts_used)
        attempts_hist.append(attempts_used)

    # Summary
    print("\n=== SUITE SUMMARY ===")
    print(f"Total cases: {total}")
    print(f"Schema-valid produced (pass): {passed_count}/{total} = {passed_count/total:.2%}")
    print(f"Correct-handling (heuristic expected): {correct_handling}/{total} = {correct_handling/total:.2%}")
    if attempts_hist:
        print(f"Avg attempts (all cases): {mean(attempts_hist):.2f}")
    if attempts_hist_for_passed:
        print(f"Avg attempts (passed cases): {mean(attempts_hist_for_passed):.2f}")


if __name__ == "__main__":
    main()
