import os
import json
import streamlit as st
import pandas as pd
from src.agent.graph import run_agent
from src.core.cleaning import clean_dataframe
from src.core.query import plan_query_with_llm, execute_query, summarize_results_with_llm
from src.core.security import basic_injection_check

st.set_page_config(page_title="AI Data Validation Agent (Industry-ready)", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Validation Agent (Industry-ready)")
st.caption("Deterministic pandas answers + LLM planning/summarization (no LLM guessing).")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY",""), type="password")
    model = st.selectbox("Model", ["gpt-4.1-mini","gpt-4.1","gpt-4o-mini","gpt-4o"], index=0)
    st.divider()
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.markdown("**Tip:** Clean the data first, then ask questions.")

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "clean_report" not in st.session_state:
    st.session_state.clean_report = None

tab1, tab2, tab3 = st.tabs(["📄 Data cleaning", "💬 Ask questions", "🧠 Self-correcting agent"])


with tab1:
    st.subheader("1) Upload & clean")
    if uploaded is None:
        st.info("Upload a CSV from the sidebar.")
    else:
        df = pd.read_csv(uploaded)
        st.session_state.df_raw = df
        st.write("Raw preview")
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("Clean & normalize", type="primary"):
            dfc, report = clean_dataframe(df)
            st.session_state.df_clean = dfc
            st.session_state.clean_report = report

        if st.session_state.df_clean is not None:
            st.success("Cleaned dataset ready ✅")
            report = st.session_state.clean_report
            st.write("Cleaning report")
            st.json({"rows": report.rows, "fixes": report.fixes, "warnings": report.warnings})
            st.write("Clean preview")
            st.dataframe(st.session_state.df_clean, use_container_width=True)

            csv_bytes = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
            st.download_button("Download cleaned CSV", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")

with tab2:
    st.subheader("2) Ask deterministic questions")
    if st.session_state.df_clean is None:
        st.warning("Clean your dataset first (Data cleaning tab).")
    else:
        question = st.text_input("Ask a question about the dataset", placeholder='e.g., "Names of users in Artificial Intelligence department"')
        colA, colB = st.columns([1,1])
        with colA:
            run = st.button("Run query", type="primary")
        with colB:
            show_plan = st.checkbox("Show query plan (JSON)", value=False)

        if run:
            if not api_key:
                st.error("Please add your OpenAI API key in the sidebar.")
            elif not question.strip():
                st.error("Please type a question.")
            else:
                blocked, msg = basic_injection_check(question)
                if blocked:
                    st.error(msg)
                else:
                    try:
                        spec = plan_query_with_llm(question, st.session_state.df_clean, api_key=api_key, model=model)
                        if show_plan:
                            st.code(spec.model_dump_json(indent=2), language="json")

                        result = execute_query(spec, st.session_state.df_clean)

                        st.write("Result table")
                        st.dataframe(result, use_container_width=True)

                        answer = summarize_results_with_llm(question, result, api_key=api_key, model=model)
                        st.markdown("### Answer")
                        st.write(answer)

                    except Exception as e:
                        st.error(str(e))

        st.divider()
        st.markdown("### Why this is accurate")
        st.markdown("- LLM only creates a small JSON query plan.\n- Pandas executes it deterministically.\n- LLM only summarizes already computed results.")
from src.agent.graph import run_agent
import pandas as pd

with tab3:
    st.subheader("Self-Correcting Data Validation Agent")
    st.caption("Paste messy data → Extract JSON → Validate → Auto-correct retries → Final schema-perfect output (NO hallucination)")

    raw = st.text_area("Paste messy employee data (any format)", height=220)
    max_attempts = st.slider("Max retries", 1, 6, 3)

    # --- 1) VISUALIZE STATE MACHINE (diagram) ---
    st.markdown("### 🧭 State Machine (Extract → Validate → Correct → Finalize)")

    dot = """
    digraph G {
        rankdir=LR;
        node [shape=box, style="rounded,filled", color="#444444", fillcolor="#F4F6F8"];

        Extract [label="extract"];
        Validate [label="validate"];
        Correct [label="correct"];
        Finalize [label="finalize"];

        Extract -> Validate;
        Validate -> Finalize [label="pass OR max_retries"];
        Validate -> Correct [label="fail AND retries_left"];
        Correct -> Validate;
    }
    """
    try:
        st.graphviz_chart(dot, use_container_width=True)
    except Exception:
        st.code(
            "extract → validate → (pass) finalize\n"
            "            ↘ (fail) correct → validate (loop)\n",
            language="text"
        )

    if st.button("Run Agent", type="primary"):
        if not api_key:
            st.error("Please add your OpenAI API key in the sidebar.")
            st.stop()
        if not raw.strip():
            st.error("Paste some messy data first.")
            st.stop()

        with st.spinner("Running extract → validate → correct loop..."):
            final_state = run_agent(raw, api_key=api_key, model=model, max_attempts=max_attempts)

        log = final_state.get("log", [])
        result = final_state.get("result")

        # --- 3) CORRECTION COUNT SUMMARY (metrics) ---
        # attempts used
        attempts_used = max((x.get("attempt", 0) for x in log), default=0)

        # counts
        employees_n = len(result.get("employees", [])) if result else 0
        rejected_n = len(result.get("rejected", [])) if result else 0

        # how many correct steps happened
        correct_steps = sum(1 for x in log if x.get("step") == "correct")
        validate_fails = sum(1 for x in log if x.get("step") == "validate" and x.get("status") == "fail")

        st.markdown("### 📊 Run Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attempts used", attempts_used if attempts_used else 1)
        c2.metric("Corrections", correct_steps)
        c3.metric("Valid employees", employees_n)
        c4.metric("Rejected records", rejected_n)

        # Optional: show pass/fail clearly
        if result is None:
            st.error("Could not produce schema-valid JSON within retry limit.")
        else:
            st.success("Schema-valid output ✅")

        # --- 2) BEFORE / AFTER COMPARISON ---
        st.markdown("### 🔁 Before vs After")
        left, right = st.columns(2)

        with left:
            st.markdown("#### Before (Raw Input)")
            st.code(raw.strip(), language="text")

        with right:
            st.markdown("#### After (Schema Output)")
            if result is None:
                st.code(final_state.get("last_json_text", ""), language="json")
            else:
                st.code(json.dumps(result, indent=2, default=str), language="json")

        # --- Correction Log (keep your existing) ---
        st.markdown("### 🧾 Correction Log")
        st.json(log)

        # If failed, stop here
        if result is None:
            st.markdown("### Last JSON Attempt (debug)")
            st.code(final_state.get("last_json_text", ""), language="json")
            st.stop()

        # -------- Valid employees table --------
        st.markdown("### ✅ Valid Employees")
        employees = result.get("employees", [])
        if employees:
            df_emp = pd.DataFrame(employees)
            st.dataframe(df_emp, use_container_width=True)
        else:
            st.info("No valid employees extracted (all records were rejected).")

        # -------- Rejected records table --------
        st.markdown("### 🚫 Rejected Records (No hallucination)")
        rejected = result.get("rejected", [])
        if rejected:
            rej_rows = []
            for r in rejected:
                rej_rows.append(
                    {
                        "raw_record": r.get("raw_record", ""),
                        "reasons": "; ".join(r.get("reasons", [])),
                    }
                )
            df_rej = pd.DataFrame(rej_rows)
            st.dataframe(df_rej, use_container_width=True)
        else:
            st.info("No rejected records. Everything was schema-valid.")

        # -------- Download --------
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name="validated_output.json",
            mime="application/json",
        )
