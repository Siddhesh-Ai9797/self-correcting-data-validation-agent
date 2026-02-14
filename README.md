# AI Data Validation Agent (Industry-ready)

A Streamlit app that:
- Cleans & normalizes messy HR-like tabular data (emails, dates, ages, salaries, departments)
- Produces an **audit report** of what was fixed
- Answers natural-language questions **deterministically** using pandas (not "LLM guesses")
- Uses an LLM only to convert the user's question into a small JSON query spec + to explain results

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Environment variables

Set your OpenAI key in **Streamlit sidebar** or via env var:

```bash
export OPENAI_API_KEY="..."
```


## Evaluation (accuracy benchmarks)

Run deterministic benchmarks (no LLM required):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.eval.run_eval --csv your_data.csv
```

Optional end-to-end LLM evaluation:
- Set `OPENAI_API_KEY`, and change a benchmark case `mode` to `"llm"` in `src/eval/benchmarks.json`.
- Then re-run the command above.
