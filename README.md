# 🤖 Self-Correcting Data Validation Agent

[![CI](https://github.com/Siddhesh-Ai9797/self-correcting-data-validation-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Siddhesh-Ai9797/self-correcting-data-validation-agent/actions/workflows/ci.yml)
### Schema-Enforced, No-Hallucination Agentic Data Pipeline

A production-style AI system that converts messy employee text into schema-valid JSON using:

- LangGraph (state machine orchestration)  
- OpenAI LLMs (structured extraction)  
- Pydantic v2 (strict schema validation)  
- Pandas (deterministic execution)  
- Streamlit (interactive UI)

---

## 🧠 Problem

LLMs hallucinate missing fields and fabricate identifiers.

This project enforces:

- Strict schema validation  
- Bounded self-correction retries  
- Deterministic query execution  
- Explicit rejection handling  
- Zero fabrication of required fields  

---

## 🏗 Architecture

Messy Text  
→ LLM Extraction  
→ Schema Validation  
→ Self-Correction Loop (if needed)  
→ Final Valid JSON  

Records missing required fields are rejected — never hallucinated.

---

## 🔁 Agent Flow

extract → validate
if fail → correct → validate → repeat
finalize

Retry attempts are limited and controlled.

---

## 💬 Deterministic Query Engine

1. LLM generates structured query plan  
2. Pandas executes it  
3. LLM summarizes computed results  

No synthetic answers.

---

## 🚀 Run Locally

```bash
git clone git@github.com:Siddhesh-Ai9797/self-correcting-data-validation-agent.git
cd self-correcting-data-validation-agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="your_key_here"
streamlit run app.py

---

# Stress Testing
python -m src.eval.run_agent_suite


