🤖 Self-Correcting Data Validation Agent
Schema-Enforced, No-Hallucination, Agentic Data Extraction Pipeline
A production-style Agentic AI system that converts unstructured employee data into schema-perfect JSON using:
LangGraph (state orchestration)
Pydantic v2 (strict schema validation)
OpenAI Structured Extraction
Deterministic Pandas Execution
Streamlit UI
Designed to demonstrate safe LLM integration with strict validation and zero hallucinated required fields.
🧠 Problem
Large Language Models are powerful at extracting structure from messy text —
but they:
Hallucinate missing required fields
Produce schema-invalid JSON
Fabricate IDs
Output inconsistent formats
This project solves that by enforcing:
✔ Strict schema validation
✔ Autonomous correction retries
✔ Deterministic execution
✔ Explicit rejection handling
✔ No fabricated required fields
🏗 System Architecture
Messy Text Input
→ LLM Extraction
→ Pydantic Schema Validation
→ (If Invalid) Self-Correction Loop
→ Final Schema-Valid Output
If required fields are missing (e.g., user_id), records are rejected — not hallucinated.
🔁 Agentic State Machine (LangGraph)
State:
raw_text
last_json_text
attempt
max_attempts
validation_error
result
log
Flow:
extract → validate
If fail → correct → validate → repeat
Else → finalize
The retry loop is controlled and bounded.
📦 Core Features
1️⃣ Structured Extraction
LLM converts messy input into strict JSON format:
{
  "employees": [
    {
      "user_id": int,
      "name": string,
      "age": int|null,
      "email": string|null,
      "salary": number|null,
      "join_date": YYYY-MM-DD|null,
      "department": enum,
      "performance_score": 0–10|null,
      "location": string|null,
      "job_title": string|null
    }
  ]
}
Normalization rules include:
Word numbers → integers
Salary cleaning
Department mapping
ISO date formatting
Out-of-range correction
2️⃣ Schema Enforcement (Pydantic v2)
All extracted data must pass:
Required fields enforced
Numeric bounds checked
Enum constraints enforced
Date type validation
No NaN allowed
No empty strings for missing values
If validation fails → correction step triggered.
3️⃣ Self-Correction Loop
When validation fails:
The model receives:
Previous JSON
Validation error message
It must correct the structure
No records may be silently dropped
Retry limit enforced
Zero hallucinated user_id.
4️⃣ No-Hallucination Guarantee
If required fields are missing:
{
  "employees": [],
  "rejected": [
    {
      "raw_record": "...",
      "reasons": ["missing user_id"]
    }
  ]
}
The system never fabricates identifiers.
5️⃣ Deterministic Query Engine
For dataset questions:
LLM generates structured query plan
Pandas executes filtering/aggregation
LLM only summarizes computed results
This prevents synthetic answer fabrication.
🧪 Stress Testing
Includes automated agent suite:
python -m src.eval.run_agent_suite
Results:
15/15 cases passed
100% schema-valid outputs
Correct rejection behavior
Avg attempts: 1.0
No hallucinated required fields
Edge cases tested:
Word numbers
Missing fields
Conflicting values
Garbage text
Extreme noise
Multiple records
Unstructured paragraphs
🖥 Interface (Streamlit)
Run:
streamlit run app.py
UI includes:
📄 Data Cleaning Tab
💬 Deterministic Query Tab
🧠 Self-Correcting Agent Tab
Agent tab provides:
Correction Log
Retry Count
Before / After view
Valid vs Rejected records
JSON download
📂 Project Structure
industry_ready_data_agent/

app.py
requirements.txt
README.md

src/
    agent/
        graph.py
        schemas.py
    core/
        cleaning.py
        query.py
    eval/
        run_agent_suite.py

test_inputs/
employee_dataset_50rows.csv
🚀 Installation
Clone repository:
git clone git@github.com:Siddhesh-Ai9797/self-correcting-data-validation-agent.git
cd self-correcting-data-validation-agent
Create environment:
python -m venv .venv
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Set API key:
export OPENAI_API_KEY="your_key_here"
Run:
streamlit run app.py
🔐 Design Principles
Fail closed, not open
Deterministic where possible
Explicit rejection > silent correction
Structured retries only
Bounded correction loop
JSON-safe serialization
📈 Production Readiness
Current system demonstrates:
✔ Agent orchestration
✔ Schema enforcement
✔ Deterministic execution layer
✔ Retry loop logic
✔ Controlled LLM behavior
Future extensions:
FastAPI backend
Persistent storage
Docker containerization
Structured logging
Role-based validation
Deployment to AWS
🎯 Why This Project Matters
This is not just a demo app.
It demonstrates how to build:
Safe LLM pipelines
Controlled agent systems
Validation-first AI architecture
Production-oriented AI engineering
Most LLM apps hallucinate.
This one enforces correctness.
👨‍💻 Author
Siddhesh Patil
M.S. Artificial Intelligence
DePaul University
