import json
from src.agent.offline_runner import run_offline_agent

def test_offline_runner_fails_then_passes():
    bad = json.dumps({
        "employees": [{
            "user_id": "101",   # wrong type
            "name": "",         # invalid (min length)
            "department": "ai"  # invalid enum
        }]
    })

    good = json.dumps({
        "employees": [{
            "user_id": 101,
            "name": "Michael Chen",
            "age": 29,
            "email": None,
            "salary": 120000,
            "join_date": None,
            "department": "Artificial Intelligence",
            "performance_score": None,
            "location": None,
            "job_title": None
        }]
    })

    out = run_offline_agent("raw", [bad, good], max_attempts=2)
    assert out["result"] is not None
    assert len(out["log"]) >= 3
