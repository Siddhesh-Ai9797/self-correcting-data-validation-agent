from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import ValidationError
from src.agent.schemas import ExtractedData

def run_offline_agent(raw_text: str, candidate_json_outputs: List[str], max_attempts: int = 3):
    """
    Offline testing: instead of calling OpenAI, we feed the agent JSON outputs
    and see if validation passes, and how many attempts it takes.
    """
    log = []
    last = ""
    attempt = 1

    for out in candidate_json_outputs[:max_attempts]:
        last = out
        log.append({"step": "extract", "attempt": attempt, "output": out})

        try:
            data = ExtractedData.model_validate_json(out)
            log.append({"step": "validate", "attempt": attempt, "status": "pass"})
            return {"result": data.model_dump(), "log": log, "last_json_text": last}
        except ValidationError as e:
            log.append({"step": "validate", "attempt": attempt, "status": "fail", "error": str(e)})
            attempt += 1

    return {"result": None, "log": log, "last_json_text": last}
