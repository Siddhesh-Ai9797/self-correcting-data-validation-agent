from __future__ import annotations

import re
from typing import Tuple

INJECTION_PATTERNS = [
    r"ignore (all|any) previous",
    r"system prompt",
    r"reveal.*(key|secret|token)",
    r"exfiltrat",
    r"prompt injection",
]

def basic_injection_check(user_text: str) -> Tuple[bool, str]:
    t = (user_text or "").lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, t):
            return True, "That request looks like a prompt-injection attempt. I can only answer questions about the uploaded dataset."
    return False, ""
