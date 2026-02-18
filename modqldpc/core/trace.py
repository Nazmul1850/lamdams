from __future__ import annotations
import json, os
from typing import Dict, Any
from datetime import datetime, timezone

class Trace:
    def __init__(self, trace_path: str):
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)

    def event(self, t: str, **fields: Any) -> None:
        rec: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "t": t,
            **fields,
        }
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
