"""
Logging utilities.

Reviewers often check:
- is the run reproducible?
- does it save parameters + key outputs?
- is there enough diagnostics to debug?

This module provides lightweight structured logging to console + JSON lines.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


class JsonlLogger:
    def __init__(self, out_dir: str, filename: str = "run.log.jsonl"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, filename)

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None):
        record = {
            "ts": time.time(),
            "event": event,
            "payload": payload or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj
