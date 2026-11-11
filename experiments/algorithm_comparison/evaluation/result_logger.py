"""Append-only JSONL result logging for isolated experiments."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_result(path: str | Path, result: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")

