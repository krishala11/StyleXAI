"""Checkpoint logging utilities — every pipeline stage writes traceable JSON."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _serialize(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return str(obj)


def save_checkpoint(name: str, data: dict, log_dir: Path | None = None) -> Path:
    """Save a named checkpoint JSON with timestamp metadata."""
    log_dir = log_dir or config.CHECKPOINTS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": _serialize(data),
    }
    path = log_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def append_trace(event: str, details: dict, trace_log: list) -> None:
    trace_log.append({
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": _serialize(details),
    })
