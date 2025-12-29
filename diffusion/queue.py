# diffusion/queue.py
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

ROOT = Path(os.environ.get("SYM_QUEUE_DIR", "queue"))
PENDING = ROOT / "pending"
WORKING = ROOT / "working"
DONE = ROOT / "done"
FAILED = ROOT / "failed"

for d in (PENDING, WORKING, DONE, FAILED):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    id: str
    prompt: str
    created_at: float
    params: dict[str, Any]


def _now() -> float:
    return time.time()


def submit(prompt: str, params: Optional[dict[str, Any]] = None) -> str:
    job_id = uuid4().hex
    data = {
        "id": job_id,
        "prompt": prompt,
        "created_at": _now(),
        "params": params or {},
    }
    tmp = PENDING / f".{job_id}.tmp"
    final = PENDING / f"{job_id}.json"
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(final)
    return job_id


def claim_next() -> Optional[Path]:
    # Claim the oldest pending job
    jobs = sorted(PENDING.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in jobs:
        dest = WORKING / p.name
        try:
            # Atomic claim on same filesystem
            p.replace(dest)
            return dest
        except FileNotFoundError:
            continue
        except PermissionError:
            continue
    return None


def complete(job_path: Path, result: dict[str, Any]) -> Path:
    out = DONE / job_path.name
    payload = {"status": "done", **result}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    job_path.unlink(missing_ok=True)
    return out


def fail(job_path: Path, error: str) -> Path:
    out = FAILED / job_path.name
    payload = {"status": "failed", "error": error}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    job_path.unlink(missing_ok=True)
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))