"""Workflow watchdog (research-only).

Purpose: if the 3-hour patchnote workflow silently stops, recover automatically.

Heuristic:
- If tmp/arch_loop/runs.jsonl is older than STALE_MINUTES, consider it stalled.
- Then run scripts/patchnote_3h.py once to re-kick the workflow.

This script is safe to run frequently.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
PATCHNOTE = REPO / "scripts" / "patchnote_3h.py"
DUMPROOT = REPO / "tmp" / "arch_loop"
RUNS = DUMPROOT / "runs.jsonl"
LOG = DUMPROOT / "watchdog.log"

STALE_MINUTES = int(os.environ.get("WATCHDOG_STALE_MINUTES", "240"))  # 4h default


def log(msg: str) -> None:
    DUMPROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG.write_text((LOG.read_text(encoding="utf-8") if LOG.exists() else "") + f"[{ts}] {msg}\n", encoding="utf-8")


def main() -> int:
    now = datetime.now()
    if not RUNS.exists():
        log("runs.jsonl missing -> kick patchnote")
        return kick()

    mtime = datetime.fromtimestamp(RUNS.stat().st_mtime)
    age = now - mtime
    if age <= timedelta(minutes=STALE_MINUTES):
        log(f"ok age={age}")
        return 0

    log(f"stale age={age} -> kick patchnote")
    return kick()


def kick() -> int:
    env = dict(**{**dict(os.environ), "PYTHONIOENCODING": "utf-8"})
    try:
        subprocess.run([str(PY), str(PATCHNOTE)], cwd=str(REPO), check=True, env=env)
        log("kick success")
        return 0
    except Exception as e:
        log(f"kick failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
