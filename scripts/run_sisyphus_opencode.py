"""Call Sisyphus via OpenCode to generate candidates.json.

This script:
- Reads tmp/arch_loop/sisyphus_request.txt (prompt)
- Runs: scripts/opencode_run_safe.py -m <model> --file <prompt>
- Extracts JSON from output and writes tmp/arch_loop/candidates.json

Notes:
- Requires OpenCode installed (opencode.cmd available).
- Model is provided via --model or env OPENCODE_SISYPHUS_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUMPROOT = REPO / "tmp" / "arch_loop"
PROMPT_TXT = DUMPROOT / "sisyphus_request.txt"
CANDIDATES = DUMPROOT / "candidates.json"
RUN_SAFE = REPO / "scripts" / "opencode_run_safe.py"
PY = REPO / ".venv" / "Scripts" / "python.exe"


def extract_json(text: str) -> dict:
    # Try to find the last JSON object in the output.
    # We expect something like: {"candidates": [...]}
    matches = list(re.finditer(r"\{[\s\S]*\}", text))
    for m in reversed(matches):
        chunk = m.group(0).strip()
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except Exception:
            continue
    raise ValueError("No valid candidates JSON found in output")


ALLOWED = {
    "rem_exit_on_riskoff",
    "rem_time_stop_bars",
    "tp1_ratio",
    "be_move_mode",
    "swing_stop_confirm_bars",
    "tp2_r",
    "tp1_r",
    "adx_min",
}


def validate(obj: dict) -> dict:
    cands = obj.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise ValueError("candidates must be a non-empty list")
    if len(cands) > 6:
        cands = cands[:6]
    out = []
    seen = set()
    for c in cands:
        if not isinstance(c, dict):
            continue
        # filter unknown keys
        c2 = {k: c[k] for k in c.keys() if k in ALLOWED}
        if not c2:
            continue
        if not (1 <= len(c2.keys()) <= 3):
            continue
        key = json.dumps(c2, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(c2)
    if not out:
        raise ValueError("no valid candidates after validation")
    return {"candidates": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENCODE_SISYPHUS_MODEL", ""), help="OpenCode model id")
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("OPENCODE_TIMEOUT", "900")))
    args = ap.parse_args()

    if not args.model:
        raise SystemExit("Missing --model (or OPENCODE_SISYPHUS_MODEL)")
    if not PROMPT_TXT.exists():
        raise SystemExit(f"Missing prompt file: {PROMPT_TXT}")

    cmd = [
        str(PY),
        str(RUN_SAFE),
        "-m",
        str(args.model),
        "--file",
        str(PROMPT_TXT),
        "--timeout",
        str(args.timeout),
    ]

    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise SystemExit(out[-4000:])

    obj = extract_json(out)
    obj = validate(obj)
    CANDIDATES.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {CANDIDATES} candidates={len(obj['candidates'])}")


if __name__ == "__main__":
    main()
