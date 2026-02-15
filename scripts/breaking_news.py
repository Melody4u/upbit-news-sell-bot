"""Breaking news notifier for major version bumps.

Checks tmp/arch_loop/state.json for version changes. If major increased since last alert,
prints a short breaking-news message (intended for Discord announce via cron).

State file (local): tmp/arch_loop/breaking_state.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUMPROOT = REPO / "tmp" / "arch_loop"
STATE = DUMPROOT / "state.json"
BREAK_STATE = DUMPROOT / "breaking_state.json"


def load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse(v: str) -> tuple[int, int, int, int, int]:
    # v2_YY.M.DD.major.minor
    try:
        s = v.replace("v2_", "")
        yy, mm, dd, maj, minr = s.split(".")
        return int(yy), int(mm), int(dd), int(maj), int(minr)
    except Exception:
        return (0, 0, 0, 0, 0)


def main() -> int:
    st = load(STATE)
    v = str(st.get("version", ""))
    if not v:
        return 0

    _, _, _, maj, _ = parse(v)
    bs = load(BREAK_STATE)
    last_v = str(bs.get("last_version", ""))
    _, _, _, last_maj, _ = parse(last_v) if last_v else (0, 0, 0, 0, 0)

    if maj <= last_maj:
        return 0

    # Build a short breaking message
    last_accept = st.get("last_accept", {}) or {}
    label = last_accept.get("label", "")
    reason = last_accept.get("reason", "")

    msg = []
    msg.append(f"[BREAKING] 메이저 버전 상승: {last_v or '(none)'} → {v}")
    if label:
        msg.append(f"- Trigger: {label}")
    if reason:
        msg.append(f"- Why: {reason}")
    msg.append(f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    print("\n".join(msg))

    bs["last_version"] = v
    BREAK_STATE.write_text(json.dumps(bs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
