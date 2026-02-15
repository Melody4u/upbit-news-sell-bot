"""Persona advice generator (deterministic stub).

Writes tmp/arch_loop/persona.json with two persona summaries based on evidence.
This is intentionally deterministic (no LLM) to keep the pipeline reliable.
Later: can be replaced by real persona LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUMPROOT = REPO / "tmp" / "arch_loop"
EVIDENCE = DUMPROOT / "evidence.json"
OUT = DUMPROOT / "persona.json"


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ev = load_json(EVIDENCE)
    h2s = ((ev.get("baseline") or {}).get("H2") or {}).get("summ") or {}
    loss = (h2s.get("loss_by_reason") or {})
    rem = float(loss.get("rem", 0.0) or 0.0)
    maxr = float(h2s.get("pos_r_max", 0.0) or 0.0)

    wallst = (
        "[WallSt] rem(잔여) 손실이 크면 잔여를 옵션처럼 관리: "
        "risk-off에서 잔여 축소/청산 + time-stop 우선. "
        f"(rem={rem:.1f}R)"
    )
    crypto = (
        "[Crypto] 승자꼬리(maxR)가 짧으면 TP1/BE/트레일이 수익을 절단: "
        "TP1 비중↓, BE를 약세 구간에만 적용(weak_only) 등으로 maxR 확보. "
        f"(maxR={maxr:.2f})"
    )

    OUT.write_text(
        json.dumps({"wallst": wallst, "crypto": crypto}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
