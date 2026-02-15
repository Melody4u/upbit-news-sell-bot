"""Generate candidates.json from evidence.json (Sisyphus placeholder).

This is a deterministic generator to *prove the pipeline works*:
- evidence.json must exist
- writes tmp/arch_loop/candidates.json with up to 6 unique candidate dicts
- avoids duplicates in state.json.tried_labels when possible

Later: replace body with real Sisyphus/LLM output.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUMPROOT = REPO / "tmp" / "arch_loop"
EVIDENCE = DUMPROOT / "evidence.json"
CANDIDATES = DUMPROOT / "candidates.json"
STATE = DUMPROOT / "state.json"
PERSONA = DUMPROOT / "persona.json"


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def label(cand: dict) -> str:
    # mirror arch_autoloop apply_candidate label style as much as possible
    return ", ".join([f"{k}={cand[k]}" for k in sorted(cand.keys())])


def main() -> None:
    ev = load_json(EVIDENCE)
    st = load_json(STATE)
    tried = set(st.get("tried_labels", []) or [])

    focus = (ev.get("focus") or "").strip().lower()
    # baseline hints
    base_h2 = ((ev.get("baseline") or {}).get("H2") or {}).get("summ") or {}
    rem_loss = float((base_h2.get("loss_by_reason") or {}).get("rem", 0.0) or 0.0)
    maxr = float(base_h2.get("pos_r_max", 0.0) or 0.0)

    persona = load_json(PERSONA)
    wallst = str(persona.get("wallst", ""))
    crypto = str(persona.get("crypto", ""))

    # Core candidate pool (rem tail-risk + right tail)
    pool: list[dict] = []

    # rem tail-risk levers
    pool += [
        {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 120},
        {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 168},
        {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 240},
        {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 0},
    ]

    # right-tail / churn levers
    pool += [
        {"be_move_mode": "weak_only"},
        {"tp1_ratio": 0.4},
        {"tp1_ratio": 0.5},
    ]

    # Persona emphasis tweaks (simple deterministic heuristics)
    if "time-stop" in wallst.lower() or "time-stop" in crypto.lower():
        pool.insert(0, {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 168})
    if "weak_only" in crypto.lower():
        pool.insert(0, {"be_move_mode": "weak_only"})
    if "tp1" in crypto.lower():
        pool.insert(0, {"tp1_ratio": 0.4})

    # If rem loss is extreme, prioritize time-stop variations
    if "rem" in focus or rem_loss < -10.0:
        pass

    # If maxR is very small, prioritize leaving more remainder
    if maxr < 0.5:
        pool.insert(0, {"tp1_ratio": 0.4})

    # Deduplicate by label, avoid tried where possible
    out: list[dict] = []
    seen = set()
    for c in pool:
        lb = label(c)
        if lb in seen:
            continue
        seen.add(lb)
        if lb in tried:
            continue
        out.append(c)
        if len(out) >= 6:
            break

    # If everything is tried, at least output something deterministic
    if not out:
        out = [{"rem_exit_on_riskoff": True, "rem_time_stop_bars": 240}]

    CANDIDATES.write_text(json.dumps({"candidates": out}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {CANDIDATES} candidates={len(out)} focus={focus} rem={rem_loss:.2f} maxR={maxr:.2f}")


if __name__ == "__main__":
    main()
