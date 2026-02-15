"""Package evidence + recent runs + code snippets into a Sisyphus request.

Outputs:
- tmp/arch_loop/sisyphus_request.json  (structured)
- tmp/arch_loop/sisyphus_request.txt   (prompt-ready)

This does NOT call any LLM. It just prepares inputs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUMPROOT = REPO / "tmp" / "arch_loop"
EVIDENCE = DUMPROOT / "evidence.json"
STATE = DUMPROOT / "state.json"
RUNS = DUMPROOT / "runs.jsonl"


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tail_runs(n: int = 8) -> list[dict]:
    if not RUNS.exists():
        return []
    lines = [ln.strip() for ln in RUNS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def determine_mode(state: dict) -> str:
    # Step-based macro goal.
    step = int(state.get("accept_step", 0) or 0)
    if step <= 0:
        return "AS0"
    if step < 10:
        return "A"
    if step < 20:
        return "B"
    return "C"


def goal_constraints_for(mode: str, evidence: dict, state: dict) -> tuple[str, list[str]]:
    focus = evidence.get("focus", "")
    if mode == "AS0":
        lvl = int(state.get("as0_level", 0) or 0)
        cfg = dict(state.get("as0_cfg", {}) or {})
        rem_steps = list(cfg.get("rem_steps") or [])
        mdd_steps = list(cfg.get("mdd_steps") or [])
        maxr_steps = list(cfg.get("maxr_steps") or [])

        # pick next target thresholds (best-effort)
        rem_t = rem_steps[lvl] if lvl < len(rem_steps) else (rem_steps[-1] if rem_steps else 1.0)
        mdd_t = mdd_steps[min(lvl, len(mdd_steps) - 1)] if mdd_steps else 0.004
        maxr_t = maxr_steps[min(lvl, len(maxr_steps) - 1)] if maxr_steps else 0.06

        goal = (
            f"AS0-L{lvl+1} 달성: (anchor 대비) rem 개선 ≥ {rem_t:.2f}R "
            f"또는 H2 MDD 개선 ≥ {mdd_t:.3f} "
            f"또는 maxR 개선 ≥ {maxr_t:.2f} 중 1개 이상"
        )
        constraints = [
            "후보는 1~3 레버 조합만(딱 1~2개도 OK)",
            "후보는 최대 6개",
            "Q4 safety 유지(급격한 악화 금지)",
            "과최적화 위험 높은 복잡한 조건 추가 금지",
            "허용 knobs: rem_exit_on_riskoff, rem_time_stop_bars, tp1_ratio, be_move_mode, swing_stop_confirm_bars, tp1_r,tp2_r,adx_min",
            f"현재 focus: {focus}",
        ]
        return goal, constraints

    if mode == "A":
        goal = "A(S10) 달성: H2 return↑ + H2 MDD↓ (거래수(표본) 과도 감소 금지)"
    elif mode == "B":
        goal = "B(S20) 달성: Q4 return을 0 근처로 개선 + 안전성 강화(동시에 H2 훼손 금지)"
    else:
        goal = "C(S30) 달성: 장기 CAGR≥20% & MDD≤30% (다구간/다레짐 일관성)"

    constraints = [
        "후보는 1~3 레버 조합만",
        "후보는 최대 6개",
        "Q4 safety 유지(급격한 악화 금지)",
        "과최적화 위험 높은 복잡한 조건 추가 금지",
        f"현재 focus: {focus}",
    ]
    return goal, constraints


def code_snippets() -> dict:
    # Keep snippets short and stable: just pointers + a few key knobs.
    # (We avoid reading whole files here to keep payload small.)
    return {
        "knobs": [
            "rem_exit_on_riskoff",
            "rem_time_stop_bars",
            "tp1_ratio",
            "be_move_mode",
            "swing_stop_confirm_bars",
        ],
        "files": [
            "backtest_date_range_phaseA.py (remainder exit section)",
            "scripts/arch_autoloop.py (accept/AS0/policy gate)",
        ],
    }


def main():
    ev = load_json(EVIDENCE)
    st = load_json(STATE)
    mode = determine_mode(st)
    goal, constraints = goal_constraints_for(mode, ev, st)

    pkg = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "goal": goal,
        "constraints": constraints,
        "evidence": ev,
        "recent_runs": tail_runs(8),
        "snippets": code_snippets(),
        "output_format": {
            "candidates.json": {
                "schema": {"candidates": [{"key": "value"}]},
                "rules": ["<=6", "each candidate 1~3 keys", "no duplicates"],
            }
        },
    }

    (DUMPROOT / "sisyphus_request.json").write_text(
        json.dumps(pkg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Prompt-ready text
    txt = []
    txt.append("[Sisyphus 요청] candidates.json 생성")
    txt.append(f"- Mode: {mode}")
    txt.append(f"- Goal: {goal}")
    txt.append("- Constraints:")
    for c in constraints:
        txt.append(f"  - {c}")
    txt.append("\n출력은 JSON만:")
    txt.append('{"candidates": [ {"rem_exit_on_riskoff": true, "rem_time_stop_bars": 168}, {"tp1_ratio": 0.4} ]}')

    (DUMPROOT / "sisyphus_request.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print(f"wrote sisyphus_request.* mode={mode}")


if __name__ == "__main__":
    main()
