"""Architecture auto-loop (V2 research-only).

Includes SAFE_PATCH (preset-only) auto-improvement + Sisyphus-catalog hook.

Modes:
- Baseline ladder (H2/Q4) + dump attribution summary
- Write an evidence JSON (`tmp/arch_loop/evidence.json`) that an LLM("Sisyphus") can use
- If a candidates file is provided, test those candidates (max N per cycle)
- Otherwise, fall back to a small built-in heuristic candidate list

No live trading logic is touched.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
BACKTEST_V2 = REPO / "backtest_v2.py"
PRESET = REPO / "v2" / "preset.json"
DUMPROOT = REPO / "tmp" / "arch_loop"
STATE = DUMPROOT / "state.json"
EVIDENCE = DUMPROOT / "evidence.json"
CANDIDATES = DUMPROOT / "candidates.json"


@dataclass
class Slice:
    name: str
    start: str
    end: str


SLICES = [
    Slice("H2", "2025-07-01", "2026-01-01"),
    Slice("Q4", "2025-10-01", "2026-01-01"),
]


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{out[-4000:]}")
    return out


def git(cmd: list[str]) -> str:
    return run(["git", *cmd])


def parse_result_dict(stdout: str) -> dict:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return ast.literal_eval(ln)
            except Exception:
                continue
    raise RuntimeError("Could not parse result dict from output")


def backtest(slice_: Slice, dump_name: str) -> tuple[dict, Path]:
    os.makedirs(DUMPROOT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_base = DUMPROOT / f"{dump_name}_{slice_.name}_{ts}.csv"
    cmd = [
        str(PY),
        str(BACKTEST_V2),
        "--preset",
        str(PRESET),
        "--market",
        "KRW-BTC",
        "--start",
        slice_.start,
        "--end",
        slice_.end,
        "--cache-dir",
        "ohlcv_cache",
        "--dump-trades-csv",
        str(dump_base),
    ]
    out = run(cmd)
    res = parse_result_dict(out)
    return res, dump_base


def summarize_dumps(dump_base: Path) -> dict:
    legs_path = Path(str(dump_base).replace(".csv", "_legs.csv"))
    pos_path = Path(str(dump_base).replace(".csv", "_pos.csv"))
    legs = pd.read_csv(legs_path)
    pos = pd.read_csv(pos_path)

    loss_by_reason = (
        legs[legs["r"] < 0]
        .groupby("exit_reason")["r"]
        .sum()
        .sort_values()
        .head(8)
        .to_dict()
    )
    count_by_reason = legs["exit_reason"].value_counts().head(8).to_dict()

    return {
        "legs": int(len(legs)),
        "positions": int(len(pos)),
        "pos_r_mean": float(pos["r"].mean()) if len(pos) else 0.0,
        "pos_r_min": float(pos["r"].min()) if len(pos) else 0.0,
        "pos_r_max": float(pos["r"].max()) if len(pos) else 0.0,
        "loss_by_reason": loss_by_reason,
        "count_by_reason": count_by_reason,
    }


def persona_wallst(summary: dict) -> str:
    rem_loss = float(summary.get("loss_by_reason", {}).get("rem", 0.0))
    return f"WallSt: rem tail-risk 통제(리스크오프 잔여 축소/청산 + time-stop). rem={rem_loss:.1f}R"


def persona_crypto(summary: dict) -> str:
    pr_max = float(summary.get("pos_r_max", 0.0))
    return f"Crypto: 승자 꼬리(maxR) 확장 위해 TP1/BE/트레일 재설계(+time-stop 병행). maxR={pr_max:.2f}"


def load_preset() -> dict:
    return json.loads(PRESET.read_text(encoding="utf-8"))


def save_preset(p: dict) -> None:
    PRESET.write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_run(h2: dict, q4: dict) -> float:
    # Higher is better.
    # We mainly optimize H2 return and penalize H2 MDD; lightly penalize negative Q4 return.
    h2_r = float(h2.get("return_pct", 0.0) or 0.0)
    h2_mdd = float(h2.get("mdd_pct", 0.0) or 0.0)
    q4_r = float(q4.get("return_pct", 0.0) or 0.0)

    score = h2_r - (h2_mdd * 60.0)
    if q4_r < 0:
        score += q4_r * 0.5  # penalty
    return float(score)


def accept_patch(base_h2: dict, base_q4: dict, new_h2: dict, new_q4: dict) -> bool:
    # Accept if score improves and Q4 doesn't catastrophically regress.
    base_score = score_run(base_h2, base_q4)
    new_score = score_run(new_h2, new_q4)

    q4_r_drop = float(base_q4.get("return_pct", 0.0) or 0.0) - float(new_q4.get("return_pct", 0.0) or 0.0)
    return bool((new_score > base_score + 1.0) and q4_r_drop <= 3.0)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_candidate(preset: dict, cand: dict) -> tuple[dict, str]:
    p = json.loads(json.dumps(preset))
    sim = dict(p.get("simulate_kwargs", {}) or {})
    for k, v in cand.items():
        sim[k] = v
    p["simulate_kwargs"] = sim
    label = ", ".join([f"{k}={v}" for k, v in cand.items()])
    return p, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default=str(CANDIDATES))
    ap.add_argument("--max-candidates", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(DUMPROOT, exist_ok=True)

    state = load_state()
    tried = set(state.get("tried_labels", []) or [])

    # 1) baseline ladder
    base = {}
    for s in SLICES:
        res, dump = backtest(s, dump_name="base")
        summ = summarize_dumps(dump)
        base[s.name] = {"res": res, "summ": summ}

    base_h2 = base["H2"]["res"]
    base_q4 = base["Q4"]["res"]
    base_h2_s = base["H2"]["summ"]

    focus = "rem 손실 집중" if "rem" in base_h2_s.get("loss_by_reason", {}) else "unknown"

    ladder_a = "PASS" if (float(base_h2.get("return_pct", 0.0) or 0.0) >= 10.0 and float(base_h2.get("mdd_pct", 1.0) or 1.0) <= 0.30) else "FAIL"
    ladder_b = "PASS" if float(base_q4.get("return_pct", 0.0) or 0.0) >= 0.0 else "FAIL"

    action = "관측만"
    runs_backtest = 2
    runs_patch = 0
    runs_rollback = 0

    applied = False
    preset0 = load_preset()

    rem_loss = float(base_h2_s.get("loss_by_reason", {}).get("rem", 0.0))

    # 1.5) write evidence for Sisyphus
    evidence = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "focus": focus,
        "ladder": {"A_H2": ladder_a, "B_Q4": ladder_b, "C_Long": "SKIP"},
        "baseline": {
            "H2": {"res": base_h2, "summ": base_h2_s},
            "Q4": {"res": base_q4, "summ": base["Q4"]["summ"]},
        },
        "notes": {
            "rem_loss": rem_loss,
            "objective": "H2 개선 우선(수익↑/MDD↓), Q4 급격한 악화는 금지",
            "knobs": [
                "rem_exit_on_riskoff(bool)",
                "rem_time_stop_bars(int)",
                "tp1_ratio(float)",
                "be_move_mode(str)",
                "swing_stop_confirm_bars(int)",
            ],
        },
    }
    EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 2) Candidate selection (SAFE_PATCH)
    candidates: list[dict] = []

    # Try load candidates.json produced by Sisyphus (LLM). Expected: {"candidates": [ {..}, ... ]}
    cand_path = Path(args.candidates)
    if cand_path.exists():
        try:
            obj = json.loads(cand_path.read_text(encoding="utf-8"))
            candidates = list(obj.get("candidates", []) or [])
        except Exception:
            candidates = []

    # fallback heuristic catalog
    if not candidates and ladder_a == "FAIL" and rem_loss < -10.0:
        candidates = [
            {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 72},
            {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 120},
            {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 168},
            {"rem_exit_on_riskoff": False, "rem_time_stop_bars": 120},
            {"rem_exit_on_riskoff": True, "rem_time_stop_bars": 0},
            {"tp1_ratio": 0.4},
        ]

    best = None
    best_label = ""
    best_score = score_run(base_h2, base_q4)

    max_cands = int(max(0, args.max_candidates))
    if max_cands <= 0:
        candidates = []

    for cand in candidates:
        p2, label = apply_candidate(preset0, cand)
        if label in tried:
            continue

        save_preset(p2)
        runs_patch += 1
        action = f"SAFE_PATCH 후보 테스트: {label}"

        new = {}
        for s in SLICES:
            res, dump = backtest(s, dump_name="patch")
            summ = summarize_dumps(dump)
            new[s.name] = {"res": res, "summ": summ}
        runs_backtest += 2

        new_score = score_run(new["H2"]["res"], new["Q4"]["res"])
        tried.add(label)

        if accept_patch(base_h2, base_q4, new["H2"]["res"], new["Q4"]["res"]):
            if new_score > best_score:
                best = new
                best_label = label
                best_score = new_score

        save_preset(preset0)
        runs_rollback += 1

        if runs_patch >= int(max(0, args.max_candidates)):
            break

    if best is not None:
        applied = True
        for cand in candidates:
            _p2, _label = apply_candidate(preset0, cand)
            if _label == best_label:
                save_preset(_p2)
                break

        base = best
        base_h2 = base["H2"]["res"]
        base_q4 = base["Q4"]["res"]
        base_h2_s = base["H2"]["summ"]
        ladder_a = "PASS" if (float(base_h2.get("return_pct", 0.0) or 0.0) >= 10.0 and float(base_h2.get("mdd_pct", 1.0) or 1.0) <= 0.30) else "FAIL"
        ladder_b = "PASS" if float(base_q4.get("return_pct", 0.0) or 0.0) >= 0.0 else "FAIL"
        action = f"ACCEPT: {best_label}"

    # 3) persist state + commit accepted preset
    state["tried_labels"] = sorted(list(tried))[-400:]
    state["last_focus"] = focus
    state["last_evidence"] = str(EVIDENCE)
    save_state(state)

    if applied:
        msg = (
            f"auto: {best_label} | "
            f"H2 return {base['H2']['res'].get('return_pct')} / mdd {base['H2']['res'].get('mdd_pct')} | "
            f"Q4 return {base['Q4']['res'].get('return_pct')} / mdd {base['Q4']['res'].get('mdd_pct')}"
        )
        git(["add", str(PRESET), str(STATE)])
        git(["commit", "-m", msg])
        git(["push"])

    # 4) story report
    h2r = base["H2"]["res"]
    q4r = base["Q4"]["res"]
    h2s = base["H2"]["summ"]

    print(f"[arch-loop] {datetime.now().strftime('%Y-%m-%d %H:00')}")
    print(f"- Focus(원인): {focus}")
    print(f"- Ladder: A(H2)={ladder_a}, B(Q4)={ladder_b}, C(Long)=SKIP")
    print(f"- Action(이번): {action}")
    print(f"- Result(전/후): 관측 기반" if (not applied) else "- Result(전/후): 개선 적용")
    print(f"  · H2: return {h2r.get('return_pct')} / mdd {h2r.get('mdd_pct')} / entries {h2r.get('entries')}")
    print(f"  · Q4: return {q4r.get('return_pct')} / mdd {q4r.get('mdd_pct')} / entries {q4r.get('entries')}")

    loss = h2s.get("loss_by_reason", {})
    top2 = list(loss.items())[:2]
    top2s = ", ".join([f"{k} {v:.2f}R" for k, v in top2]) if top2 else "n/a"
    print(
        f"- Evidence: H2 loss TOP2 {top2s} / posR mean {h2s.get('pos_r_mean'):.3f} min {h2s.get('pos_r_min'):.3f} max {h2s.get('pos_r_max'):.3f}"
    )
    print(f"- Runs: backtest {runs_backtest}회 / patch {runs_patch}회 / rollback {runs_rollback}회")
    print(f"- Next: Sisyphus가 {EVIDENCE.name} 기반으로 {cand_path.name} 갱신 → 다음 사이클 테스트")

    print(persona_wallst(h2s))
    print(persona_crypto(h2s))


if __name__ == "__main__":
    main()
