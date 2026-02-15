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
RUNS_LOG = DUMPROOT / "runs.jsonl"
POLICY_EVAL = DUMPROOT / "policy_eval.json"


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


def step_ladder_defs() -> list[dict]:
    # Small steps first: we want ACCEPTs to happen so improvements can accumulate.
    # Later steps converge to original ladder goals.
    return [
        {
            "name": "S0-safety",
            "desc": "Q4 급격한 악화 금지(안전) — 변화 누적 가능",
        },
        {
            "name": "S1-rem_any",
            "desc": "H2 rem 손실이 조금이라도 개선(>= +1R) 또는 H2 MDD -0.005 개선",
        },
        {
            "name": "S2-rem_real",
            "desc": "H2 rem 손실 의미있게 개선(>= +5R) 또는 H2 MDD -0.02 개선",
        },
        {
            "name": "S3-score",
            "desc": "H2(return↑/MDD↓) 종합점수 개선(>+1) + Q4 방어",
        },
        {
            "name": "S4-ladder",
            "desc": "A(H2) PASS & B(Q4) PASS (원래 라더 목표)",
        },
    ]


def _q4_safe(base_q4: dict, new_q4: dict) -> bool:
    # Safety guardrail: don't let Q4 deteriorate too much.
    base_r = float(base_q4.get("return_pct", 0.0) or 0.0)
    new_r = float(new_q4.get("return_pct", 0.0) or 0.0)
    base_mdd = float(base_q4.get("mdd_pct", 0.0) or 0.0)
    new_mdd = float(new_q4.get("mdd_pct", 0.0) or 0.0)
    return bool((base_r - new_r) <= 3.0 and (new_mdd - base_mdd) <= 0.02)


def achieved_step(base: dict, new: dict, current_step: int) -> tuple[int, str]:
    """Return (best_step, reason)."""
    base_h2, base_q4 = base["H2"]["res"], base["Q4"]["res"]
    new_h2, new_q4 = new["H2"]["res"], new["Q4"]["res"]
    base_h2s, new_h2s = base["H2"]["summ"], new["H2"]["summ"]

    if not _q4_safe(base_q4, new_q4):
        return current_step, "Q4 safety fail"

    a_pass = bool(float(new_h2.get("return_pct", 0.0) or 0.0) >= 10.0 and float(new_h2.get("mdd_pct", 1.0) or 1.0) <= 0.30)
    b_pass = bool(float(new_q4.get("return_pct", 0.0) or 0.0) >= 0.0)

    base_rem = float(base_h2s.get("loss_by_reason", {}).get("rem", 0.0) or 0.0)
    new_rem = float(new_h2s.get("loss_by_reason", {}).get("rem", 0.0) or 0.0)
    rem_improve = new_rem - base_rem

    base_mdd = float(base_h2.get("mdd_pct", 0.0) or 0.0)
    new_mdd = float(new_h2.get("mdd_pct", 0.0) or 0.0)
    mdd_improve = base_mdd - new_mdd

    base_score = score_run(base_h2, base_q4)
    new_score = score_run(new_h2, new_q4)

    best = current_step
    reason = "S0 safety"

    if rem_improve >= 1.0 or mdd_improve >= 0.005:
        best = max(best, 1)
        reason = f"S1 rem_improve={rem_improve:.2f}R mdd_improve={mdd_improve:.3f}"

    if rem_improve >= 5.0 or mdd_improve >= 0.02:
        best = max(best, 2)
        reason = f"S2 rem_improve={rem_improve:.2f}R mdd_improve={mdd_improve:.3f}"

    if new_score > base_score + 1.0:
        best = max(best, 3)
        reason = f"S3 score {base_score:.2f}->{new_score:.2f}"

    if a_pass and b_pass:
        best = max(best, 4)
        reason = "S4 ladder pass"

    return best, reason


def as0_level_hit(anchor: dict, cur: dict, cfg: dict | None = None) -> tuple[int, str]:
    """Return (level 0..10, reason) based on H2-only improvements vs anchor.

    cfg may contain threshold arrays:
      - rem_steps, mdd_steps, maxr_steps
    """
    cfg = cfg or {}

    a_rem = float(anchor.get("rem_loss", 0.0) or 0.0)
    a_mdd = float(anchor.get("mdd", 0.0) or 0.0)
    a_maxr = float(anchor.get("maxR", 0.0) or 0.0)
    c_rem = float(cur.get("rem_loss", 0.0) or 0.0)
    c_mdd = float(cur.get("mdd", 0.0) or 0.0)
    c_maxr = float(cur.get("maxR", 0.0) or 0.0)

    rem_imp = c_rem - a_rem
    mdd_imp = a_mdd - c_mdd
    maxr_imp = c_maxr - a_maxr

    # default steps (very small); can be subdivided dynamically when stuck.
    rem_steps = list(cfg.get("rem_steps") or [0.3, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0])
    mdd_steps = list(cfg.get("mdd_steps") or [0.002, 0.004, 0.006, 0.01, 0.015, 0.02])
    maxr_steps = list(cfg.get("maxr_steps") or [0.03, 0.06, 0.10])

    lvl = 0
    for t in rem_steps:
        if rem_imp >= float(t):
            lvl += 1
    for t in mdd_steps:
        if mdd_imp >= float(t):
            lvl = max(lvl, 1 + mdd_steps.index(t))
    for t in maxr_steps:
        if maxr_imp >= float(t):
            lvl = max(lvl, 1 + maxr_steps.index(t))

    lvl = int(min(10, max(0, lvl)))
    reason = f"AS0 rem_imp={rem_imp:.2f}R mdd_imp={mdd_imp:.3f} maxR_imp={maxr_imp:.2f}"
    return lvl, reason


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


ALLOWED_KNOBS = {
    "rem_exit_on_riskoff",
    "rem_time_stop_bars",
    "tp1_ratio",
    "be_move_mode",
    "swing_stop_confirm_bars",
    # keep a few additional safe knobs that exist in simulate signature
    "tp2_r",
    "tp1_r",
    "adx_min",
}


def apply_candidate(preset: dict, cand: dict) -> tuple[dict, str]:
    # Filter unknown keys to avoid crashing simulate() with unexpected kwargs.
    safe = {k: v for k, v in (cand or {}).items() if k in ALLOWED_KNOBS}
    p = json.loads(json.dumps(preset))
    sim = dict(p.get("simulate_kwargs", {}) or {})
    for k, v in safe.items():
        sim[k] = v
    p["simulate_kwargs"] = sim
    label = ", ".join([f"{k}={safe[k]}" for k in sorted(safe.keys())]) if safe else "(filtered_empty)"
    return p, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default=str(CANDIDATES))
    ap.add_argument("--max-candidates", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(DUMPROOT, exist_ok=True)

    state = load_state()
    tried = set(state.get("tried_labels", []) or [])
    current_step = int(state.get("accept_step", 0) or 0)
    as0_level = int(state.get("as0_level", 0) or 0)
    as0_cfg = dict(state.get("as0_cfg", {}) or {})
    as0_stuck = int(state.get("as0_stuck", 0) or 0)
    as0_stuck_cycles = int(state.get("as0_stuck_cycles", 5) or 5)

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
    # set AS0 anchor once (H2-only)
    if "as0_anchor" not in state:
        state["as0_anchor"] = {
            "rem_loss": float(base_h2_s.get("loss_by_reason", {}).get("rem", 0.0) or 0.0),
            "mdd": float(base_h2.get("mdd_pct", 0.0) or 0.0),
            "maxR": float(base_h2_s.get("pos_r_max", 0.0) or 0.0),
        }
    if "as0_cfg" not in state:
        state["as0_cfg"] = {
            "rem_steps": [0.3, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0],
            "mdd_steps": [0.002, 0.004, 0.006, 0.01, 0.015, 0.02],
            "maxr_steps": [0.03, 0.06, 0.10]
        }

    evidence = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "focus": focus,
        "ladder": {"A_H2": ladder_a, "B_Q4": ladder_b, "C_Long": "SKIP"},
        "baseline": {
            "H2": {"res": base_h2, "summ": base_h2_s},
            "Q4": {"res": base_q4, "summ": base["Q4"]["summ"]},
        },
        "as0": {
            "anchor": state.get("as0_anchor", {}),
            "current_level": int(as0_level),
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

    # Data-quality score (extended): include rem loss + maxR in addition to score_run
    base_rem0 = float(base_h2_s.get("loss_by_reason", {}).get("rem", 0.0) or 0.0)
    base_maxr0 = float(base_h2_s.get("pos_r_max", 0.0) or 0.0)

    def score_ext(h2_res: dict, q4_res: dict, h2_summ: dict) -> float:
        s = float(score_run(h2_res, q4_res))
        rem = float(h2_summ.get("loss_by_reason", {}).get("rem", 0.0) or 0.0)
        maxr = float(h2_summ.get("pos_r_max", 0.0) or 0.0)
        # rem less negative is better; maxR higher is better
        s += 0.15 * (rem - base_rem0)
        s += 1.5 * (maxr - base_maxr0)
        return float(s)

    best = None
    best_label = ""
    best_reason = ""
    best_step = current_step
    best_score = score_ext(base_h2, base_q4, base_h2_s)

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

        new_score = score_ext(new["H2"]["res"], new["Q4"]["res"], new["H2"]["summ"])  # extended score
        tried.add(label)

        step_hit, reason = achieved_step(base, new, current_step)

        # AS0 micro-progress (H2-only). Applies when we're still at S0.
        anchor = state.get("as0_anchor", {}) or {}
        cur_h2 = {
            "rem_loss": float(new["H2"]["summ"].get("loss_by_reason", {}).get("rem", 0.0) or 0.0),
            "mdd": float(new["H2"]["res"].get("mdd_pct", 0.0) or 0.0),
            "maxR": float(new["H2"]["summ"].get("pos_r_max", 0.0) or 0.0),
        }
        lvl_hit, lvl_reason = as0_level_hit(anchor, cur_h2, as0_cfg) if int(current_step) == 0 else (0, "")

        # Auto-accept policy:
        # - Prefer step progress
        # - Or AS0 micro-level progress (while in S0)
        # - Or micro-accept by score improvement
        micro_accept = bool(step_hit == current_step and new_score > best_score + 0.5)
        as0_progress = bool(int(current_step) == 0 and int(lvl_hit) > int(as0_level))

        if step_hit > current_step or as0_progress or micro_accept:
            # Prefer bigger step jumps; then AS0 level; then score
            better = False
            if step_hit > best_step:
                better = True
            elif step_hit == best_step:
                if int(current_step) == 0 and int(lvl_hit) > int(state.get("_best_as0_level", as0_level)):
                    better = True
                elif new_score > best_score:
                    better = True

            if better:
                best = new
                best_label = label
                if step_hit > current_step:
                    best_reason = reason
                elif as0_progress:
                    best_reason = f"AS0 {as0_level}->{lvl_hit} {lvl_reason}"
                else:
                    best_reason = f"micro_accept score {best_score:.2f}->{new_score:.2f}"
                best_step = step_hit
                best_score = new_score
                state["_best_as0_level"] = int(lvl_hit)

        save_preset(preset0)
        runs_rollback += 1

        if runs_patch >= int(max(0, args.max_candidates)):
            break

    step_jump = 0
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
        step_jump = int(best_step) - int(current_step)

        # update AS0 level if we're in S0
        if int(current_step) == 0:
            anchor = state.get("as0_anchor", {}) or {}
            cur_h2 = {
                "rem_loss": float(base_h2_s.get("loss_by_reason", {}).get("rem", 0.0) or 0.0),
                "mdd": float(base_h2.get("mdd_pct", 0.0) or 0.0),
                "maxR": float(base_h2_s.get("pos_r_max", 0.0) or 0.0),
            }
            lvl_hit, _ = as0_level_hit(anchor, cur_h2, as0_cfg)
            state["as0_level"] = int(max(int(as0_level), int(lvl_hit)))

        action = f"ACCEPT(S{current_step}->S{best_step}, +{step_jump}): {best_label}"

    # 3) persist state + commit accepted preset
    # AS0 stuck tracking + auto-subdivide (H2-only)
    if int(current_step) == 0:
        new_level_now = int(state.get("as0_level", as0_level) or 0)
        if new_level_now <= int(as0_level):
            as0_stuck += 1
        else:
            as0_stuck = 0
        state["as0_stuck"] = int(as0_stuck)

        if int(as0_stuck) >= int(as0_stuck_cycles):
            # subdivide: shrink thresholds by 0.8 to make progress easier
            cfg = dict(state.get("as0_cfg", {}) or {})
            rs = [float(x) * 0.8 for x in (cfg.get("rem_steps") or [])]
            ms = [float(x) * 0.8 for x in (cfg.get("mdd_steps") or [])]
            xs = [float(x) * 0.8 for x in (cfg.get("maxr_steps") or [])]
            cfg["rem_steps"] = rs
            cfg["mdd_steps"] = ms
            cfg["maxr_steps"] = xs
            state["as0_cfg"] = cfg
            state["as0_stuck"] = 0
            state.setdefault("as0_cfg_history", []).append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "reason": f"stuck>={as0_stuck_cycles}: auto-subdivide thresholds x0.8"
            })

    state["tried_labels"] = sorted(list(tried))[-400:]
    state["last_focus"] = focus
    state["last_evidence"] = str(EVIDENCE)
    if applied:
        state["accept_step"] = int(best_step)
        state["last_accept"] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "from": int(current_step),
            "to": int(best_step),
            "jump": int(step_jump),
            "label": best_label,
            "reason": best_reason,
        }
    save_state(state)

    # Policy evaluation (benefit vs risk) — deterministic v0
    try:
        base_h2 = evidence["baseline"]["H2"]["res"]
        base_q4 = evidence["baseline"]["Q4"]["res"]
        cur_h2 = base["H2"]["res"]
        cur_q4 = base["Q4"]["res"]
        base_s = float(score_run(base_h2, base_q4))
        cur_s = float(score_run(cur_h2, cur_q4))
        benefit = max(0.0, cur_s - base_s)
        # risk = q4 deterioration + big mdd
        q4_drop = float(base_q4.get("return_pct", 0.0) or 0.0) - float(cur_q4.get("return_pct", 0.0) or 0.0)
        q4_mdd_up = float(cur_q4.get("mdd_pct", 0.0) or 0.0) - float(base_q4.get("mdd_pct", 0.0) or 0.0)
        risk = max(0.0, q4_drop) + max(0.0, q4_mdd_up * 100.0)
        policy = {"benefit": benefit, "risk": risk, "delta_score": cur_s - base_s, "q4_drop": q4_drop, "q4_mdd_up": q4_mdd_up}
        POLICY_EVAL.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        policy = {"benefit": 0.0, "risk": 0.0}

    if applied:
        # Only proceed if benefit outweighs risk.
        if float(policy.get("benefit", 0.0)) >= float(policy.get("risk", 0.0)):
            msg = (
                f"auto[S{current_step}->S{best_step}]: {best_label} | {best_reason} | "
                f"H2 {base['H2']['res'].get('return_pct')} mdd {base['H2']['res'].get('mdd_pct')} | "
                f"Q4 {base['Q4']['res'].get('return_pct')} mdd {base['Q4']['res'].get('mdd_pct')}"
            )
            # Only commit preset (tmp/** is gitignored, state stays local)
            git(["add", str(PRESET)])
            git(["commit", "-m", msg])
            git(["push"])
        else:
            # reject: rollback preset
            save_preset(preset0)
            applied = False
            action = f"REJECT(policy): benefit<{policy.get('risk',0):.2f} (rollback)"

    # 4) story report
    h2r = base["H2"]["res"]
    q4r = base["Q4"]["res"]
    h2s = base["H2"]["summ"]

    print(f"[arch-loop] {datetime.now().strftime('%Y-%m-%d %H:00')}")
    print(f"- Focus(원인): {focus}")
    if int(current_step) == 0:
        print(f"- AS0: L{int(as0_level)}" + (f" -> L{int(state.get('as0_level', as0_level))}" if applied else ""))
    print(f"- Step: S{current_step}" + (f" -> S{best_step}(+{step_jump})" if applied else ""))
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

    # 5) append run record (for daily report)
    try:
        base_score = score_ext(evidence["baseline"]["H2"]["res"], evidence["baseline"]["Q4"]["res"], evidence["baseline"]["H2"]["summ"])
        new_score = score_ext(base["H2"]["res"], base["Q4"]["res"], base["H2"]["summ"]) 
    except Exception:
        base_score = 0.0
        new_score = 0.0

    def _ascii_safe(s: str) -> str:
        try:
            return (s or "").encode("utf-8", errors="ignore").decode("utf-8")
        except Exception:
            return str(s)

    run_rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "focus": _ascii_safe(focus),
        "ladder": {"A_H2": ladder_a, "B_Q4": ladder_b, "C_Long": "SKIP"},
        "step": {
            "from": int(current_step),
            "to": int(best_step) if applied else int(current_step),
            "jump": int(step_jump) if applied else 0,
        },
        "action": action,
        "applied": bool(applied),
        "reason": best_reason if applied else "",
        "metrics": {
            "H2": {
                "return_pct": base["H2"]["res"].get("return_pct"),
                "mdd_pct": base["H2"]["res"].get("mdd_pct"),
                "entries": base["H2"]["res"].get("entries"),
            },
            "Q4": {
                "return_pct": base["Q4"]["res"].get("return_pct"),
                "mdd_pct": base["Q4"]["res"].get("mdd_pct"),
                "entries": base["Q4"]["res"].get("entries"),
            },
        },
        "delta": {
            "score": float(new_score - base_score),
        },
        "runs": {
            "backtest": int(runs_backtest),
            "patch": int(runs_patch),
            "rollback": int(runs_rollback),
        },
    }

    try:
        RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUNS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run_rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
