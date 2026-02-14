"""Architecture auto-loop (V2 research-only).

Now includes SAFE_PATCH (preset-only) auto-improvement:
- Baseline ladder (H2/Q4)
- If A(H2) fails with rem-loss dominant -> apply 1~2 levers:
  * rem_exit_on_riskoff
  * rem_time_stop_bars
- Re-run ladder; accept if H2 improves without catastrophic Q4 regression.
- Commit preset change with structured message; else rollback.

No live trading logic is touched.
"""

from __future__ import annotations

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


def accept_patch(base_h2: dict, base_q4: dict, new_h2: dict, new_q4: dict) -> bool:
    # Accept if H2 improves meaningfully and Q4 doesn't catastrophically regress.
    def _r(x):
        return float(x.get("return_pct", 0.0) or 0.0)

    def _mdd(x):
        return float(x.get("mdd_pct", 0.0) or 0.0)

    h2_r_improve = _r(new_h2) - _r(base_h2)
    h2_mdd_improve = _mdd(base_h2) - _mdd(new_h2)

    q4_r_drop = _r(base_q4) - _r(new_q4)

    return bool((h2_r_improve >= 2.0 or h2_mdd_improve >= 0.02) and q4_r_drop <= 3.0)


def main():
    os.makedirs(DUMPROOT, exist_ok=True)

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

    # Ladder status (simple): PASS if return>=10 and mdd<=0.30 for H2; Q4 pass if return>=0
    ladder_a = "PASS" if (float(base_h2.get("return_pct", 0.0) or 0.0) >= 10.0 and float(base_h2.get("mdd_pct", 1.0) or 1.0) <= 0.30) else "FAIL"
    ladder_b = "PASS" if float(base_q4.get("return_pct", 0.0) or 0.0) >= 0.0 else "FAIL"

    action = "관측만"
    runs_backtest = 2
    runs_patch = 0
    runs_rollback = 0

    # 2) SAFE_PATCH: if A fails and rem dominates, try riskoff+time-stop
    applied = False
    preset0 = load_preset()

    rem_loss = float(base_h2_s.get("loss_by_reason", {}).get("rem", 0.0))
    if ladder_a == "FAIL" and rem_loss < -10.0:
        p = load_preset()
        sim = dict(p.get("simulate_kwargs", {}) or {})
        # candidate levers (1~2): enable riskoff remainder exit + time stop 72 bars
        sim["rem_exit_on_riskoff"] = True
        sim["rem_time_stop_bars"] = 72
        p["simulate_kwargs"] = sim
        save_preset(p)
        runs_patch += 1
        action = "SAFE_PATCH: rem_exit_on_riskoff=true + rem_time_stop_bars=72"

        # rerun ladder
        new = {}
        for s in SLICES:
            res, dump = backtest(s, dump_name="patch")
            summ = summarize_dumps(dump)
            new[s.name] = {"res": res, "summ": summ}
        runs_backtest += 2

        if accept_patch(base_h2, base_q4, new["H2"]["res"], new["Q4"]["res"]):
            applied = True
            base = new
            base_h2 = base["H2"]["res"]
            base_q4 = base["Q4"]["res"]
            base_h2_s = base["H2"]["summ"]
            ladder_a = "PASS" if (float(base_h2.get("return_pct", 0.0) or 0.0) >= 10.0 and float(base_h2.get("mdd_pct", 1.0) or 1.0) <= 0.30) else "FAIL"
            ladder_b = "PASS" if float(base_q4.get("return_pct", 0.0) or 0.0) >= 0.0 else "FAIL"
        else:
            # rollback preset
            save_preset(preset0)
            runs_rollback += 1

    # 3) commit if applied
    if applied:
        msg = (
            f"auto: rem_exit_on_riskoff false->true, rem_time_stop_bars 0->72 | "
            f"H2 return {base['H2']['res'].get('return_pct')} / mdd {base['H2']['res'].get('mdd_pct')} | "
            f"Q4 return {base['Q4']['res'].get('return_pct')} / mdd {base['Q4']['res'].get('mdd_pct')}"
        )
        git(["add", str(PRESET)])
        git(["commit", "-m", msg])
        git(["push"])

    # 4) story report (printed; cron will post this)
    h2r = base["H2"]["res"]
    q4r = base["Q4"]["res"]
    h2s = base["H2"]["summ"]

    print(f"[arch-loop] {datetime.now().strftime('%Y-%m-%d %H:00')}")
    print(f"- Focus(원인): {focus}")
    print(f"- Ladder: A(H2)={ladder_a}, B(Q4)={ladder_b}, C(Long)=SKIP")
    print(f"- Action(이번): {action}")
    print(f"- Result(전/후): 관측 기반" if (runs_patch == 0) else "- Result(전/후): 패치 적용")
    print(f"  · H2: return {h2r.get('return_pct')} / mdd {h2r.get('mdd_pct')} / entries {h2r.get('entries')}")
    print(f"  · Q4: return {q4r.get('return_pct')} / mdd {q4r.get('mdd_pct')} / entries {q4r.get('entries')}")

    # evidence
    loss = h2s.get("loss_by_reason", {})
    top2 = list(loss.items())[:2]
    top2s = ", ".join([f"{k} {v:.2f}R" for k, v in top2]) if top2 else "n/a"
    print(
        f"- Evidence: H2 loss TOP2 {top2s} / posR mean {h2s.get('pos_r_mean'):.3f} min {h2s.get('pos_r_min'):.3f} max {h2s.get('pos_r_max'):.3f}"
    )
    print(f"- Runs: backtest {runs_backtest}회 / patch {runs_patch}회 / rollback {runs_rollback}회")
    print("- Next: (WIP) 시시퍼스 최소변수 정리 + 레버 교체 정책 연결")

    print(persona_wallst(h2s))
    print(persona_crypto(h2s))


if __name__ == "__main__":
    main()
