"""Architecture auto-loop (V2 research-only).

This script is meant to be the *workflow engine*:
- Build evidence from dumps (reverse-analysis)
- Simulate personas (WallSt / Crypto trader)
- Ask Sisyphus to compress into 1~3 levers (later: LLM call + fallback policy)
- Apply safe patches (later)

For the first boot, we implement:
- Validation ladder A/B (H2/Q4/ALL optional)
- Dump summaries (exit_reason loss attribution + worst positions)

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
DUMPROOT = REPO / "tmp" / "arch_loop"


@dataclass
class Slice:
    name: str
    start: str
    end: str


SLICES = [
    Slice("H2", "2025-07-01", "2026-01-01"),
    Slice("Q4", "2025-10-01", "2026-01-01"),
    Slice("ALL2025", "2025-01-01", "2026-01-01"),
]


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{out[-4000:]}")
    return out


def parse_result_dict(stdout: str) -> dict:
    # backtests print a python dict
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    # find last line that looks like a dict
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

    worst_pos = pos.sort_values("r").head(10)[
        ["pos_id", "entry_ts", "exit_ts", "entry_ctx", "addon_count", "mfe_r", "mae_r", "r", "exit_reason_final"]
    ].to_dict(orient="records")

    return {
        "legs": int(len(legs)),
        "positions": int(len(pos)),
        "pos_r_mean": float(pos["r"].mean()) if len(pos) else 0.0,
        "pos_r_min": float(pos["r"].min()) if len(pos) else 0.0,
        "pos_r_max": float(pos["r"].max()) if len(pos) else 0.0,
        "loss_by_reason": loss_by_reason,
        "count_by_reason": count_by_reason,
        "worst_positions": worst_pos,
    }


def persona_wallst(summary: dict) -> str:
    # simple deterministic persona stub
    loss_by_reason = summary.get("loss_by_reason", {})
    rem_loss = float(loss_by_reason.get("rem", 0.0))
    return (
        "[WallSt persona] 잔여(rem) 손실이 크면 잔여를 '옵션'처럼 관리해야 함. "
        "Risk-off에서 잔여 강제 축소/청산 + 잔여 time-stop(시간 기반 종료) 2개를 우선 고려. "
        f"(rem loss sum={rem_loss:.2f}R)"
    )


def persona_crypto(summary: dict) -> str:
    pr_max = float(summary.get("pos_r_max", 0.0))
    return (
        "[Crypto trader persona] 승자 꼬리(maxR)가 짧으면 TP1/BE/트레일 구조가 수익을 죽임. "
        "TP1 비중을 줄이고 잔여는 ATR 트레일 또는 time-stop로 손익비를 정리. "
        f"(maxR={pr_max:.2f})"
    )


def main():
    print("== arch_autoloop v2 :: boot ==")
    results = {}
    for s in SLICES[:2]:  # A/B only for first boot
        res, dump = backtest(s, dump_name="v2")
        summ = summarize_dumps(dump)
        results[s.name] = {"res": res, "dump": str(dump), "summ": summ}

        print(f"\n[{s.name}] return={res.get('return_pct')} mdd={res.get('mdd_pct')} entries={res.get('entries')}")
        print(" top loss_by_reason:", json.dumps(summ["loss_by_reason"], ensure_ascii=False))
        print(" pos_r mean/min/max:", summ["pos_r_mean"], summ["pos_r_min"], summ["pos_r_max"])

    # personas based on H2 summary (primary)
    h2 = results.get("H2", {}).get("summ", {})
    print("\n== personas (simulated) ==")
    print(persona_wallst(h2))
    print(persona_crypto(h2))

    print("\n== next ==")
    print("- Next we will plug Sisyphus call + safe patch application (1~3 levers) with rollback.")
    print("OK")


if __name__ == "__main__":
    main()
