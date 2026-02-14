"""Architecture auto-loop (research-only).

Workflow (SAFE_PATCH by default):
1) Run backtests (H2/Q4/ALL) with dumps.
2) Summarize: top loss exit_reason, rem-loss share, maxR tail, regime contribution.
3) Simulate persona answers (WallSt risk / Crypto trader execution).
4) Ask Sisyphus (LLM) to propose 1~3 lever changes.
5) Apply patch (whitelisted) -> re-run ladder -> commit with structured message.

NOTE: This script is a scaffold. It does NOT touch bot.py/tradingbot.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
BACKTEST = REPO / "backtest_date_range_phaseA.py"
DUMPDIR = REPO / "tmp" / "arch_loop"


@dataclass
class Slice:
    name: str
    start: str
    end: str


SLICES_DEFAULT = [
    Slice("H2", "2025-07-01", "2026-01-01"),
    Slice("Q4", "2025-10-01", "2026-01-01"),
    Slice("ALL2025", "2025-01-01", "2026-01-01"),
]


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{out[-2000:]}")
    return out


def backtest(slice_: Slice, extra_args: list[str]) -> dict:
    os.makedirs(DUMPDIR, exist_ok=True)
    dump_path = DUMPDIR / f"{slice_.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    cmd = [
        str(PY),
        str(BACKTEST),
        "--market",
        "KRW-BTC",
        "--start",
        slice_.start,
        "--end",
        slice_.end,
        "--dump-trades-csv",
        str(dump_path),
        *extra_args,
    ]
    out = run(cmd)
    # last json-ish dict in stdout
    m = re.findall(r"\{.*\}", out, flags=re.S)
    if not m:
        raise RuntimeError("no result dict found")
    return json.loads(m[-1].replace("'", '"'))  # best-effort


def summarize_dump(base_csv: Path) -> dict:
    legs_path = Path(str(base_csv).replace(".csv", "_legs.csv"))
    pos_path = Path(str(base_csv).replace(".csv", "_pos.csv"))
    legs = pd.read_csv(legs_path)
    pos = pd.read_csv(pos_path)

    # top loss reasons
    loss_by_reason = legs[legs["r"] < 0].groupby("exit_reason")["r"].sum().sort_values()
    counts = legs["exit_reason"].value_counts()

    return {
        "legs": int(len(legs)),
        "positions": int(len(pos)),
        "loss_by_reason": loss_by_reason.head(5).to_dict(),
        "count_by_reason": counts.head(5).to_dict(),
        "pos_r_mean": float(pos["r"].mean()) if len(pos) else 0.0,
        "pos_r_min": float(pos["r"].min()) if len(pos) else 0.0,
        "pos_r_max": float(pos["r"].max()) if len(pos) else 0.0,
    }


def persona_wallst(summary: dict) -> str:
    return (
        "[WallSt persona] rem 손실이 압도적이면 잔여 포지션의 tail-risk 통제가 최우선. "
        "Risk-off에서 잔여 강제 축소/청산 + 잔여 time-stop(시간 기반 종료) 2개 정책을 우선 고려."
    )


def persona_crypto(summary: dict) -> str:
    return (
        "[Crypto trader persona] 승자 꼬리가 너무 짧으면 TP1 비중/BE/트레일 구조가 문제. "
        "돌파 트리거가 죽었다면 레벨 정의를 박스 상단/짧은 lookback으로 현실화하고, 잔여는 ATR 트레일로 꼬리 확보."
    )


def main():
    # Placeholder: just run & summarize once.
    extra = [
        "--box-mode",
        "d1_adx",
        "--box-d1-adx-max",
        "30",
        "--riskoff-mode",
        "close_below_ma200",
        "--riskoff-tf",
        "day",
        "--riskoff-action",
        "block_new",
        "--wallst-v1",
        "--wallst-soft",
        "--highwr-v1",
        "--early-fail",
        "--early-fail-mode",
        "hybrid",
        "--pyramiding",
        "--pos-cap-total",
        "0.90",
        "--swing-stop",
        "--swing-stop-confirm-bars",
        "3",
    ]

    results = []
    for s in SLICES_DEFAULT:
        res = backtest(s, extra)
        dump = Path(res.get("downtrend_layer0", {}).get("dump_path", "")) if False else None
        results.append((s.name, res))
        print(s.name, res.get("return_pct"), res.get("mdd_pct"), res.get("entries"))

    print("OK")


if __name__ == "__main__":
    main()
