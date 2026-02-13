"""Backtest for a fixed date range using Upbit OHLCV.

Goal: approximate Phase A signal gating & count trades/PnL over a historical window.
- Uses candle data only (no orderbook history).
- Intended for quick reality-check / trade frequency, not a perfect execution simulator.

Usage (PowerShell):
  .\.venv\Scripts\python.exe backtest_date_range_phaseA.py --market KRW-ETH --start 2025-01-01 --end 2025-10-31

Notes:
- Interval for decision loop: 60m candles.
- MTF score intervals: 30m/60m/240m/day/week (Phase A default).
- Minute consensus (3m/5m/15m) is NOT used here (too noisy + heavy API for long ranges).
"""

import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyupbit


def fetch_ohlcv_paged(market: str, interval: str, start: pd.Timestamp, end: pd.Timestamp, pause_sec: float = 0.12) -> pd.DataFrame:
    """Fetch OHLCV between [start, end) by paging backwards with `to`.

    Upbit/pyupbit typically returns up to 200 rows per call.
    """
    to: Optional[pd.Timestamp] = end
    chunks = []
    safety = 0
    while True:
        safety += 1
        if safety > 5000:
            break
        df = pyupbit.get_ohlcv(market, interval=interval, count=200, to=to)
        if df is None or df.empty:
            break
        df = df.sort_index()
        chunks.append(df)
        oldest = df.index[0]
        if oldest <= start:
            break
        # move `to` slightly before the oldest candle to avoid overlap
        to = oldest - pd.Timedelta(seconds=1)
        time.sleep(pause_sec)

    if not chunks:
        return pd.DataFrame()

    out = pd.concat(chunks).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[(out.index >= start) & (out.index < end)]
    return out


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def mtf_ok(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 220:
        return False
    close = df["close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    if pd.isna(ma50.iloc[-1]) or pd.isna(ma200.iloc[-1]):
        return False
    return bool(float(ma50.iloc[-1]) > float(ma200.iloc[-1]) and float(close.iloc[-1]) > float(ma50.iloc[-1]))


def score_mtf(row_ts: pd.Timestamp, df_by_itv: Dict[str, pd.DataFrame], weights: Dict[str, int]) -> int:
    score = 0
    for itv, w in weights.items():
        df = df_by_itv.get(itv)
        if df is None or df.empty:
            continue
        # Some intervals may fail to fetch and return an empty frame with RangeIndex.
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        sub = df[df.index <= row_ts]
        if len(sub) < 220:
            continue
        if mtf_ok(sub.tail(260)):
            score += int(w)
    return score


def simulate(
    market: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rr: float = 3.0,
    cost_bps: float = 10.0,
    weights: Optional[Dict[str, int]] = None,
) -> Dict:
    weights = weights or {"minute30": 15, "minute60": 20, "minute240": 30, "day": 20, "week": 15}

    # Fetch needed OHLCV
    df60 = fetch_ohlcv_paged(market, "minute60", start, end)
    if df60.empty or len(df60) < 300:
        return {"error": "ohlcv_fetch_failed", "market": market}

    # MTF frames
    df30 = fetch_ohlcv_paged(market, "minute30", start - pd.Timedelta(days=30), end)
    df240 = fetch_ohlcv_paged(market, "minute240", start - pd.Timedelta(days=120), end)
    dfd = fetch_ohlcv_paged(market, "day", start - pd.Timedelta(days=400), end)
    dfw = fetch_ohlcv_paged(market, "week", start - pd.Timedelta(days=2000), end)

    df_by_itv = {"minute30": df30, "minute60": df60, "minute240": df240, "day": dfd, "week": dfw}

    # Indicators on 60m
    df = df60.copy()
    df["atr"] = compute_atr(df, 14)

    in_pos = False
    entry = stop = tp = 0.0
    trades = []

    for i in range(220, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        if atr <= 0:
            continue

        if not in_pos:
            # Phase A-ish: require some MTF score (>= scout_min default 30)
            s = score_mtf(ts, df_by_itv, weights)
            if s < 30:
                continue

            # Simple breakout trigger to approximate entry (close>prev high)
            if float(row["close"]) <= float(prev["high"]):
                continue

            entry = float(row["close"])
            stop = entry - atr
            tp = entry + atr * float(min_rr)
            if entry <= stop:
                continue
            in_pos = True
        else:
            low = float(row["low"])
            high = float(row["high"])
            exit_price: Optional[float] = None
            result = ""
            if low <= stop:
                exit_price = stop
                result = "loss"
            elif high >= tp:
                exit_price = tp
                result = "win"
            elif i == len(df) - 1:
                exit_price = float(row["close"])
                result = "win" if exit_price > entry else "loss"

            if exit_price is not None:
                # apply simple round-trip cost
                cost = (cost_bps / 10000.0) * 2.0
                gross_r = (exit_price - entry) / max(1e-9, (entry - stop))
                net_r = gross_r - (cost / max(1e-9, (entry - stop) / entry))
                trades.append({"result": result, "r": float(net_r)})
                in_pos = False

    total = len(trades)
    wins = sum(1 for t in trades if t["result"] == "win")
    losses = total - wins
    win_rate = (wins / total * 100.0) if total else 0.0

    gross_win = sum(max(0.0, t["r"]) for t in trades)
    gross_loss = abs(sum(min(0.0, t["r"]) for t in trades))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    avg_r = float(np.mean([t["r"] for t in trades])) if trades else 0.0

    return {
        "market": market,
        "start": str(start.date()),
        "end": str((end - pd.Timedelta(days=0)).date()),
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "avg_r": round(avg_r, 4),
        "cost_bps": cost_bps,
        "min_rr": min_rr,
        "decision_tf": "minute60",
        "mtf_weights": weights,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="KRW-ETH")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive end)")
    ap.add_argument("--min-rr", type=float, default=3.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    start = pd.Timestamp(datetime.fromisoformat(args.start))
    end = pd.Timestamp(datetime.fromisoformat(args.end))

    res = simulate(args.market, start, end, min_rr=args.min_rr, cost_bps=args.cost_bps)
    print(res)


if __name__ == "__main__":
    main()
