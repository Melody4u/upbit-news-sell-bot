import os
from datetime import datetime, timedelta

import pandas as pd
import pyupbit
from dotenv import load_dotenv


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def simulate(market: str, days: int, min_rr: float = 3.0, relaxed: bool = False):
    bars_needed = max(400, days * 48 + 120)  # 30m bars/day = 48
    df30 = pyupbit.get_ohlcv(market, interval="minute30", count=bars_needed)
    df240 = pyupbit.get_ohlcv(market, interval="minute240", count=max(120, days * 6 + 40))

    if df30 is None or len(df30) < 120 or df240 is None or len(df240) < 40:
        return {"market": market, "days": days, "error": "ohlcv_fetch_failed"}

    df30 = df30.copy()
    df240 = df240.copy()

    df30["ma20"] = df30["close"].rolling(20).mean()
    df30["ma50"] = df30["close"].rolling(50).mean()
    df30["atr"] = compute_atr(df30, 14)

    df240["ma20_240"] = df240["close"].rolling(20).mean()
    mtf = df240[["ma20_240", "close"]].reindex(df30.index, method="ffill")
    df30["ma20_240"] = mtf["ma20_240"]
    df30["close_240"] = mtf["close"]

    now_kst = datetime.now() + timedelta(hours=9)
    end = pd.Timestamp(datetime(now_kst.year, now_kst.month, now_kst.day))
    start = end - pd.Timedelta(days=days)

    window = df30[(df30.index >= start) & (df30.index < end)].copy()
    if window.empty:
        return {"market": market, "days": days, "error": "no_bars"}

    in_pos = False
    entry = stop = tp = 0.0
    blocked = 0
    trades = []

    for i in range(1, len(window)):
        row = window.iloc[i]
        prev = window.iloc[i - 1]

        if pd.isna(row["ma20"]) or pd.isna(row["ma50"]) or pd.isna(row["atr"]) or pd.isna(row["ma20_240"]):
            continue

        if not in_pos:
            cond30 = row["ma20"] > row["ma50"]
            cond240 = row["close_240"] > row["ma20_240"]
            breakout = (row["high"] > prev["high"]) if relaxed else (row["close"] > prev["high"])

            if cond30 and cond240 and breakout:
                entry = float(row["close"])
                atr = float(row["atr"])
                rr = max(2.0, min_rr - 1.0) if relaxed else min_rr
                stop = entry - atr
                tp = entry + atr * rr
                if entry <= stop:
                    blocked += 1
                    continue
                in_pos = True
            else:
                blocked += 1
                continue
        else:
            low = float(row["low"])
            high = float(row["high"])
            exit_price = None
            result = None

            if low <= stop:
                exit_price = stop
                result = "loss"
            if high >= tp and exit_price is None:
                exit_price = tp
                result = "win"

            if i == len(window) - 1 and exit_price is None:
                exit_price = float(row["close"])
                result = "win" if exit_price > entry else "loss"

            if exit_price is not None:
                pnl_r = (exit_price - entry) / (entry - stop) if (entry - stop) > 0 else 0
                trades.append({"result": result, "pnl_r": pnl_r})
                in_pos = False

    total = len(trades)
    wins = sum(1 for t in trades if t["result"] == "win")
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0.0

    gross_win = sum(max(0.0, t["pnl_r"]) for t in trades)
    gross_loss = abs(sum(min(0.0, t["pnl_r"]) for t in trades))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    return {
        "market": market,
        "days": days,
        "mode": "relaxed" if relaxed else "strict",
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "blocked_signals": blocked,
    }


if __name__ == "__main__":
    load_dotenv()
    min_rr = float(os.getenv("MIN_RR", "3.0"))

    for days in (7, 30):
        print(f"\n=== RANGE: {days} days ===")
        for market in ("KRW-BTC", "KRW-ETH"):
            strict = simulate(market, days=days, min_rr=min_rr, relaxed=False)
            print(strict)
            if strict.get("trades", 0) == 0:
                print(simulate(market, days=days, min_rr=min_rr, relaxed=True))
