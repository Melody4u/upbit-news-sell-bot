import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from statistics import median

import pandas as pd
import pyupbit
from dotenv import load_dotenv


def fetch_ohlcv_retry(market: str, interval: str, count: int, retries: int = 5, sleep_sec: float = 0.7):
    last = None
    for _ in range(retries):
        try:
            last = pyupbit.get_ohlcv(market, interval=interval, count=count)
            if last is not None and len(last) > 0:
                return last
        except Exception:
            pass
        time.sleep(sleep_sec)
    return last


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


def max_drawdown_pct(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        if peak > 0:
            dd = (peak - x) / peak * 100.0
            mdd = max(mdd, dd)
    return mdd


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def simulate(days: int, cfg: dict):
    bars_needed = max(500, days * 48 + 160)
    df30 = fetch_ohlcv_retry("KRW-ETH", "minute30", bars_needed)
    df240 = fetch_ohlcv_retry("KRW-ETH", "minute240", max(140, days * 6 + 60))
    if df30 is None or df240 is None or len(df30) < 140 or len(df240) < 50:
        return {"days": days, "error": "ohlcv_fetch_failed"}

    df30 = df30.copy()
    df240 = df240.copy()

    df30["ma20"] = df30["close"].rolling(20).mean()
    df30["ma50"] = df30["close"].rolling(50).mean()
    df30["atr"] = compute_atr(df30, 14)
    df30["atr_pct"] = (df30["atr"] / df30["close"]) * 100

    df240["ma20_240"] = df240["close"].rolling(20).mean()
    mtf = df240[["ma20_240", "close"]].reindex(df30.index, method="ffill")
    df30["ma20_240"] = mtf["ma20_240"]
    df30["close_240"] = mtf["close"]

    now_kst = datetime.now() + timedelta(hours=9)
    end = pd.Timestamp(datetime(now_kst.year, now_kst.month, now_kst.day))
    start = end - pd.Timedelta(days=days)
    w = df30[(df30.index >= start) & (df30.index < end)].copy()
    if w.empty:
        return {"days": days, "error": "no_bars"}

    in_pos = False
    entry = stop = tp = 0.0
    entry_i = -1
    cooldown = 0

    signals = 0
    fills = 0
    trades = []
    hold_bars = []

    # synthetic cost assumption (for summary only)
    fee_r = float(cfg.get("fee_r", 0.03))
    slip_r = float(cfg.get("slippage_r", 0.02))

    equity = 1.0
    equity_curve = [equity]

    for i in range(1, len(w)):
        row = w.iloc[i]
        prev = w.iloc[i - 1]

        if pd.isna(row["ma20"]) or pd.isna(row["ma50"]) or pd.isna(row["atr"]) or pd.isna(row["ma20_240"]):
            continue

        if in_pos:
            lo = float(row["low"])
            hi = float(row["high"])
            exit_price = None
            if lo <= stop:
                exit_price = stop
            elif hi >= tp:
                exit_price = tp
            elif i == len(w) - 1:
                exit_price = float(row["close"])

            if exit_price is not None:
                r_raw = (exit_price - entry) / (entry - stop) if (entry - stop) > 0 else 0.0
                cost_r = fee_r + slip_r
                r = r_raw - cost_r
                trades.append(r)
                hold_bars.append(max(1, i - entry_i))
                equity *= (1 + r * 0.01)
                equity_curve.append(equity)
                in_pos = False
                cooldown = cfg["cooldown_bars"]
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        cond30 = row["ma20"] > row["ma50"]
        cond240 = row["close_240"] > row["ma20_240"]
        trend_ok = (cond30 or cond240) if cfg["trend_mode"] == "loose" else (cond30 and cond240)
        breakout_ok = (row["close"] > prev["high"]) if cfg["breakout_mode"] == "strict" else (row["high"] > prev["high"])

        if not trend_ok or not breakout_ok:
            continue
        if float(row["atr_pct"]) < cfg["atr_min_pct"]:
            continue

        signals += 1
        entry = float(row["close"])
        atr = float(row["atr"])
        stop = entry - atr
        tp = entry + atr * cfg["min_rr"]
        if entry <= stop:
            continue
        fills += 1
        in_pos = True
        entry_i = i

    n = len(trades)
    wins = sum(1 for x in trades if x > 0)
    losses = n - wins
    wr = (wins / n * 100.0) if n else 0.0
    gw = sum(x for x in trades if x > 0)
    gl = abs(sum(x for x in trades if x < 0))
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
    exp = (sum(trades) / n) if n else 0.0
    avgw = (sum(x for x in trades if x > 0) / wins) if wins else 0.0
    avgl = (sum(x for x in trades if x <= 0) / losses) if losses else 0.0
    mdd = max_drawdown_pct(equity_curve)

    p50_hold = float(median(hold_bars)) if hold_bars else 0.0

    return {
        "days": days,
        "cfg": cfg,
        "config_hash": config_hash(cfg),
        "signal_count": signals,
        "filled_count": fills,
        "fill_rate": round((fills / signals * 100.0), 2) if signals else 0.0,
        "trades": n,
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 3),
        "expectancy_r": round(exp, 4),
        "avg_win_r": round(avgw, 4),
        "avg_loss_r": round(avgl, 4),
        "median_hold_bars": p50_hold,
        "mdd_pct": round(mdd, 3),
        "cost_summary": {
            "fee_r": fee_r,
            "slippage_r": slip_r,
            "round_trip_cost_r": round(fee_r + slip_r, 4),
        },
    }


if __name__ == "__main__":
    load_dotenv()
    min_rr = float(os.getenv("MIN_RR", "3.0"))

    # reproduce target: prior winner-ish profile
    cfg = {
        "min_rr": min_rr,
        "trend_mode": "loose",
        "breakout_mode": "strict",
        "cooldown_bars": 4,
        "atr_min_pct": 0.7,
        "fee_r": 0.03,
        "slippage_r": 0.02,
    }

    for d in (7, 30, 90):
        print(simulate(d, cfg))
