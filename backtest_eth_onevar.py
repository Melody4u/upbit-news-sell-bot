import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
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
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


@dataclass
class Cfg:
    min_rr: float = 3.0
    trend_mode: str = "loose"     # fixed
    breakout_mode: str = "strict" # fixed
    cooldown_bars: int = 4         # fixed
    atr_min_pct: float = 0.7

    # one-variable experiment knobs
    early_fail_cut_r: float = 1.0   # A: 1.0 -> 0.9 (loss cut)
    trailing_atr_mult: float = 1.4  # B: 1.4 -> 1.8 (profit extension)


def simulate(days: int, cfg: Cfg) -> Dict:
    bars_needed = max(600, days * 48 + 220)
    df30 = fetch_ohlcv_retry("KRW-ETH", "minute30", bars_needed)
    df240 = fetch_ohlcv_retry("KRW-ETH", "minute240", max(180, days * 6 + 80))
    if df30 is None or df240 is None or len(df30) < 200 or len(df240) < 70:
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
    entry = stop = tp = atr_entry = 0.0
    entry_i = -1
    cooldown = 0
    peak_since_entry = 0.0

    trades_r: List[float] = []
    hold_bars: List[int] = []

    for i in range(1, len(w)):
        row = w.iloc[i]
        prev = w.iloc[i - 1]

        if pd.isna(row["ma20"]) or pd.isna(row["ma50"]) or pd.isna(row["atr"]) or pd.isna(row["ma20_240"]):
            continue

        if in_pos:
            lo = float(row["low"])
            hi = float(row["high"])
            close = float(row["close"])
            peak_since_entry = max(peak_since_entry, hi)

            # A) early loss cut (first 3 bars after entry)
            early_bars = i - entry_i
            if early_bars <= 3:
                early_cut_price = entry - atr_entry * cfg.early_fail_cut_r
                if lo <= early_cut_price:
                    r = (early_cut_price - entry) / atr_entry
                    trades_r.append(r)
                    hold_bars.append(max(1, early_bars))
                    in_pos = False
                    cooldown = cfg.cooldown_bars
                    continue

            # B) profit extension via trailing after +1R reached
            trailing_on = peak_since_entry >= (entry + atr_entry * 1.0)
            if trailing_on:
                trail_stop = peak_since_entry - atr_entry * cfg.trailing_atr_mult
                if lo <= trail_stop:
                    exit_price = trail_stop
                    r = (exit_price - entry) / atr_entry
                    trades_r.append(r)
                    hold_bars.append(max(1, i - entry_i))
                    in_pos = False
                    cooldown = cfg.cooldown_bars
                    continue

            # fallback fixed stop / tp / EOD close
            exit_price: Optional[float] = None
            if lo <= stop:
                exit_price = stop
            elif hi >= tp:
                exit_price = tp
            elif i == len(w) - 1:
                exit_price = close

            if exit_price is not None:
                r = (exit_price - entry) / atr_entry
                trades_r.append(r)
                hold_bars.append(max(1, i - entry_i))
                in_pos = False
                cooldown = cfg.cooldown_bars
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        cond30 = float(row["ma20"]) > float(row["ma50"])
        cond240 = float(row["close_240"]) > float(row["ma20_240"])
        trend_ok = (cond30 or cond240) if cfg.trend_mode == "loose" else (cond30 and cond240)
        if not trend_ok:
            continue

        breakout_ok = float(row["close"]) > float(prev["high"]) if cfg.breakout_mode == "strict" else float(row["high"]) > float(prev["high"])
        if not breakout_ok:
            continue

        atr_pct = float(row["atr_pct"])
        if atr_pct < cfg.atr_min_pct:
            continue

        entry = float(row["close"])
        atr_entry = float(row["atr"])
        stop = entry - atr_entry * 1.0
        tp = entry + atr_entry * cfg.min_rr
        peak_since_entry = entry
        if atr_entry <= 0:
            continue
        in_pos = True
        entry_i = i

    arr = np.array(trades_r, dtype=float) if trades_r else np.array([], dtype=float)
    n = int(arr.size)
    wins = int(np.sum(arr > 0)) if n else 0
    losses = n - wins
    wr = (wins / n * 100.0) if n else 0.0
    gross_win = float(np.sum(arr[arr > 0])) if n else 0.0
    gross_loss = float(abs(np.sum(arr[arr < 0]))) if n else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    expectancy = float(np.mean(arr)) if n else 0.0
    avg_win = float(np.mean(arr[arr > 0])) if wins else 0.0
    avg_loss = float(np.mean(arr[arr <= 0])) if losses else 0.0

    hb = np.array(hold_bars, dtype=float) if hold_bars else np.array([], dtype=float)
    p25 = float(np.percentile(hb, 25)) if hb.size else 0.0
    med = float(np.percentile(hb, 50)) if hb.size else 0.0
    p75 = float(np.percentile(hb, 75)) if hb.size else 0.0

    worst10 = float(np.mean(np.sort(arr)[: max(1, int(np.ceil(n * 0.1))) ])) if n else 0.0

    # cost sensitivity: add extra cost in R per trade (10%/20% worse)
    # baseline synthetic cost 0.05R each trade
    def pf_with_cost(extra_mult: float) -> float:
        if n == 0:
            return 0.0
        cost = 0.05 * (1.0 + extra_mult)
        adj = arr - cost
        gw = float(np.sum(adj[adj > 0]))
        gl = float(abs(np.sum(adj[adj < 0])))
        return (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)

    return {
        "days": days,
        "n": n,
        "win_rate_pct": round(wr, 2),
        "pf": round(pf, 3),
        "expectancy_r": round(expectancy, 4),
        "avg_win_r": round(avg_win, 4),
        "avg_loss_r": round(avg_loss, 4),
        "hold_p25": round(p25, 2),
        "hold_p50": round(med, 2),
        "hold_p75": round(p75, 2),
        "worst10_avg_r": round(worst10, 4),
        "pf_cost_plus10": round(pf_with_cost(0.10), 3),
        "pf_cost_plus20": round(pf_with_cost(0.20), 3),
    }


def run_target_n(cfg: Cfg, min_n: int = 50) -> Dict:
    last = {"error": "no_run"}
    for d in (30, 45, 60, 90, 120, 150):
        r = simulate(d, cfg)
        last = r
        if "error" in r:
            continue
        if r.get("n", 0) >= min_n:
            return r
    return last


if __name__ == "__main__":
    load_dotenv()
    min_rr = float(os.getenv("MIN_RR", "3.0"))

    base = Cfg(min_rr=min_rr, early_fail_cut_r=1.0, trailing_atr_mult=1.4)
    exp_a = Cfg(min_rr=min_rr, early_fail_cut_r=0.9, trailing_atr_mult=1.4)  # loss cut only
    exp_b = Cfg(min_rr=min_rr, early_fail_cut_r=1.0, trailing_atr_mult=1.8)  # profit extension only

    print("=== 7d ===")
    print({"name": "baseline", **simulate(7, base)})
    print({"name": "expA_early_fail_0.9R", **simulate(7, exp_a)})
    print({"name": "expB_trailing_1.8", **simulate(7, exp_b)})

    print("\n=== 30d ===")
    print({"name": "baseline", **simulate(30, base)})
    print({"name": "expA_early_fail_0.9R", **simulate(30, exp_a)})
    print({"name": "expB_trailing_1.8", **simulate(30, exp_b)})

    print("\n=== target N>=50 ===")
    print({"name": "baseline", **run_target_n(base, 50)})
    print({"name": "expA_early_fail_0.9R", **run_target_n(exp_a, 50)})
    print({"name": "expB_trailing_1.8", **run_target_n(exp_b, 50)})
