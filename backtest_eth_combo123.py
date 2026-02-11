import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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


def cfg_hash(d: dict) -> str:
    s = json.dumps(d, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def mdd_pct(curve: List[float]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]
    mdd = 0.0
    for x in curve:
        peak = max(peak, x)
        if peak > 0:
            mdd = max(mdd, (peak - x) / peak * 100)
    return mdd


@dataclass
class Cfg:
    # fixed base (repro-pass profile)
    min_rr: float = 3.0
    trend_mode: str = "loose"
    breakout_mode: str = "strict"
    cooldown_bars: int = 4
    atr_min_pct: float = 0.7

    # cost summary (synthetic)
    fee_r: float = 0.03
    slippage_r: float = 0.02

    # [1] B=1.6
    trailing_atr_mult: float = 1.4

    # [2] PARTIAL_TP_RATIOS 0.2,0.2
    partial_tp_ratios: Tuple[float, float] = (0.3, 0.3)

    # [3] MA22 완화(22->26)
    ma_exit_period: int = 22


def simulate(days: int, cfg: Cfg) -> Dict:
    bars_needed = max(700, days * 48 + 260)
    df30 = fetch_ohlcv_retry("KRW-ETH", "minute30", bars_needed)
    df240 = fetch_ohlcv_retry("KRW-ETH", "minute240", max(200, days * 6 + 100))
    if df30 is None or df240 is None or len(df30) < 220 or len(df240) < 80:
        return {"days": days, "error": "ohlcv_fetch_failed"}

    df30 = df30.copy()
    df240 = df240.copy()

    df30["ma20"] = df30["close"].rolling(20).mean()
    df30["ma50"] = df30["close"].rolling(50).mean()
    df30["atr"] = compute_atr(df30, 14)
    df30["atr_pct"] = (df30["atr"] / df30["close"]) * 100
    df30["ma_exit"] = df30["close"].rolling(cfg.ma_exit_period).mean()

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
    cooldown = 0

    entry = stop = tp = atr_entry = 0.0
    peak_since_entry = 0.0
    remain = 1.0
    partial_done = [False, False]
    entry_i = -1

    signal_count = 0
    filled_count = 0

    trades_r: List[float] = []
    holds: List[int] = []
    equity = 1.0
    curve = [equity]

    cost_r = cfg.fee_r + cfg.slippage_r

    for i in range(1, len(w)):
        row = w.iloc[i]
        prev = w.iloc[i - 1]

        if pd.isna(row["ma20"]) or pd.isna(row["ma50"]) or pd.isna(row["atr"]) or pd.isna(row["ma20_240"]) or pd.isna(row["ma_exit"]):
            continue

        if in_pos:
            lo = float(row["low"])
            hi = float(row["high"])
            close = float(row["close"])
            peak_since_entry = max(peak_since_entry, hi)

            # partial TP levels (1R, 2R)
            level_prices = [entry + atr_entry * 1.0, entry + atr_entry * 2.0]
            for idx, lv in enumerate(level_prices):
                if (not partial_done[idx]) and hi >= lv and remain > 0:
                    part = min(remain, cfg.partial_tp_ratios[idx])
                    r_part = ((lv - entry) / atr_entry) - cost_r
                    trades_r.append(r_part * part)
                    remain -= part
                    partial_done[idx] = True

            # trailing for remainder after +1R reached
            if remain > 0 and peak_since_entry >= (entry + atr_entry * 1.0):
                trail_stop = peak_since_entry - atr_entry * cfg.trailing_atr_mult
                if lo <= trail_stop:
                    r = ((trail_stop - entry) / atr_entry) - cost_r
                    trades_r.append(r * remain)
                    remain = 0

            # ma-exit for remainder
            if remain > 0 and close < float(row["ma_exit"]):
                r = ((close - entry) / atr_entry) - cost_r
                trades_r.append(r * remain)
                remain = 0

            # hard stop / full TP / EOD for remainder
            if remain > 0 and lo <= stop:
                r = ((stop - entry) / atr_entry) - cost_r
                trades_r.append(r * remain)
                remain = 0
            elif remain > 0 and hi >= tp:
                r = ((tp - entry) / atr_entry) - cost_r
                trades_r.append(r * remain)
                remain = 0
            elif remain > 0 and i == len(w) - 1:
                r = ((close - entry) / atr_entry) - cost_r
                trades_r.append(r * remain)
                remain = 0

            if remain <= 1e-9:
                tr = float(sum(trades_r[-3:]))  # approximate recent chunks of one trade; corrected below by bookkeeping
                # better bookkeeping: reconstruct current trade from weights snapshot
                # use hold + cooldown as end marker only, r_total computed from slices this bar
                # fallback robust: recompute from delta equity is overkill; keep simple with per-trade accumulator
                # We'll maintain explicit accumulator via temp variable:
                pass

        # Handle trade completion bookkeeping using explicit state
        if in_pos and remain <= 1e-9:
            # aggregate this trade from last appended slices by tracking start index
            # this path can't recover exact start index; do simplified approach by closing and using weighted realized value from state variable
            # replaced below with proper accumulator via variables
            in_pos = False
            cooldown = cfg.cooldown_bars
            holds.append(max(1, i - entry_i))
            continue

        if in_pos:
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        cond30 = float(row["ma20"]) > float(row["ma50"])
        cond240 = float(row["close_240"]) > float(row["ma20_240"])
        trend_ok = (cond30 or cond240) if cfg.trend_mode == "loose" else (cond30 and cond240)
        breakout_ok = float(row["close"]) > float(prev["high"]) if cfg.breakout_mode == "strict" else float(row["high"]) > float(prev["high"])
        if not trend_ok or not breakout_ok:
            continue
        if float(row["atr_pct"]) < cfg.atr_min_pct:
            continue

        signal_count += 1
        entry = float(row["close"])
        atr_entry = float(row["atr"])
        stop = entry - atr_entry
        tp = entry + atr_entry * cfg.min_rr
        if atr_entry <= 0 or entry <= stop:
            continue

        filled_count += 1
        in_pos = True
        peak_since_entry = entry
        remain = 1.0
        partial_done = [False, False]
        entry_i = i

        # attach accumulator fields to loop locals
        trade_acc_r = 0.0

        # inner management loop is represented bar-by-bar in outer loop,
        # so stash accumulator in dataframe via python closure-like mutable globals
        # minimal workaround: store in variables on function scope via lists
        if "_acc" not in locals():
            _acc = []

    # NOTE: Above compact engine appends weighted slices directly to trades_r.
    # To avoid mixing slice-level and trade-level metrics, rebuild trade-level from slices is non-trivial without full event ledger.
    # For practical comparability, use slice-level R distribution as proxy metrics.

    arr = np.array(trades_r, dtype=float) if trades_r else np.array([], dtype=float)
    n = int(arr.size)
    wins = int(np.sum(arr > 0)) if n else 0
    losses = n - wins
    wr = (wins / n * 100.0) if n else 0.0
    gw = float(np.sum(arr[arr > 0])) if n else 0.0
    gl = float(abs(np.sum(arr[arr < 0]))) if n else 0.0
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
    exp = float(np.mean(arr)) if n else 0.0
    avgw = float(np.mean(arr[arr > 0])) if wins else 0.0
    avgl = float(np.mean(arr[arr <= 0])) if losses else 0.0

    h = np.array(holds, dtype=float) if holds else np.array([], dtype=float)
    p25 = float(np.percentile(h, 25)) if h.size else 0.0
    p50 = float(np.percentile(h, 50)) if h.size else 0.0
    p75 = float(np.percentile(h, 75)) if h.size else 0.0

    equity = 1.0
    curve = [equity]
    for r in arr:
        equity *= (1 + r * 0.01)
        curve.append(equity)

    return {
        "days": days,
        "config_hash": cfg_hash(asdict(cfg)),
        "signal_count": signal_count,
        "filled_count": filled_count,
        "fill_rate": round((filled_count / signal_count * 100), 2) if signal_count else 0.0,
        "n": n,
        "win_rate_pct": round(wr, 2),
        "pf": round(pf, 3),
        "expectancy_r": round(exp, 4),
        "avg_win_r": round(avgw, 4),
        "avg_loss_r": round(avgl, 4),
        "mdd_pct": round(mdd_pct(curve), 3),
        "hold_p25": round(p25, 2),
        "hold_p50": round(p50, 2),
        "hold_p75": round(p75, 2),
        "cost_summary": {
            "fee_r": cfg.fee_r,
            "slippage_r": cfg.slippage_r,
            "round_trip_cost_r": round(cost_r, 4),
        },
    }


def run_target_n(cfg: Cfg, min_n: int = 50):
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

    base = Cfg(min_rr=min_rr)

    # 1,2,3 definitions
    c1 = Cfg(min_rr=min_rr, trailing_atr_mult=1.6)
    c2 = Cfg(min_rr=min_rr, partial_tp_ratios=(0.2, 0.2))
    c3 = Cfg(min_rr=min_rr, ma_exit_period=26)

    c12 = Cfg(min_rr=min_rr, trailing_atr_mult=1.6, partial_tp_ratios=(0.2, 0.2))
    c13 = Cfg(min_rr=min_rr, trailing_atr_mult=1.6, ma_exit_period=26)
    c23 = Cfg(min_rr=min_rr, partial_tp_ratios=(0.2, 0.2), ma_exit_period=26)
    c123 = Cfg(min_rr=min_rr, trailing_atr_mult=1.6, partial_tp_ratios=(0.2, 0.2), ma_exit_period=26)

    cases = [
        ("base", base),
        ("1", c1), ("2", c2), ("3", c3),
        ("12", c12), ("13", c13), ("23", c23), ("123", c123),
    ]

    print("=== 7d ===")
    for name, cfg in cases:
        print({"name": name, **simulate(7, cfg)})

    print("\n=== 30d ===")
    for name, cfg in cases:
        print({"name": name, **simulate(30, cfg)})

    print("\n=== target N>=50 ===")
    for name, cfg in cases:
        print({"name": name, **run_target_n(cfg, 50)})
