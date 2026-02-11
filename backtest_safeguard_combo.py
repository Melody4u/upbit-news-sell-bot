import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyupbit
from dotenv import load_dotenv


# 1: drift guard, 2: loss-streak cooldown, 3: cost-sensitivity gate, 4: OOS gate


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
    trend_mode: str = "loose"
    breakout_mode: str = "strict"
    cooldown_bars: int = 4
    atr_min_pct: float = 0.7
    fee_r: float = 0.03
    slippage_r: float = 0.02

    # safeguard #2 params
    loss_streak_n: int = 3
    loss_streak_cooldown_bars: int = 12


def cfg_hash(cfg: Cfg) -> str:
    s = json.dumps(cfg.__dict__, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def simulate(df30: pd.DataFrame, df240: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cfg: Cfg, use_loss_streak: bool) -> Dict:
    d30 = df30[(df30.index >= start) & (df30.index < end)].copy()
    if d30.empty:
        return {"error": "no_bars"}

    in_pos = False
    cooldown = 0
    streak_cooldown = 0
    loss_streak = 0

    entry = stop = tp = 0.0
    entry_i = -1

    signals = 0
    fills = 0
    trades = []
    holds = []

    cost_r = cfg.fee_r + cfg.slippage_r

    for i in range(1, len(d30)):
        row = d30.iloc[i]
        prev = d30.iloc[i - 1]
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
            elif i == len(d30) - 1:
                exit_price = float(row["close"])

            if exit_price is not None:
                r = ((exit_price - entry) / (entry - stop)) - cost_r if (entry - stop) > 0 else -cost_r
                trades.append(r)
                holds.append(max(1, i - entry_i))
                in_pos = False
                cooldown = cfg.cooldown_bars
                if r <= 0:
                    loss_streak += 1
                    if use_loss_streak and loss_streak >= cfg.loss_streak_n:
                        streak_cooldown = cfg.loss_streak_cooldown_bars
                        loss_streak = 0
                else:
                    loss_streak = 0
            continue

        if streak_cooldown > 0:
            streak_cooldown -= 1
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        cond30 = row["ma20"] > row["ma50"]
        cond240 = row["close_240"] > row["ma20_240"]
        trend_ok = (cond30 or cond240) if cfg.trend_mode == "loose" else (cond30 and cond240)
        breakout_ok = (row["close"] > prev["high"]) if cfg.breakout_mode == "strict" else (row["high"] > prev["high"])
        if not trend_ok or not breakout_ok:
            continue
        if float(row["atr_pct"]) < cfg.atr_min_pct:
            continue

        signals += 1
        entry = float(row["close"])
        atr = float(row["atr"])
        stop = entry - atr
        tp = entry + atr * cfg.min_rr
        if entry <= stop:
            continue

        fills += 1
        in_pos = True
        entry_i = i

    arr = np.array(trades, dtype=float) if trades else np.array([])
    n = int(arr.size)
    wins = int(np.sum(arr > 0)) if n else 0
    losses = n - wins
    wr = (wins / n * 100) if n else 0.0
    gw = float(np.sum(arr[arr > 0])) if n else 0.0
    gl = float(abs(np.sum(arr[arr < 0]))) if n else 0.0
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
    exp = float(np.mean(arr)) if n else 0.0
    avgw = float(np.mean(arr[arr > 0])) if wins else 0.0
    avgl = float(np.mean(arr[arr <= 0])) if losses else 0.0
    h = np.array(holds, dtype=float) if holds else np.array([])
    p25 = float(np.percentile(h, 25)) if h.size else 0.0
    p50 = float(np.percentile(h, 50)) if h.size else 0.0
    p75 = float(np.percentile(h, 75)) if h.size else 0.0

    return {
        "signal_count": signals,
        "filled_count": fills,
        "fill_rate": round((fills / signals * 100), 2) if signals else 0.0,
        "n": n,
        "win_rate_pct": round(wr, 2),
        "pf": round(pf, 3),
        "expectancy_r": round(exp, 4),
        "avg_win_r": round(avgw, 4),
        "avg_loss_r": round(avgl, 4),
        "hold_p25": round(p25, 2),
        "hold_p50": round(p50, 2),
        "hold_p75": round(p75, 2),
    }


def pf_exp_with_extra_cost(metrics: Dict, extra_cost_r: float) -> Tuple[float, float]:
    # approximate by shifting expectancy and pf with avg win/loss distances
    exp = metrics["expectancy_r"] - extra_cost_r
    # rough pf adjustment from avg win/loss
    aw = metrics["avg_win_r"] - extra_cost_r
    al = abs(metrics["avg_loss_r"] - extra_cost_r)
    wr = metrics["win_rate_pct"] / 100.0
    if aw <= 0:
        pf = 0.0
    else:
        gross_w = wr * aw
        gross_l = max((1 - wr) * al, 1e-9)
        pf = gross_w / gross_l
    return round(pf, 3), round(exp, 4)


def combo_name(bits: Tuple[int, ...]) -> str:
    return "".join(str(b) for b in bits)


if __name__ == "__main__":
    load_dotenv()
    cfg = Cfg(min_rr=float(os.getenv("MIN_RR", "3.0")))

    df30 = fetch_ohlcv_retry("KRW-ETH", "minute30", 9000)
    df240 = fetch_ohlcv_retry("KRW-ETH", "minute240", 1600)
    if df30 is None or df240 is None:
        raise SystemExit("ohlcv fetch failed")

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
    start_90 = end - pd.Timedelta(days=90)
    start_120 = end - pd.Timedelta(days=120)
    split = end - pd.Timedelta(days=30)  # OOS 30d

    # baseline for drift check (expected from current run snapshot)
    baseline = simulate(df30, df240, start_90, end, cfg, use_loss_streak=False)
    expected_pf = baseline.get("pf", 0.0)
    expected_exp = baseline.get("expectancy_r", 0.0)

    bit_sets = []
    for r in range(1, 5):
        bit_sets.extend(combinations([1, 2, 3, 4], r))

    print({"base_config_hash": cfg_hash(cfg), "baseline_90d": baseline})
    print("=== combos ===")

    for bits in bit_sets:
        name = combo_name(bits)
        use2 = 2 in bits

        main = simulate(df30, df240, start_90, end, cfg, use_loss_streak=use2)

        # gates
        drift_pass = True
        if 1 in bits:
            drift_pass = (abs(main.get("pf", 0.0) - expected_pf) <= 0.05) and (abs(main.get("expectancy_r", 0.0) - expected_exp) <= 0.05)

        cost_pass = True
        pf10 = exp10 = pf20 = exp20 = None
        if 3 in bits:
            pf10, exp10 = pf_exp_with_extra_cost(main, 0.005)
            pf20, exp20 = pf_exp_with_extra_cost(main, 0.01)
            cost_pass = (pf10 >= 1.0 and exp10 >= 0.0 and pf20 >= 1.0 and exp20 >= 0.0)

        oos_pass = True
        oos = None
        if 4 in bits:
            # IS: 120->30, OOS: last 30
            is_metrics = simulate(df30, df240, start_120, split, cfg, use_loss_streak=use2)
            oos = simulate(df30, df240, split, end, cfg, use_loss_streak=use2)
            oos_pass = (oos.get("pf", 0.0) >= 1.0 and oos.get("expectancy_r", 0.0) >= 0.0)
        else:
            is_metrics = None

        overall_pass = drift_pass and cost_pass and oos_pass

        out = {
            "name": name,
            "metrics_90d": main,
            "drift_pass": drift_pass,
            "cost_gate": {"pass": cost_pass, "pf10": pf10, "exp10": exp10, "pf20": pf20, "exp20": exp20} if 3 in bits else None,
            "oos_gate": {"pass": oos_pass, "is": is_metrics, "oos": oos} if 4 in bits else None,
            "overall_pass": overall_pass,
        }
        print(out)
