import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Dict, List, Optional

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


def max_drawdown_pct(equity_curve: List[float]) -> float:
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


@dataclass
class SimConfig:
    min_rr: float = 3.0
    atr_min_pct: float = 0.7
    atr_max_pct: float = 5.0
    cooldown_bars: int = 3
    breakout_mode: str = "strict"  # strict: close>prev_high, loose: high>prev_high
    trend_mode: str = "strict"  # strict: ma20>ma50 and close240>ma20_240, loose: one of two


def simulate_eth(days: int, cfg: SimConfig) -> Dict:
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

    blocks = {
        "trend_block": 0,
        "breakout_block": 0,
        "atr_block": 0,
        "cooldown_block": 0,
    }

    trades: List[float] = []
    wins: List[float] = []
    losses: List[float] = []
    hold_bars: List[int] = []

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
            exit_price: Optional[float] = None
            if lo <= stop:
                exit_price = stop
            elif hi >= tp:
                exit_price = tp
            elif i == len(w) - 1:
                exit_price = float(row["close"])

            if exit_price is not None:
                r = (exit_price - entry) / (entry - stop) if (entry - stop) > 0 else 0.0
                trades.append(r)
                if r > 0:
                    wins.append(r)
                else:
                    losses.append(r)
                hold_bars.append(max(1, i - entry_i))
                equity *= (1 + r * 0.01)
                equity_curve.append(equity)
                in_pos = False
                cooldown = cfg.cooldown_bars
            continue

        if cooldown > 0:
            cooldown -= 1
            blocks["cooldown_block"] += 1
            continue

        cond30 = row["ma20"] > row["ma50"]
        cond240 = row["close_240"] > row["ma20_240"]
        trend_ok = (cond30 and cond240) if cfg.trend_mode == "strict" else (cond30 or cond240)
        if not trend_ok:
            blocks["trend_block"] += 1
            continue

        breakout_ok = (row["close"] > prev["high"]) if cfg.breakout_mode == "strict" else (row["high"] > prev["high"])
        if not breakout_ok:
            blocks["breakout_block"] += 1
            continue

        atr_pct = float(row["atr_pct"])
        if atr_pct < cfg.atr_min_pct or atr_pct > cfg.atr_max_pct:
            blocks["atr_block"] += 1
            continue

        entry = float(row["close"])
        atr = float(row["atr"])
        stop = entry - atr
        tp = entry + atr * cfg.min_rr
        if entry <= stop:
            continue
        in_pos = True
        entry_i = i

    total = len(trades)
    win_n = len(wins)
    loss_n = len(losses)
    wr = (win_n / total * 100) if total else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    expectancy = (sum(trades) / total) if total else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    med_hold = float(median(hold_bars)) if hold_bars else 0.0
    mdd = max_drawdown_pct(equity_curve)

    # 무결성 체크
    integrity_ok = (win_n + loss_n == total)

    block_total = sum(blocks.values())
    block_ratio = {k: round((v / block_total * 100), 2) if block_total else 0.0 for k, v in blocks.items()}

    return {
        "days": days,
        "cfg": {
            "breakout_mode": cfg.breakout_mode,
            "trend_mode": cfg.trend_mode,
            "atr_min_pct": cfg.atr_min_pct,
            "cooldown_bars": cfg.cooldown_bars,
        },
        "trades": total,
        "wins": win_n,
        "losses": loss_n,
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 3),
        "expectancy_r": round(expectancy, 4),
        "avg_win_r": round(avg_win, 4),
        "avg_loss_r": round(avg_loss, 4),
        "median_hold_bars": med_hold,
        "mdd_pct": round(mdd, 3),
        "blocks": blocks,
        "block_ratio_pct": block_ratio,
        "integrity_ok": integrity_ok,
    }


def run_with_target_n(cfg: SimConfig, min_n: int = 30, max_days: int = 120):
    for days in (7, 14, 30, 45, 60, 90, 120):
        if days > max_days:
            break
        r = simulate_eth(days, cfg)
        if "error" in r:
            continue
        if r["trades"] >= min_n:
            return r
    return r


if __name__ == "__main__":
    load_dotenv()
    min_rr = float(os.getenv("MIN_RR", "3.0"))

    base_cfg = SimConfig(min_rr=min_rr, breakout_mode="strict", trend_mode="strict", atr_min_pct=0.7, cooldown_bars=3)
    breakout_loose_cfg = SimConfig(min_rr=min_rr, breakout_mode="loose", trend_mode="strict", atr_min_pct=0.7, cooldown_bars=3)
    trend_loose_cfg = SimConfig(min_rr=min_rr, breakout_mode="strict", trend_mode="loose", atr_min_pct=0.7, cooldown_bars=3)

    print("=== fixed windows ===")
    for days in (7, 30):
        print({"name": "baseline", **simulate_eth(days, base_cfg)})
        print({"name": "breakout_loose", **simulate_eth(days, breakout_loose_cfg)})
        print({"name": "trend_loose", **simulate_eth(days, trend_loose_cfg)})

    print("\n=== target N>=30 (auto window) ===")
    print({"name": "baseline", **run_with_target_n(base_cfg, min_n=30)})
    print({"name": "breakout_loose", **run_with_target_n(breakout_loose_cfg, min_n=30)})
    print({"name": "trend_loose", **run_with_target_n(trend_loose_cfg, min_n=30)})
