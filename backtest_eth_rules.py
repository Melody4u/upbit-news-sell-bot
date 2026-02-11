import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import pyupbit
from dotenv import load_dotenv


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


@dataclass
class SimConfig:
    min_rr: float = 3.0
    atr_regime_min_pct: float = 0.7
    atr_regime_max_pct: float = 5.0
    post_entry_cooldown_bars: int = 3
    spread_bps_max: float = 18.0  # NOTE: historical orderbook 부재로 bar range proxy 사용


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


def fetch_ohlcv_retry(market: str, interval: str, count: int, retries: int = 4, sleep_sec: float = 0.7):
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


def simulate_eth(days: int, cfg: SimConfig) -> Dict:
    bars_needed = max(400, days * 48 + 120)
    df30 = fetch_ohlcv_retry("KRW-ETH", interval="minute30", count=bars_needed)
    df240 = fetch_ohlcv_retry("KRW-ETH", interval="minute240", count=max(120, days * 6 + 40))

    if df30 is None or df240 is None or len(df30) < 120 or len(df240) < 40:
        return {"days": days, "error": "ohlcv_fetch_failed"}

    df30 = df30.copy()
    df240 = df240.copy()

    df30["ma20"] = df30["close"].rolling(20).mean()
    df30["ma50"] = df30["close"].rolling(50).mean()
    df30["atr"] = compute_atr(df30, 14)
    df30["atr_pct"] = (df30["atr"] / df30["close"]) * 100
    # orderbook 히스토리가 없어 봉 range를 축소계수로 변환한 proxy 사용
    df30["spread_proxy_bps"] = (((df30["high"] - df30["low"]) / df30["close"]) * 10000) * 0.03

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
    cooldown_left = 0

    trades = []
    blocks = {
        "trend_block": 0,
        "breakout_block": 0,
        "atr_regime_block": 0,
        "spread_block_proxy": 0,
        "post_entry_cooldown_block": 0,
    }

    equity = 1.0
    equity_curve = [equity]

    for i in range(1, len(w)):
        row = w.iloc[i]
        prev = w.iloc[i - 1]

        if pd.isna(row["ma20"]) or pd.isna(row["ma50"]) or pd.isna(row["atr"]) or pd.isna(row["ma20_240"]):
            continue

        if in_pos:
            low = float(row["low"])
            high = float(row["high"])
            exit_price: Optional[float] = None

            if low <= stop:
                exit_price = stop
            elif high >= tp:
                exit_price = tp
            elif i == len(w) - 1:
                exit_price = float(row["close"])

            if exit_price is not None:
                r = (exit_price - entry) / (entry - stop) if (entry - stop) > 0 else 0.0
                trades.append(r)
                equity *= (1 + r * 0.01)  # 보수적 스케일(1R=1%)
                equity_curve.append(equity)
                in_pos = False
                cooldown_left = cfg.post_entry_cooldown_bars
            continue

        # entry path
        if cooldown_left > 0:
            blocks["post_entry_cooldown_block"] += 1
            cooldown_left -= 1
            continue

        cond30 = row["ma20"] > row["ma50"]
        cond240 = row["close_240"] > row["ma20_240"]
        breakout = row["close"] > prev["high"]

        if not (cond30 and cond240):
            blocks["trend_block"] += 1
            continue
        if not breakout:
            blocks["breakout_block"] += 1
            continue

        atr_pct = float(row["atr_pct"])
        if atr_pct < cfg.atr_regime_min_pct or atr_pct > cfg.atr_regime_max_pct:
            blocks["atr_regime_block"] += 1
            continue

        spread_proxy = float(row["spread_proxy_bps"])
        if spread_proxy > cfg.spread_bps_max:
            blocks["spread_block_proxy"] += 1
            continue

        entry = float(row["close"])
        atr = float(row["atr"])
        stop = entry - atr
        tp = entry + atr * cfg.min_rr
        if entry <= stop:
            continue
        in_pos = True

    total = len(trades)
    wins = sum(1 for x in trades if x > 0)
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0.0
    gross_win = sum(x for x in trades if x > 0)
    gross_loss = abs(sum(x for x in trades if x < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    mdd = max_drawdown_pct(equity_curve)

    total_blocks = sum(blocks.values())
    block_ratio = {k: round((v / total_blocks * 100), 2) if total_blocks else 0.0 for k, v in blocks.items()}

    return {
        "days": days,
        "cfg": {
            "spread_bps_max": cfg.spread_bps_max,
            "atr_regime_min_pct": cfg.atr_regime_min_pct,
            "post_entry_cooldown_bars": cfg.post_entry_cooldown_bars,
        },
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "mdd_pct": round(mdd, 3),
        "blocks": blocks,
        "block_ratio_pct": block_ratio,
        "holdout_due_to_sample": total < 20,
    }


def verdict(base: Dict, cand: Dict) -> Dict:
    # 실패 기준: PF -0.05 이하, MDD +0.5%p 이상
    if "error" in base or "error" in cand:
        return {"status": "error"}

    pf_delta = cand["profit_factor"] - base["profit_factor"]
    mdd_delta = cand["mdd_pct"] - base["mdd_pct"]
    trade_drop_pct = ((base["trades"] - cand["trades"]) / base["trades"] * 100) if base["trades"] > 0 else 0.0

    fail = (pf_delta <= -0.05) or (mdd_delta >= 0.5)
    pass_gate = (pf_delta >= 0.1) and (trade_drop_pct <= 35.0)
    sample_hold = cand["trades"] < 20

    return {
        "pf_delta": round(pf_delta, 3),
        "mdd_delta_pctp": round(mdd_delta, 3),
        "trade_drop_pct": round(trade_drop_pct, 2),
        "fail": fail,
        "pass": (not fail) and pass_gate and (not sample_hold),
        "sample_hold": sample_hold,
    }


if __name__ == "__main__":
    load_dotenv()
    min_rr = float(os.getenv("MIN_RR", "3.0"))

    base_cfg = SimConfig(min_rr=min_rr, spread_bps_max=18, atr_regime_min_pct=0.7, post_entry_cooldown_bars=3)
    candidates = [
        SimConfig(min_rr=min_rr, spread_bps_max=16, atr_regime_min_pct=0.7, post_entry_cooldown_bars=3),
        SimConfig(min_rr=min_rr, spread_bps_max=14, atr_regime_min_pct=0.7, post_entry_cooldown_bars=3),
        SimConfig(min_rr=min_rr, spread_bps_max=18, atr_regime_min_pct=0.9, post_entry_cooldown_bars=3),
        SimConfig(min_rr=min_rr, spread_bps_max=18, atr_regime_min_pct=1.0, post_entry_cooldown_bars=3),
        SimConfig(min_rr=min_rr, spread_bps_max=18, atr_regime_min_pct=0.7, post_entry_cooldown_bars=4),
    ]

    for days in (7, 30):
        print(f"\n=== ETH {days}d ===")
        base = simulate_eth(days, base_cfg)
        print({"name": "baseline", **base})
        for idx, c in enumerate(candidates, 1):
            r = simulate_eth(days, c)
            v = verdict(base, r)
            print({"name": f"cand_{idx}", **r, "verdict": v})
