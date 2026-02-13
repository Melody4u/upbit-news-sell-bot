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
- "Wall St v1" mode in this script uses:
  - D1 regime filter (MA50>MA200 and MA200 slope > 0)
  - Fib pullback zone (0.382~0.618 of recent swing) + EMA20 rebound confirmation
"""

import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List

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


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = compute_atr(df, period).replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0)


def parse_csv_floats(s: str, default: List[float]) -> List[float]:
    try:
        vals = [float(x.strip()) for x in (s or "").split(",") if x.strip()]
        return vals if vals else default
    except Exception:
        return default


def mtf_ok(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 220:
        return False
    close = df["close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    if pd.isna(ma50.iloc[-1]) or pd.isna(ma200.iloc[-1]):
        return False
    return bool(float(ma50.iloc[-1]) > float(ma200.iloc[-1]) and float(close.iloc[-1]) > float(ma50.iloc[-1]))


def d1_regime_ok(dfd: pd.DataFrame, ts: pd.Timestamp, slope_lookback: int = 5) -> bool:
    """D1 regime filter: MA50>MA200 and MA200 slope>0 at timestamp ts."""
    if dfd is None or dfd.empty or not isinstance(dfd.index, pd.DatetimeIndex):
        return False
    sub = dfd[dfd.index <= ts]
    if len(sub) < (200 + slope_lookback + 5):
        return False
    close = sub["close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    if pd.isna(ma50.iloc[-1]) or pd.isna(ma200.iloc[-1]):
        return False
    slope = float(ma200.iloc[-1] - ma200.iloc[-1 - slope_lookback])
    return bool(float(ma50.iloc[-1]) > float(ma200.iloc[-1]) and slope > 0)


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
    wallst_v1: bool = True,
    wallst_soft: bool = True,
    highwr_v1: bool = True,
    scout_min_score: int = 20,
    pos_min_frac: float = 0.15,
    pos_max_frac: float = 1.0,
    fib_lookback: int = 120,
    fib_min: float = 0.382,
    fib_max: float = 0.618,
    fib_swing_confirm: bool = False,
    d1_slope_lookback: int = 5,
    tp1_r: float = 0.8,
    tp2_r: float = 1.6,
    tp1_ratio: float = 0.6,
    be_offset_bps: float = 8.0,
    be_move_mode: str = "hybrid",  # always|weak_only|hybrid
    be_strong_stop_r: float = -0.2,
    adx_min: float = 20.0,
    early_fail_enabled: bool = True,
    early_fail_mode: str = "weak",  # off|weak|always|hybrid
    early_fail_levels_r: str = "0.6,0.9,1.2,1.6,2.0",
    early_fail_cut_ratios: str = "0.1,0.3,0.5,0.7,1.0",
    early_fail_strong_levels_r: str = "1.6,2.0",
    early_fail_strong_cut_ratios: str = "0.5,1.0",
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
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["adx"] = compute_adx(df, 14)

    in_pos = False
    entry = stop = 0.0
    risk0 = 0.0
    tp = 0.0
    tp1 = tp2 = 0.0
    partial_done = False
    early_cut_done = False
    pos_frac = 1.0
    trades = []

    for i in range(220, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        if atr <= 0:
            continue

        if not in_pos:
            # Phase A-ish: require some MTF score (>= scout_min)
            s = score_mtf(ts, df_by_itv, weights)
            if s < int(scout_min_score):
                continue

            # position fraction (score-based sizing proxy)
            s_clamped = max(int(scout_min_score), min(100, int(s)))
            if 100 == int(scout_min_score):
                pos_frac = float(pos_min_frac)
            else:
                pos_frac = float(pos_min_frac) + (float(pos_max_frac) - float(pos_min_frac)) * (
                    (float(s_clamped) - float(scout_min_score)) / max(1.0, (100.0 - float(scout_min_score)))
                )

            if wallst_v1:
                # 1) D1 regime filter (hard by default; soft mode will downsize instead)
                if not d1_regime_ok(dfd, ts, slope_lookback=d1_slope_lookback):
                    if not wallst_soft:
                        continue
                    pos_frac *= 0.4

                # 2) Fib pullback zone (soft by default to keep trade frequency)
                lb = int(max(50, fib_lookback))
                win = df.iloc[max(0, i - lb + 1): i + 1]
                if win.empty:
                    continue
                if fib_swing_confirm:
                    hi_i = int(win["high"].values.argmax())
                    lo_i = int(win["low"].values.argmin())
                    if not (lo_i < hi_i):
                        if not wallst_soft:
                            continue
                        pos_frac *= 0.5

                swing_high = float(win["high"].max())
                swing_low = float(win["low"].min())
                rng = max(1e-9, swing_high - swing_low)
                zone_high = swing_high - (rng * float(fib_min))
                zone_low = swing_high - (rng * float(fib_max))
                in_zone = (float(row["close"]) >= zone_low) and (float(row["close"]) <= zone_high)
                if not in_zone:
                    if not wallst_soft:
                        continue
                    pos_frac *= 0.7

                # 3) Rebound confirmation: close crosses above EMA20 (soft by default)
                ema_now = float(row["ema20"]) if pd.notna(row["ema20"]) else 0.0
                ema_prev = float(prev["ema20"]) if pd.notna(prev["ema20"]) else 0.0
                rebound_ok = (ema_now > 0 and ema_prev > 0 and float(row["close"]) > ema_now and float(prev["close"]) <= ema_prev)
                if not rebound_ok:
                    if not wallst_soft:
                        continue
                    pos_frac *= 0.6

                pos_frac = max(0.05, min(1.0, float(pos_frac)))
            else:
                # Simple breakout trigger to approximate entry (close>prev high)
                if float(row["close"]) <= float(prev["high"]):
                    continue
                pos_frac = max(0.05, min(1.0, float(pos_frac)))

            entry = float(row["close"])
            stop = entry - atr
            risk0 = max(1e-9, entry - stop)
            if highwr_v1:
                # high win-rate exit plan: TP1 partial + BE move + TP2
                tp1 = entry + (atr * float(tp1_r))
                tp2 = entry + (atr * float(tp2_r))
                tp = tp2
            else:
                tp = entry + atr * float(min_rr)
                tp1 = tp2 = 0.0
            if entry <= stop:
                continue
            partial_done = False
            early_cut_done = False
            in_pos = True
        else:
            low = float(row["low"])
            high = float(row["high"])

            # Use initial risk (from entry to initial stop) for R calculations.
            risk_unit = max(1e-9, float(risk0))

            if highwr_v1:
                # Multi-leg exit
                # Early fail cut: ladder in R, mode controls aggressiveness
                if early_fail_enabled and (not early_cut_done):
                    mode = str(early_fail_mode or "weak").lower()
                    adx_now = float(row.get("adx", 0.0) or 0.0)
                    ema_now = float(row.get("ema20", 0.0) or 0.0)
                    weak_ctx = (adx_now < 20.0) or (ema_now > 0 and float(row["close"]) < ema_now)

                    if mode in ("off", "false", "0"):
                        pass
                    else:
                        levels = parse_csv_floats(str(early_fail_levels_r), [0.6, 0.9, 1.2, 1.6, 2.0])
                        ratios = parse_csv_floats(str(early_fail_cut_ratios), [0.1, 0.3, 0.5, 0.7, 1.0])
                        slevels = parse_csv_floats(str(early_fail_strong_levels_r), [1.6, 2.0])
                        sratios = parse_csv_floats(str(early_fail_strong_cut_ratios), [0.5, 1.0])

                        if mode == "hybrid" and (not weak_ctx):
                            levels, ratios = slevels, sratios

                        n = min(len(levels), len(ratios))
                        for lvl, ratio in list(zip(levels, ratios))[:n]:
                            lvl = abs(float(lvl))
                            if lvl <= 0:
                                continue
                            cut_price = entry - (risk_unit * lvl)
                            if low <= cut_price:
                                gross_r_cut = (cut_price - entry) / risk_unit
                                cost_leg = (cost_bps / 10000.0) * 2.0
                                net_r_cut = gross_r_cut - (cost_leg / max(1e-9, risk_unit / entry))
                                trades.append({"result": "loss", "r": float(net_r_cut) * float(ratio) * float(pos_frac), "leg": f"early_cut_{lvl:.2f}R"})
                                early_cut_done = True
                                break

                # If TP1 hit: realize partial, then move stop to BE (+bps)
                if (not partial_done) and high >= tp1 and tp1 > 0:
                    # partial exit at tp1
                    gross_r1 = (tp1 - entry) / risk_unit
                    # cost for this leg (entry+partial exit). assume round trip on that fraction
                    cost_leg = (cost_bps / 10000.0) * 2.0
                    net_r1 = gross_r1 - (cost_leg / max(1e-9, risk_unit / entry))
                    trades.append({"result": "win", "r": float(net_r1) * float(tp1_ratio) * float(pos_frac), "leg": "tp1"})
                    partial_done = True
                    # move stop (context-aware): protect in weak trend, preserve right-tail in strong trend
                    mode_be = str(be_move_mode or "hybrid").lower()
                    adx_now2 = float(row.get("adx", 0.0) or 0.0)
                    ema_now2 = float(row.get("ema20", 0.0) or 0.0)
                    strong_ctx2 = (adx_now2 >= float(adx_min)) and (ema_now2 > 0 and float(row["close"]) >= ema_now2)
                    weak_ctx2 = (adx_now2 < float(adx_min)) or (ema_now2 > 0 and float(row["close"]) < ema_now2)

                    if mode_be == "always":
                        be_price = entry * (1.0 + (be_offset_bps / 10000.0))
                        stop = max(stop, float(be_price))
                    elif mode_be == "weak_only":
                        if weak_ctx2:
                            be_price = entry * (1.0 + (be_offset_bps / 10000.0))
                            stop = max(stop, float(be_price))
                    else:  # hybrid
                        if weak_ctx2 and (not strong_ctx2):
                            be_price = entry * (1.0 + (be_offset_bps / 10000.0))
                            stop = max(stop, float(be_price))
                        elif strong_ctx2:
                            # lift stop closer but leave room: entry + be_strong_stop_r * risk_unit
                            lift = entry + (risk_unit * float(be_strong_stop_r))
                            stop = max(stop, float(lift))

                # For the remainder: check stop/TP2/end
                exit_price: Optional[float] = None
                result = ""
                if low <= stop:
                    exit_price = stop
                    result = "loss" if stop < entry else "win"
                elif high >= tp2 and tp2 > 0:
                    exit_price = tp2
                    result = "win"
                elif i == len(df) - 1:
                    exit_price = float(row["close"])
                    result = "win" if exit_price > entry else "loss"

                if exit_price is not None:
                    gross_r = (exit_price - entry) / risk_unit
                    cost_leg = (cost_bps / 10000.0) * 2.0
                    net_r = gross_r - (cost_leg / max(1e-9, risk_unit / entry))
                    remain_ratio = (1.0 - float(tp1_ratio)) if partial_done else 1.0
                    trades.append({"result": result, "r": float(net_r) * float(remain_ratio) * float(pos_frac), "leg": "rem"})
                    in_pos = False

            else:
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
                    cost = (cost_bps / 10000.0) * 2.0
                    gross_r = (exit_price - entry) / risk_unit
                    net_r = gross_r - (cost / max(1e-9, risk_unit / entry))
                    trades.append({"result": result, "r": float(net_r)})
                    in_pos = False

    total = len(trades)
    days = max(1.0, float((end - start).days))
    trades_per_month = total / (days / 30.0)
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
        "wallst_v1": bool(wallst_v1),
        "wallst_soft": bool(wallst_soft),
        "highwr_v1": bool(highwr_v1),
        "scout_min_score": int(scout_min_score),
        "pos_frac": {"min": float(pos_min_frac), "max": float(pos_max_frac)},
        "trades": total,
        "trades_per_month": round(float(trades_per_month), 2),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "avg_r": round(avg_r, 4),
        "cost_bps": cost_bps,
        "min_rr": min_rr,
        "decision_tf": "minute60",
        "mtf_weights": weights,
        "fib": {"lookback": fib_lookback, "min": fib_min, "max": fib_max, "swing_confirm": fib_swing_confirm},
        "d1_slope_lookback": d1_slope_lookback,
        "tp": {"tp1_r": tp1_r, "tp2_r": tp2_r, "tp1_ratio": tp1_ratio, "be_offset_bps": be_offset_bps, "be_move_mode": be_move_mode, "be_strong_stop_r": be_strong_stop_r, "adx_min": adx_min},
        "early_fail": {
            "enabled": early_fail_enabled,
            "mode": early_fail_mode,
            "levels": early_fail_levels_r,
            "ratios": early_fail_cut_ratios,
            "strong_levels": early_fail_strong_levels_r,
            "strong_ratios": early_fail_strong_cut_ratios,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="KRW-ETH")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive end)")
    ap.add_argument("--min-rr", type=float, default=3.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--wallst-v1", action="store_true", help="Enable Wall St v1 entry: D1 regime + fib pullback + EMA20 rebound")
    ap.add_argument("--wallst-soft", action="store_true", help="Soft gates for fib/rebound to keep trade frequency")
    ap.add_argument("--highwr-v1", action="store_true", help="Enable high win-rate v1 exits: TP1 partial + BE + TP2")
    ap.add_argument("--scout-min-score", type=int, default=20)
    ap.add_argument("--pos-min-frac", type=float, default=0.15)
    ap.add_argument("--pos-max-frac", type=float, default=1.0)
    ap.add_argument("--fib-lookback", type=int, default=120)
    ap.add_argument("--fib-min", type=float, default=0.382)
    ap.add_argument("--fib-max", type=float, default=0.618)
    ap.add_argument("--fib-swing-confirm", action="store_true", help="Require swing confirmation (low before high) in fib window")
    ap.add_argument("--no-fib-swing-confirm", action="store_true", help="Disable fib swing confirmation")
    ap.set_defaults(fib_swing_confirm=False)
    ap.add_argument("--d1-slope-lookback", type=int, default=5)
    ap.add_argument("--tp1-r", type=float, default=0.8)
    ap.add_argument("--tp2-r", type=float, default=1.6)
    ap.add_argument("--tp1-ratio", type=float, default=0.6)
    ap.add_argument("--be-offset-bps", type=float, default=8.0)
    ap.add_argument("--be-move-mode", type=str, default="hybrid", help="always|weak_only|hybrid")
    ap.add_argument("--be-strong-stop-r", type=float, default=-0.2)
    ap.add_argument("--adx-min", type=float, default=20.0)
    ap.add_argument("--early-fail", action="store_true", help="Enable early fail cut")
    ap.add_argument("--early-fail-mode", type=str, default="hybrid", help="weak|always|hybrid")
    ap.add_argument("--early-fail-levels", type=str, default="0.6,0.9,1.2,1.6,2.0")
    ap.add_argument("--early-fail-ratios", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--early-fail-strong-levels", type=str, default="1.6,2.0")
    ap.add_argument("--early-fail-strong-ratios", type=str, default="0.5,1.0")
    args = ap.parse_args()

    start = pd.Timestamp(datetime.fromisoformat(args.start))
    end = pd.Timestamp(datetime.fromisoformat(args.end))

    res = simulate(
        args.market,
        start,
        end,
        min_rr=args.min_rr,
        cost_bps=args.cost_bps,
        wallst_v1=bool(args.wallst_v1),
        wallst_soft=bool(args.wallst_soft),
        highwr_v1=bool(args.highwr_v1),
        scout_min_score=int(args.scout_min_score),
        pos_min_frac=float(args.pos_min_frac),
        pos_max_frac=float(args.pos_max_frac),
        fib_lookback=int(args.fib_lookback),
        fib_min=float(args.fib_min),
        fib_max=float(args.fib_max),
        fib_swing_confirm=(False if bool(args.no_fib_swing_confirm) else (True if bool(args.fib_swing_confirm) else False)),
        d1_slope_lookback=int(args.d1_slope_lookback),
        tp1_r=float(args.tp1_r),
        tp2_r=float(args.tp2_r),
        tp1_ratio=float(args.tp1_ratio),
        be_offset_bps=float(args.be_offset_bps),
        be_move_mode=str(args.be_move_mode),
        be_strong_stop_r=float(args.be_strong_stop_r),
        adx_min=float(args.adx_min),
        early_fail_enabled=bool(args.early_fail),
        early_fail_mode=str(args.early_fail_mode),
        early_fail_levels_r=str(args.early_fail_levels),
        early_fail_cut_ratios=str(args.early_fail_ratios),
        early_fail_strong_levels_r=str(args.early_fail_strong_levels),
        early_fail_strong_cut_ratios=str(args.early_fail_strong_ratios),
    )
    print(res)


if __name__ == "__main__":
    main()
