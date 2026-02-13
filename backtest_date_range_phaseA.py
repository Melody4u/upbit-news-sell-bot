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
    good_gate_mode: str = "none",  # none|l1|l2|l3
    good_gate_l1_score: int = 55,
    good_gate_l2_score: int = 65,
    good_gate_l3_score: int = 78,
    fib_lookback: int = 120,
    fib_min: float = 0.382,
    fib_max: float = 0.618,
    fib_swing_confirm: bool = False,
    d1_slope_lookback: int = 5,
    tp1_r: float = 0.8,
    tp2_r: float = 1.6,
    rr_target_atr_mult: float = 1.0,
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
    pyramiding_enabled: bool = True,
    pos_cap_total: float = 0.90,
    addon_fracs: str = "0.10,0.07,0.05,0.03",
    addon_min_bars: int = 1,
    addon_hold_bars: int = 2,
    addon_max_l1: int = 1,
    addon_max_l2: int = 2,
    addon_max_l3: int = 1,
    addon_frac_l1: float = 0.07,
    addon_frac_l2: float = 0.10,
    addon_frac_l3: float = 0.15,
    addon_stop_lift_r: float = -0.2,
    risk_per_trade: float = 0.015,
    mdd_limit_pct: float = 0.30,
    downtrend_mode: str = "or",  # h4|d1|or|off
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

    # Indicators for gating on H4/D1
    for _name, _df in [("h4", df240), ("d1", dfd)]:
        if _df is None or _df.empty or not isinstance(_df.index, pd.DatetimeIndex):
            continue
        _df["ema_fast"] = _df["close"].ewm(span=20, adjust=False).mean()
        _df["ema_slow"] = _df["close"].ewm(span=50, adjust=False).mean()
        _df["adx"] = compute_adx(_df, 14)

    in_pos = False
    entry = stop = 0.0
    risk0 = 0.0
    tp = 0.0
    tp1 = tp2 = 0.0
    partial_done = False
    early_cut_done = False
    pos_frac = 1.0
    entry_ctx = ""
    entry_ctx_counts = {"l1": 0, "l2": 0, "other": 0}
    addon_count = 0
    last_addon_i = -10_000
    gate_hold = 0
    addon_counts = {"l1": 0, "l2": 0, "l3": 0}

    entries = 0
    legs_count = 0
    r_pos_sum = 0.0
    r_pos_count = 0

    equity = 1.0
    equity_peak = 1.0
    mdd_pct = 0.0

    # per-position accumulators
    r_total_pos = 0.0

    legs = []  # execution legs: early_cut/tp1/rem
    positions = []  # per-entry totals

    for i in range(220, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        if atr <= 0:
            continue

        if not in_pos:
            # Downtrend gate (하락추세): block entries depending on mode
            sub240_dt = df240[df240.index <= ts] if isinstance(df240.index, pd.DatetimeIndex) else pd.DataFrame()
            subd1_dt = dfd[dfd.index <= ts] if isinstance(dfd.index, pd.DatetimeIndex) else pd.DataFrame()

            # H4 downtrend: MA50 < MA200 on H4
            h4_down = False
            if sub240_dt is not None and not sub240_dt.empty:
                c = sub240_dt["close"]
                ma50 = c.rolling(50).mean()
                ma200 = c.rolling(200).mean()
                if len(ma200) > 0 and pd.notna(ma50.iloc[-1]) and pd.notna(ma200.iloc[-1]):
                    h4_down = bool(float(ma50.iloc[-1]) < float(ma200.iloc[-1]))

            # D1 downtrend: EMA20 < EMA50 on D1
            d1_down = False
            if subd1_dt is not None and not subd1_dt.empty and "ema_fast" in subd1_dt.columns and "ema_slow" in subd1_dt.columns:
                ef = float(subd1_dt["ema_fast"].iloc[-1]) if pd.notna(subd1_dt["ema_fast"].iloc[-1]) else 0.0
                es = float(subd1_dt["ema_slow"].iloc[-1]) if pd.notna(subd1_dt["ema_slow"].iloc[-1]) else 0.0
                if ef > 0 and es > 0:
                    d1_down = bool(ef < es)

            dm = str(downtrend_mode or "or").lower()
            if dm not in ("h4", "d1", "or", "off"):
                dm = "or"
            downtrend = (h4_down or d1_down) if dm == "or" else (h4_down if dm == "h4" else (d1_down if dm == "d1" else False))
            if downtrend:
                continue

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
                d1_ok = d1_regime_ok(dfd, ts, slope_lookback=d1_slope_lookback)
                if not d1_ok:
                    if not wallst_soft:
                        continue
                    pos_frac *= 0.4

                # good-gate context (Level 1/2/3) used for entry labeling / optional entry gating
                # L1: 현실적 good (2-of-3: score, H4 trend, H4 ADX)
                # L2: strong (2-of-3: score, D1 trend, D1 ADX)
                # L3: very strong (3-of-4: score, D1 trend, H4 trend, D1 ADX)
                sub240 = df240[df240.index <= ts] if isinstance(df240.index, pd.DatetimeIndex) else pd.DataFrame()
                subd1 = dfd[dfd.index <= ts] if isinstance(dfd.index, pd.DatetimeIndex) else pd.DataFrame()

                def _last_val(_df, col: str, default: float = 0.0) -> float:
                    try:
                        if _df is None or _df.empty or col not in _df.columns:
                            return float(default)
                        v = _df[col].iloc[-1]
                        return float(v) if pd.notna(v) else float(default)
                    except Exception:
                        return float(default)

                h4_fast = _last_val(sub240, "ema_fast")
                h4_slow = _last_val(sub240, "ema_slow")
                h4_adx = _last_val(sub240, "adx")
                d1_fast = _last_val(subd1, "ema_fast")
                d1_slow = _last_val(subd1, "ema_slow")
                d1_adx = _last_val(subd1, "adx")

                h4_trend = (h4_fast > 0 and h4_slow > 0 and h4_fast > h4_slow)
                d1_trend = (d1_fast > 0 and d1_slow > 0 and d1_fast > d1_slow)

                l1_score_ok = int(s) >= int(good_gate_l1_score)
                l2_score_ok = int(s) >= int(good_gate_l2_score)
                l3_score_ok = int(s) >= int(good_gate_l3_score)

                l1 = (sum([l1_score_ok, h4_trend, (h4_adx >= 18.0)]) >= 2)
                l2 = (sum([l2_score_ok, d1_trend, (d1_adx >= float(adx_min))]) >= 2)
                l3 = (sum([l3_score_ok, d1_trend, h4_trend, (d1_adx >= 25.0)]) >= 3)

                # optional entry gating
                gmode = str(good_gate_mode).lower()
                if gmode == "l3" and (not l3):
                    continue
                if gmode == "l2" and (not (l2 or l3)):
                    continue
                if gmode == "l1" and (not (l1 or l2 or l3)):
                    continue

                entry_ctx = "l3" if l3 else ("l2" if l2 else ("l1" if l1 else "other"))

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
                entry_ctx = "other"
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
                # basic RR target using ATR multiple
                tp = entry + atr * float(rr_target_atr_mult)
                tp1 = tp2 = 0.0
            # (basic RR mode handled above)
            if entry <= stop:
                continue
            partial_done = False
            early_cut_done = False
            in_pos = True
            entries += 1
            addon_count = 0
            last_addon_i = i
            gate_hold = 0
            r_total_pos = 0.0
            entry_ctx_counts[entry_ctx] = int(entry_ctx_counts.get(entry_ctx, 0)) + 1
        else:
            low = float(row["low"])
            high = float(row["high"])

            # Use initial risk (from entry to initial stop) for R calculations.
            risk_unit = max(1e-9, float(risk0))

            if highwr_v1:
                # Pyramiding(scale-in): expand position only when L1/L2 holds; cap total exposure
                if pyramiding_enabled and wallst_v1:
                    mode_gate = str(good_gate_mode or "none").lower()
                    # compute current ctx (same proxy as entry)
                    s_now = score_mtf(ts, df_by_itv, weights)
                    d1_ok_now = d1_regime_ok(dfd, ts, slope_lookback=d1_slope_lookback)
                    sub240_now = df240[df240.index <= ts] if isinstance(df240.index, pd.DatetimeIndex) else pd.DataFrame()
                    h4_ok_now = mtf_ok(sub240_now.tail(260)) if (sub240_now is not None and len(sub240_now) >= 220) else False
                    ema_now_ctx2 = float(row.get("ema20", 0.0) or 0.0)
                    adx_now_ctx2 = float(row.get("adx", 0.0) or 0.0)
                    # compute good-gate levels at current ts
                    sub240_now = df240[df240.index <= ts] if isinstance(df240.index, pd.DatetimeIndex) else pd.DataFrame()
                    subd1_now = dfd[dfd.index <= ts] if isinstance(dfd.index, pd.DatetimeIndex) else pd.DataFrame()
                    h4_fast2 = float(sub240_now["ema_fast"].iloc[-1]) if (not sub240_now.empty and "ema_fast" in sub240_now.columns and pd.notna(sub240_now["ema_fast"].iloc[-1])) else 0.0
                    h4_slow2 = float(sub240_now["ema_slow"].iloc[-1]) if (not sub240_now.empty and "ema_slow" in sub240_now.columns and pd.notna(sub240_now["ema_slow"].iloc[-1])) else 0.0
                    h4_adx2 = float(sub240_now["adx"].iloc[-1]) if (not sub240_now.empty and "adx" in sub240_now.columns and pd.notna(sub240_now["adx"].iloc[-1])) else 0.0
                    d1_fast2 = float(subd1_now["ema_fast"].iloc[-1]) if (not subd1_now.empty and "ema_fast" in subd1_now.columns and pd.notna(subd1_now["ema_fast"].iloc[-1])) else 0.0
                    d1_slow2 = float(subd1_now["ema_slow"].iloc[-1]) if (not subd1_now.empty and "ema_slow" in subd1_now.columns and pd.notna(subd1_now["ema_slow"].iloc[-1])) else 0.0
                    d1_adx2 = float(subd1_now["adx"].iloc[-1]) if (not subd1_now.empty and "adx" in subd1_now.columns and pd.notna(subd1_now["adx"].iloc[-1])) else 0.0

                    h4_trend2 = (h4_fast2 > 0 and h4_slow2 > 0 and h4_fast2 > h4_slow2)
                    d1_trend2 = (d1_fast2 > 0 and d1_slow2 > 0 and d1_fast2 > d1_slow2)

                    l1_now = (sum([(int(s_now) >= int(good_gate_l1_score)), h4_trend2, (h4_adx2 >= 18.0)]) >= 2)
                    l2_now = (sum([(int(s_now) >= int(good_gate_l2_score)), d1_trend2, (d1_adx2 >= float(adx_min))]) >= 2)
                    l3_now = (sum([(int(s_now) >= int(good_gate_l3_score)), d1_trend2, h4_trend2, (d1_adx2 >= 25.0)]) >= 3)
                    ctx_now = "l3" if l3_now else ("l2" if l2_now else ("l1" if l1_now else "other"))

                    if mode_gate == "l3":
                        good_now = bool(l3_now)
                    elif mode_gate == "l2":
                        good_now = bool(l2_now or l3_now)
                    elif mode_gate == "l1":
                        good_now = bool(l1_now or l2_now or l3_now)
                    else:
                        good_now = bool(l1_now or l2_now or l3_now)

                    gate_hold = (gate_hold + 1) if good_now else 0

                    # decide addon allowance by ctx
                    # add-on only when in-profit (R>0), to avoid averaging down.
                    unreal_r = (float(row["close"]) - float(entry)) / max(1e-9, float(risk_unit))

                    if ctx_now == "l3":
                        max_addons = int(addon_max_l3)
                        add_f = float(addon_frac_l3)
                    elif ctx_now == "l2":
                        max_addons = int(addon_max_l2)
                        add_f = float(addon_frac_l2)
                    else:
                        max_addons = int(addon_max_l1)
                        add_f = float(addon_frac_l1)

                    can_add = (
                        good_now
                        and (unreal_r > 0.0)
                        and (gate_hold >= int(addon_hold_bars))
                        and ((i - int(last_addon_i)) >= int(addon_min_bars))
                        and (int(addon_count) < int(max_addons))
                        and (float(pos_frac) < float(pos_cap_total) - 1e-9)
                    )

                    if can_add:
                        new_pos = min(float(pos_cap_total), float(pos_frac) + max(0.0, float(add_f)))
                        if new_pos > float(pos_frac) + 1e-9:
                            pos_frac = new_pos
                            addon_count += 1
                            last_addon_i = i
                            if ctx_now in ("l1", "l2", "l3"):
                                addon_counts[ctx_now] = int(addon_counts.get(ctx_now, 0)) + 1
                            # lift stop along with add-on (risk sync)
                            stop = max(float(stop), float(entry) + float(risk_unit) * float(addon_stop_lift_r))

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
                                r_leg = float(net_r_cut) * float(ratio) * float(pos_frac)
                                legs.append({"result": "loss", "r": r_leg, "leg": f"early_cut_{lvl:.2f}R"})
                                legs_count += 1
                                r_total_pos += float(r_leg)
                                early_cut_done = True
                                break

                # If TP1 hit: realize partial, then move stop to BE (+bps)
                if (not partial_done) and high >= tp1 and tp1 > 0:
                    # partial exit at tp1
                    gross_r1 = (tp1 - entry) / risk_unit
                    # cost for this leg (entry+partial exit). assume round trip on that fraction
                    cost_leg = (cost_bps / 10000.0) * 2.0
                    net_r1 = gross_r1 - (cost_leg / max(1e-9, risk_unit / entry))
                    r_leg = float(net_r1) * float(tp1_ratio) * float(pos_frac)
                    legs.append({"result": "win", "r": r_leg, "leg": "tp1"})
                    legs_count += 1
                    r_total_pos += float(r_leg)
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
                    r_leg = float(net_r) * float(remain_ratio) * float(pos_frac)
                    legs.append({"result": result, "r": r_leg, "leg": "rem"})
                    legs_count += 1
                    r_total_pos += float(r_leg)

                    # Close position: apply equity update ONCE per position using aggregated R
                    r_pos = float(r_total_pos)
                    positions.append({"r": r_pos, "entry_ctx": entry_ctx, "addon_count": int(addon_count)})
                    r_pos_sum += r_pos
                    r_pos_count += 1

                    equity *= max(1e-9, (1.0 + (r_pos * float(risk_per_trade))))
                    equity_peak = max(equity_peak, equity)
                    mdd_pct = max(mdd_pct, (equity_peak - equity) / equity_peak)

                    # reset per-position accumulators
                    r_total_pos = 0.0

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
                    # basic mode: one leg == one position close
                    r_pos = float(net_r)
                    legs.append({"result": result, "r": r_pos, "leg": "basic"})
                    legs_count += 1
                    positions.append({"r": r_pos, "entry_ctx": entry_ctx, "addon_count": int(addon_count)})
                    r_pos_sum += r_pos
                    r_pos_count += 1

                    equity *= max(1e-9, (1.0 + (r_pos * float(risk_per_trade))))
                    equity_peak = max(equity_peak, equity)
                    mdd_pct = max(mdd_pct, (equity_peak - equity) / equity_peak)
                    in_pos = False

    total_legs = int(legs_count)
    total_entries = int(entries)
    days = max(1.0, float((end - start).days))
    entries_per_month = total_entries / (days / 30.0)
    legs_per_month = total_legs / (days / 30.0)

    wins = sum(1 for t in legs if t["result"] == "win")
    losses = total_legs - wins
    win_rate = (wins / total_legs * 100.0) if total_legs else 0.0

    gross_win = sum(max(0.0, t["r"]) for t in legs)
    gross_loss = abs(sum(min(0.0, t["r"]) for t in legs))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    avg_r = float(np.mean([t["r"] for t in legs])) if legs else 0.0
    avg_r_pos = (float(r_pos_sum) / float(r_pos_count)) if r_pos_count > 0 else 0.0

    mdd_ok = bool(float(mdd_pct) <= float(mdd_limit_pct))

    return {
        "market": market,
        "start": str(start.date()),
        "end": str((end - pd.Timedelta(days=0)).date()),
        "wallst_v1": bool(wallst_v1),
        "wallst_soft": bool(wallst_soft),
        "highwr_v1": bool(highwr_v1),
        "scout_min_score": int(scout_min_score),
        "pos_frac": {"min": float(pos_min_frac), "max": float(pos_max_frac)},
        "good_gate": {"mode": str(good_gate_mode), "l1_score": int(good_gate_l1_score), "l2_score": int(good_gate_l2_score), "entry_ctx_counts": entry_ctx_counts},
        "pyramiding": {
            "enabled": bool(pyramiding_enabled),
            "pos_cap_total": float(pos_cap_total),
            "addon_min_bars": int(addon_min_bars),
            "addon_hold_bars": int(addon_hold_bars),
            "addon_max_l1": int(addon_max_l1),
            "addon_max_l2": int(addon_max_l2),
            "addon_max_l3": int(addon_max_l3),
            "addon_frac_l1": float(addon_frac_l1),
            "addon_frac_l2": float(addon_frac_l2),
            "addon_frac_l3": float(addon_frac_l3),
            "addon_counts": addon_counts,
        },
        "equity_start": 1.0,
        "equity_end": round(float(equity), 4),
        "return_pct": round((float(equity) - 1.0) * 100.0, 2),
        "mdd_pct": round(float(mdd_pct), 4),
        "mdd_ok": bool(mdd_ok),
        "downtrend_mode": str(downtrend_mode),
        "entries": total_entries,
        "legs": total_legs,
        "entries_per_month": round(float(entries_per_month), 2),
        "legs_per_month": round(float(legs_per_month), 2),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "avg_r": round(avg_r, 4),
        "avg_r_pos": round(float(avg_r_pos), 4),
        "cost_bps": cost_bps,
        "min_rr": min_rr,
        "decision_tf": "minute60",
        "mtf_weights": weights,
        "fib": {"lookback": fib_lookback, "min": fib_min, "max": fib_max, "swing_confirm": fib_swing_confirm},
        "d1_slope_lookback": d1_slope_lookback,
        "tp": {"tp1_r": tp1_r, "tp2_r": tp2_r, "rr_target_atr_mult": rr_target_atr_mult, "tp1_ratio": tp1_ratio, "be_offset_bps": be_offset_bps, "be_move_mode": be_move_mode, "be_strong_stop_r": be_strong_stop_r, "adx_min": adx_min},
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
    ap.add_argument("--good-gate-mode", type=str, default="none", help="none|l1|l2|l3")
    ap.add_argument("--good-gate-l1-score", type=int, default=55)
    ap.add_argument("--good-gate-l2-score", type=int, default=65)
    ap.add_argument("--good-gate-l3-score", type=int, default=78)

    # Pyramiding(scale-in) knobs
    ap.add_argument("--pyramiding", action="store_true", help="Enable pyramiding(scale-in)")
    ap.add_argument("--pos-cap-total", type=float, default=0.90)
    ap.add_argument("--addon-min-bars", type=int, default=1)
    ap.add_argument("--addon-hold-bars", type=int, default=2)
    ap.add_argument("--addon-max-l1", type=int, default=1)
    ap.add_argument("--addon-max-l2", type=int, default=2)
    ap.add_argument("--addon-max-l3", type=int, default=1)
    ap.add_argument("--addon-frac-l1", type=float, default=0.07)
    ap.add_argument("--addon-frac-l2", type=float, default=0.10)
    ap.add_argument("--addon-frac-l3", type=float, default=0.15)
    ap.add_argument("--addon-stop-lift-r", type=float, default=-0.2)
    ap.add_argument("--risk-per-trade", type=float, default=0.015)
    ap.add_argument("--mdd-limit-pct", type=float, default=0.30)
    ap.add_argument("--downtrend-mode", type=str, default="or", help="h4|d1|or|off")
    ap.add_argument("--fib-lookback", type=int, default=120)
    ap.add_argument("--fib-min", type=float, default=0.382)
    ap.add_argument("--fib-max", type=float, default=0.618)
    ap.add_argument("--fib-swing-confirm", action="store_true", help="Require swing confirmation (low before high) in fib window")
    ap.add_argument("--no-fib-swing-confirm", action="store_true", help="Disable fib swing confirmation")
    ap.set_defaults(fib_swing_confirm=False)
    ap.add_argument("--d1-slope-lookback", type=int, default=5)
    ap.add_argument("--tp1-r", type=float, default=0.8)
    ap.add_argument("--tp2-r", type=float, default=1.6)
    ap.add_argument("--rr-target-atr-mult", type=float, default=1.0)
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
        good_gate_mode=str(args.good_gate_mode),
        good_gate_l1_score=int(args.good_gate_l1_score),
        good_gate_l2_score=int(args.good_gate_l2_score),
        good_gate_l3_score=int(args.good_gate_l3_score),
        pyramiding_enabled=bool(args.pyramiding),
        pos_cap_total=float(args.pos_cap_total),
        addon_min_bars=int(args.addon_min_bars),
        addon_hold_bars=int(args.addon_hold_bars),
        addon_max_l1=int(args.addon_max_l1),
        addon_max_l2=int(args.addon_max_l2),
        addon_max_l3=int(args.addon_max_l3),
        addon_frac_l1=float(args.addon_frac_l1),
        addon_frac_l2=float(args.addon_frac_l2),
        addon_frac_l3=float(args.addon_frac_l3),
        addon_stop_lift_r=float(args.addon_stop_lift_r),
        risk_per_trade=float(args.risk_per_trade),
        mdd_limit_pct=float(args.mdd_limit_pct),
        downtrend_mode=str(args.downtrend_mode),
        fib_lookback=int(args.fib_lookback),
        fib_min=float(args.fib_min),
        fib_max=float(args.fib_max),
        fib_swing_confirm=(False if bool(args.no_fib_swing_confirm) else (True if bool(args.fib_swing_confirm) else False)),
        d1_slope_lookback=int(args.d1_slope_lookback),
        tp1_r=float(args.tp1_r),
        tp2_r=float(args.tp2_r),
        rr_target_atr_mult=float(args.rr_target_atr_mult),
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
