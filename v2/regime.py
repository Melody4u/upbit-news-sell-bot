from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeState:
    in_box: bool
    riskoff: bool
    uptrend: bool
    d1_adx: float
    d1_close: float
    d1_ma200: float


def d1_ma(series: pd.Series, period: int) -> float:
    ma = series.rolling(period).mean()
    if len(ma) == 0 or pd.isna(ma.iloc[-1]):
        return float("nan")
    return float(ma.iloc[-1])


def calc_regime(dfd: pd.DataFrame, ts: pd.Timestamp, box_adx_max: float = 30.0) -> RegimeState:
    """Compute regime on D1 at time ts.

    Definitions (v2):
    - box: D1 ADX < box_adx_max
    - riskoff: D1 close < D1 MA200
    - uptrend: (not box) and (not riskoff) and EMA20>EMA50 (optional later)

    NOTE: uptrend is deliberately simple for v2 start.
    """
    sub = dfd[dfd.index <= ts] if isinstance(dfd.index, pd.DatetimeIndex) else pd.DataFrame()
    if sub is None or sub.empty:
        return RegimeState(False, False, False, 0.0, float("nan"), float("nan"))

    close = float(sub["close"].iloc[-1]) if pd.notna(sub["close"].iloc[-1]) else float("nan")
    adx = float(sub["adx"].iloc[-1]) if ("adx" in sub.columns and pd.notna(sub["adx"].iloc[-1])) else 0.0
    ma200 = d1_ma(sub["close"].astype(float), 200)

    in_box = bool(adx > 0 and adx < float(box_adx_max))
    riskoff = bool(np.isfinite(close) and np.isfinite(ma200) and close < ma200)
    uptrend = bool((not in_box) and (not riskoff))

    return RegimeState(in_box=in_box, riskoff=riskoff, uptrend=uptrend, d1_adx=float(adx), d1_close=float(close), d1_ma200=float(ma200))
