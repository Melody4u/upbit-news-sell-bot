import os
import time
import json
import logging
import hashlib
from typing import List, Tuple, Dict

import requests
import pyupbit
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from tradingbot.config import env_bool, get_env_float, get_env_int, parse_float_list_csv
from tradingbot.log_setup import setup_logger
from tradingbot.logging.trade_log import append_trade_log
from tradingbot.state.runtime_state import load_runtime_state, save_runtime_state
from tradingbot.data.exchange_client import (
    get_account_state,
    get_krw_balance,
    get_spread_bps,
    get_order_state,
)
from tradingbot.execution.broker import place_order_with_retry, verify_position_change
from tradingbot.notifications import notify_event
from tradingbot.risk.news import NewsState, fetch_negative_news_score
from tradingbot.risk.market_risk import load_market_risk, resolve_risk_mode, apply_risk_mode


def _atomic_write_text(path: str, text: str) -> None:
    """Best-effort atomic write (write temp then replace)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
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


# (moved) exchange client + config helpers -> tradingbot.data.exchange_client / tradingbot.config


def build_stop_feedback(reasons: List[str]) -> str:
    text = " | ".join(reasons)
    tips = []
    if "entry_candle_stop" in text:
        tips.append("진입 직후 손절 발생: PROBE_ENTRY_RATIO를 낮추거나 BREAKOUT_BODY_ATR_MULT를 높여 더 강한 돌파에서만 진입")
    if "ma22_exit" in text:
        tips.append("MA22 이탈 청산 빈도 높음: MA_EXIT_PERIOD를 22→24~30으로 완화해 과민 반응 여부 점검")
    if "sideways_filter" in text:
        tips.append("횡보 구간 거래 과다 가능성: SIDEWAYS_CROSS_THRESHOLD를 상향하거나 BOX 돌파 확인 후 진입")
    if "negative_news" in text:
        tips.append("악재 뉴스 민감도 조정: NEWS_INTERVAL_SECONDS/임계치 조정으로 뉴스 필터 과민 반응 점검")
    if not tips:
        tips.append("최근 30~50개 트레이드 로그를 모아 공통 손절 원인 확인 후 파라미터 1개씩만 조정")
    return " / ".join(tips)


# (moved) append_trade_log -> tradingbot.logging.trade_log
# (moved) market risk helpers -> tradingbot.risk.market_risk


# (moved) runtime state IO -> tradingbot.state.runtime_state


def is_stop_loss_event(reasons: List[str]) -> bool:
    text = " | ".join(reasons)
    keys = ["entry_candle_stop", "loss_cut", "atr_trailing_break", "donchian_break"]
    return any(k in text for k in keys)


# (moved) place_order_with_retry -> tradingbot.execution.broker


def make_signal_hash(side: str, reasons: List[str], score: int, market: str) -> str:
    payload = f"{side}|{market}|{score}|{'|'.join(sorted(reasons))}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# (moved) verify_position_change -> tradingbot.execution.broker


# (moved) notify_event -> tradingbot.notifications


# (moved) get_order_state -> tradingbot.data.exchange_client


def evaluate_signals(df: pd.DataFrame, avg_buy_price: float, cfg: Dict[str, float], mtf_trend_ok: bool = True) -> Tuple[int, List[str], Dict[str, float]]:
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    current = float(close.iloc[-1])

    ema_fast = close.ewm(span=int(cfg["EMA_FAST"]), adjust=False).mean()
    ema_slow = close.ewm(span=int(cfg["EMA_SLOW"]), adjust=False).mean()
    adx = compute_adx(df, int(cfg["ADX_PERIOD"]))
    donchian_low = low.rolling(int(cfg["DONCHIAN_PERIOD"])).min()
    atr = compute_atr(df, int(cfg["ATR_PERIOD"]))
    rsi = compute_rsi(close, int(cfg["RSI_PERIOD"]))
    vol_ma = vol.rolling(int(cfg["VOL_PERIOD"])).mean()

    ma20 = close.rolling(int(cfg["MA_SHORT"])).mean()
    ma200 = close.rolling(int(cfg["MA_LONG"])).mean()
    vwma100 = (close * vol).rolling(int(cfg["VWMA_PERIOD"])).sum() / vol.rolling(int(cfg["VWMA_PERIOD"])).sum()

    score = 0
    reasons: List[str] = []
    buy_score = 0
    buy_reasons: List[str] = []

    # 200MA 湲곗슱湲?諛⑺뼢 寃뚯씠??
    slope_lookback = int(cfg["MA_SLOPE_LOOKBACK"])
    ma200_slope = 0.0
    if len(ma200.dropna()) > slope_lookback:
        ma200_slope = float(ma200.iloc[-1] - ma200.iloc[-1 - slope_lookback])
    trend_down = ma200_slope < 0
    trend_up = ma200_slope > 0

    ma20_lost = bool(pd.notna(ma20.iloc[-1]) and current < ma20.iloc[-1])

    # ?ㅽ댁쫰: gap 異뺤냼 + ?꾧퀎移??댄븯
    ma_gap_series = (ma20 - ma200).abs() / close
    ma_gap_ratio = float(ma_gap_series.iloc[-1]) if pd.notna(ma_gap_series.iloc[-1]) else 0.0
    squeeze_lookback = int(cfg["SQUEEZE_LOOKBACK"])
    gap_tail = ma_gap_series.tail(squeeze_lookback).dropna()
    squeeze_shrinking = bool(len(gap_tail) >= 3 and gap_tail.is_monotonic_decreasing)
    squeeze = bool(ma_gap_ratio <= cfg["SQUEEZE_GAP_PCT"] and squeeze_shrinking)

    # ?λ?/鍮꾩젒珥??뚰뙆 罹붾뱾
    body = (close - open_).abs()
    last_body = float(body.iloc[-1])
    last_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    strong_body = bool(last_atr > 0 and last_body >= cfg["BREAKOUT_BODY_ATR_MULT"] * last_atr)
    no_touch_down = bool(pd.notna(ma20.iloc[-1]) and high.iloc[-1] < ma20.iloc[-1])
    no_touch_up = bool(pd.notna(ma20.iloc[-1]) and low.iloc[-1] > ma20.iloc[-1])

    confirmed_breakdown = strong_body and no_touch_down
    confirmed_breakout = strong_body and no_touch_up

    # ?ㅽ댁쫰 ?뚮젅?? 異붿꽭 諛⑺뼢 ?뚰뙆留?梨꾪깮
    if squeeze and trend_down and confirmed_breakdown:
        score += 3
        reasons.append("squeeze_breakdown_confirmed")
    if squeeze and trend_up and confirmed_breakout:
        score += 2
        reasons.append("squeeze_breakout_confirmed")
        buy_score += 3
        buy_reasons.append("squeeze_breakout_confirmed")

    # ?ㅽ댁쫰 ?놁씠 ?덈Т 鍮좊Ⅸ ?덉텧? 臾댁떆(??텛???섏씠??
    if (not squeeze) and (confirmed_breakout or confirmed_breakdown):
        reasons.append("fast_break_without_squeeze")
        score = max(0, score - 1)

    # ?섎씫?μ뿉?쒕뒗 濡깆꽦 ?좏샇 ?쏀솕, ?곸듅?μ뿉?쒕뒗 ?륁꽦 ?좏샇 ?쏀솕
    if trend_down:
        score += 1
        reasons.append("ma200_downtrend_gate")

    if trend_up and pd.notna(ma20.iloc[-1]) and current > ma20.iloc[-1] and pd.notna(ma200.iloc[-1]) and ma20.iloc[-1] > ma200.iloc[-1]:
        buy_score += 2
        buy_reasons.append("trend_up_above_ma20_ma200")

    if pd.notna(vwma100.iloc[-1]) and current >= float(vwma100.iloc[-1]):
        buy_score += 1
        buy_reasons.append("price_above_vwma100")
    elif pd.notna(vwma100.iloc[-1]) and current < float(vwma100.iloc[-1]):
        score += 1
        reasons.append("price_below_vwma100")

    # MTF gate (buy score impact)
    buy_score_raw = int(buy_score)
    buy_reasons_raw = list(buy_reasons)
    if mtf_trend_ok:
        buy_score += 1
        buy_reasons.append("mtf_trend_ok")
    else:
        buy_score = max(0, buy_score - 2)
        buy_reasons.append("mtf_trend_block")

    cross_series = (ma20 > ma200).astype("float").diff().abs()
    recent_crosses = int(cross_series.tail(int(cfg["SIDEWAYS_LOOKBACK"])).fillna(0).sum())
    sideways = recent_crosses >= int(cfg["SIDEWAYS_CROSS_THRESHOLD"])
    if sideways:
        score = max(0, score - 1)
        buy_score = max(0, buy_score - 1)
        reasons.append("sideways_filter")
        buy_reasons.append("sideways_filter")

    # 湲곗〈 硫?곗떆洹몃꼸(蹂댁“)
    trend_break = bool(ema_fast.iloc[-1] < ema_slow.iloc[-1] and adx.iloc[-1] >= cfg["ADX_MIN"])
    if trend_break:
        score += 2
        reasons.append("trend_break(ema_fast<ema_slow & adx_high)")

    trend_up_confirm = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1] and adx.iloc[-1] >= cfg["ADX_MIN"])
    if trend_up_confirm and trend_up:
        buy_score += 1
        buy_reasons.append("trend_up_confirm(ema_fast>ema_slow & adx_high)")

    donchian_break = bool(current < donchian_low.iloc[-1]) if pd.notna(donchian_low.iloc[-1]) else False
    if donchian_break:
        score += 2
        reasons.append("donchian_break")

    recent_peak = float(high.tail(int(cfg["DONCHIAN_PERIOD"])).max())
    atr_stop = recent_peak - (cfg["ATR_MULT"] * float(atr.iloc[-1]))
    atr_break = bool(current < atr_stop)
    if atr_break:
        score += 1
        reasons.append("atr_trailing_break")

    rsi_rollover = bool(rsi.iloc[-2] > cfg["RSI_OVERBOUGHT"] and rsi.iloc[-1] < rsi.iloc[-2])
    vol_spike = bool(vol.iloc[-1] > (vol_ma.iloc[-1] * cfg["VOL_SPIKE_MULT"])) if pd.notna(vol_ma.iloc[-1]) else False
    if rsi_rollover and vol_spike:
        score += 1
        reasons.append("rsi_rollover_with_volume_spike")

    rsi_rebound = bool(rsi.iloc[-2] < cfg["RSI_OVERSOLD"] and rsi.iloc[-1] > rsi.iloc[-2])
    if rsi_rebound and trend_up:
        buy_score += 1
        buy_reasons.append("rsi_rebound")

    loss_cut = avg_buy_price > 0 and current < avg_buy_price * (1 - cfg["LOSS_CUT_PCT"])
    if loss_cut:
        score += 2
        buy_score = max(0, buy_score - 1)
        reasons.append("loss_cut")
        buy_reasons.append("loss_cut")

    metrics = {
        "price": current,
        "ema_fast": float(ema_fast.iloc[-1]),
        "ema_slow": float(ema_slow.iloc[-1]),
        "ma20": float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else 0.0,
        "ma200": float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else 0.0,
        "vwma100": float(vwma100.iloc[-1]) if pd.notna(vwma100.iloc[-1]) else 0.0,
        "ma200_slope": ma200_slope,
        "ma_gap_pct": ma_gap_ratio * 100,
        "recent_crosses": recent_crosses,
        "squeeze": int(squeeze),
        "confirmed_breakdown": int(confirmed_breakdown),
        "confirmed_breakout": int(confirmed_breakout),
        "adx": float(adx.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "atr_stop": float(atr_stop),
        "donchian_low": float(donchian_low.iloc[-1]) if pd.notna(donchian_low.iloc[-1]) else 0.0,
        "buy_score_raw": buy_score_raw,
        "buy_reasons_raw": buy_reasons_raw,
        "buy_score": buy_score,
        "buy_reasons": buy_reasons,
    }

    return score, reasons, metrics


def run():
    load_dotenv()
    setup_logger()

    access = os.getenv("UPBIT_ACCESS_KEY", "")
    secret = os.getenv("UPBIT_SECRET_KEY", "")
    market = os.getenv("MARKET", "KRW-BTC")

    dry_run = env_bool("DRY_RUN", True)
    sell_ratio = get_env_float("SELL_RATIO", 0.25)
    buy_ratio = get_env_float("BUY_RATIO", 0.25)
    min_sell_krw = get_env_float("MIN_SELL_KRW", 5000)
    min_buy_krw = get_env_float("MIN_BUY_KRW", 5000)
    check_seconds = get_env_int("CHECK_SECONDS", 30)

    brave_key = os.getenv("BRAVE_API_KEY", "")
    news_query = os.getenv("NEWS_QUERY", "bitcoin OR btc")
    news_country = os.getenv("NEWS_COUNTRY", "KR")
    news_lang = os.getenv("NEWS_LANG", "ko")
    news_result_count = get_env_int("NEWS_RESULT_COUNT", 5)
    news_interval_seconds = get_env_int("NEWS_INTERVAL_SECONDS", 3600)
    negative_news_threshold = get_env_int("NEGATIVE_NEWS_THRESHOLD", 2)

    signal_score_threshold = get_env_int("SIGNAL_SCORE_THRESHOLD", 3)
    buy_score_threshold = get_env_int("BUY_SCORE_THRESHOLD", 3)

    entry_mode = os.getenv("ENTRY_MODE", "confirm_breakout")
    breakout_gate_mode = os.getenv("BREAKOUT_GATE_MODE", "and").lower()  # and|or
    trade_cooldown_seconds = get_env_int("TRADE_COOLDOWN_SECONDS", 120)
    emergency_sell_bypass_idempotency = env_bool("EMERGENCY_SELL_BYPASS_IDEMPOTENCY", True)
    probe_entry_ratio = get_env_float("PROBE_ENTRY_RATIO", 0.15)
    breakout_lookback = get_env_int("BREAKOUT_LOOKBACK", 20)
    mtf_interval = os.getenv("MTF_TREND_INTERVAL", "minute30")
    mtf_interval_2 = os.getenv("MTF_TREND_INTERVAL_2", "minute240")
    mtf_lookback = get_env_int("MTF_TREND_LOOKBACK", 120)

    # --- MTF score sizing (Phase A)
    enable_mtf_score_sizing = env_bool("ENABLE_MTF_SCORE_SIZING", True)
    minute_consensus_intervals = [s.strip() for s in os.getenv("MINUTE_CONSENSUS_INTERVALS", "minute3,minute5,minute15").split(",") if s.strip()]
    hour_score_weights = {
        "minute30": get_env_int("HOUR_MTF_WEIGHT_M30", 15),
        "minute60": get_env_int("HOUR_MTF_WEIGHT_H1", 20),
        "minute240": get_env_int("HOUR_MTF_WEIGHT_H4", 30),
        "day": get_env_int("HOUR_MTF_WEIGHT_D1", 20),
        "week": get_env_int("HOUR_MTF_WEIGHT_W1", 15),
    }
    # score -> stage sizing (equity fraction)
    mtf_stage_thresholds = {
        "scout": get_env_int("MTF_STAGE_SCOUT_MIN", 30),
        "light": get_env_int("MTF_STAGE_LIGHT_MIN", 50),
        "medium": get_env_int("MTF_STAGE_MEDIUM_MIN", 65),
        "heavy": get_env_int("MTF_STAGE_HEAVY_MIN", 80),
    }
    mtf_stage_pcts = {
        "scout": get_env_float("MTF_STAGE_SCOUT_PCT", 0.01),
        "light": get_env_float("MTF_STAGE_LIGHT_PCT", 0.07),
        "medium": get_env_float("MTF_STAGE_MEDIUM_PCT", 0.15),
        "heavy": get_env_float("MTF_STAGE_HEAVY_PCT", 0.25),
    }

    # stops
    entry_stop_atr_mult = get_env_float("ENTRY_STOP_ATR_MULT", 2.5)
    hard_stop_pct = get_env_float("HARD_STOP_PCT", 0.10)
    hard_stop_recovery_window_sec = get_env_int("HARD_STOP_RECOVERY_WINDOW_SEC", 86400)

    # regime + pullback (Phase B-lite)
    downtrend_block_enabled = env_bool("DOWNTREND_BLOCK_ENABLED", True)
    fib_pullback_enabled = env_bool("FIB_PULLBACK_ENABLED", False)
    fib_lookback = get_env_int("FIB_LOOKBACK", 120)
    fib_min = get_env_float("FIB_MIN", 0.382)
    fib_max = get_env_float("FIB_MAX", 0.618)
    fib_swing_confirm = env_bool("FIB_SWING_CONFIRM", True)

    # volatility spike block (Phase B-lite)
    vol_spike_block_enabled = env_bool("VOL_SPIKE_BLOCK_ENABLED", False)
    vol_spike_block_atr_mult = get_env_float("VOL_SPIKE_BLOCK_ATR_MULT", 2.0)

    rr_min = get_env_float("MIN_RR", 2.5)
    rr_target_atr_mult = get_env_float("RR_TARGET_ATR_MULT", 2.0)
    spread_bps_max = get_env_float("SPREAD_BPS_MAX", 12)
    post_entry_cooldown_bars = get_env_int("POST_ENTRY_COOLDOWN_BARS", 3)
    partial_tp_levels = parse_float_list_csv(os.getenv("PARTIAL_TP_LEVELS", "0.7,1.4"), [0.7, 1.4])
    partial_tp_ratios = parse_float_list_csv(os.getenv("PARTIAL_TP_RATIOS", "0.6,0.4"), [0.6, 0.4])
    move_stop_to_be_after_tp1 = env_bool("MOVE_STOP_TO_BE_AFTER_TP1", True)
    be_offset_bps = get_env_float("BE_OFFSET_BPS", 8.0)
    be_move_mode = os.getenv("BE_MOVE_MODE", "hybrid").lower()  # always|weak_only|hybrid
    be_strong_stop_r = get_env_float("BE_STRONG_STOP_R", -0.2)  # in strong context, lift stop to entry + R* risk (negative = below entry)

    # Early fail cut (high win-rate / cut losers faster)
    early_fail_cut_enabled = env_bool("EARLY_FAIL_CUT_ENABLED", True)
    early_fail_levels_r = parse_float_list_csv(os.getenv("EARLY_FAIL_LEVELS_R", "0.6,0.9,1.2,1.6,2.0"), [0.6, 0.9, 1.2, 1.6, 2.0])
    early_fail_cut_ratios = parse_float_list_csv(os.getenv("EARLY_FAIL_CUT_RATIOS", "0.1,0.3,0.5,0.7,1.0"), [0.1, 0.3, 0.5, 0.7, 1.0])
    early_fail_max_cut_ratio = get_env_float("EARLY_FAIL_MAX_CUT_RATIO", 1.0)
    early_fail_mode = os.getenv("EARLY_FAIL_MODE", "weak_trend_only").lower()  # weak_trend_only|always|hybrid
    early_fail_strong_levels_r = parse_float_list_csv(os.getenv("EARLY_FAIL_STRONG_LEVELS_R", "1.6,2.0"), [1.6, 2.0])
    early_fail_strong_cut_ratios = parse_float_list_csv(os.getenv("EARLY_FAIL_STRONG_CUT_RATIOS", "0.5,1.0"), [0.5, 1.0])
    if len(partial_tp_levels) != len(partial_tp_ratios):
        logging.warning("PARTIAL_TP length mismatch | levels=%s ratios=%s -> truncating to min length", len(partial_tp_levels), len(partial_tp_ratios))
    n_tp = min(len(partial_tp_levels), len(partial_tp_ratios))
    partial_tp_levels = partial_tp_levels[:n_tp]
    partial_tp_ratios = partial_tp_ratios[:n_tp]
    sum_ratios = sum(max(0.0, r) for r in partial_tp_ratios)
    if sum_ratios > 1.0:
        scale = 1.0 / sum_ratios
        partial_tp_ratios = [max(0.0, r) * scale for r in partial_tp_ratios]
        logging.warning("PARTIAL_TP ratio sum > 1.0 | normalized with scale=%.4f", scale)
    atr_regime_min_pct = get_env_float("ATR_REGIME_MIN_PCT", 0.6)
    atr_regime_max_pct = get_env_float("ATR_REGIME_MAX_PCT", 4.0)
    trailing_stop_mode = os.getenv("TRAILING_STOP_MODE", "atr").lower()
    trailing_atr_mult = get_env_float("TRAILING_ATR_MULT", 2.5)
    trailing_apply_to_remainder_only = env_bool("TRAILING_APPLY_TO_REMAINDER_ONLY", True)
    stop_by_entry_candle = env_bool("STOP_BY_ENTRY_CANDLE", True)
    take_profit_by_ma22 = env_bool("TAKE_PROFIT_BY_MA22", True)
    ma_exit_period = get_env_int("MA_EXIT_PERIOD", 22)
    box_lookback = get_env_int("BOX_LOOKBACK", 40)
    box_buffer_pct = get_env_float("BOX_BUFFER_PCT", 0.003)
    enable_retest_addon = env_bool("ENABLE_RETEST_ADDON", True)
    addon_ratio = get_env_float("ADDON_RATIO", 0.1)
    trade_log_path = os.getenv("TRADE_LOG_PATH", "logs/trade_journal.jsonl")
    runtime_state_path = os.getenv("RUNTIME_STATE_PATH", "logs/runtime_state.json")
    market_risk_path = os.getenv("MARKET_RISK_PATH", "logs/market_risk.json")
    max_drawdown_pct = get_env_float("MAX_DRAWDOWN_PCT", 0.10)
    max_yearly_stop_count = get_env_int("MAX_YEARLY_STOP_COUNT", 3)
    max_daily_loss_pct = get_env_float("MAX_DAILY_LOSS_PCT", 0.03)
    max_consecutive_losses = get_env_int("MAX_CONSECUTIVE_LOSSES", 0)
    max_trades_per_day = get_env_int("MAX_TRADES_PER_DAY", 20)
    max_position_krw = get_env_float("MAX_POSITION_KRW", 0)
    max_position_fraction = get_env_float("MAX_POSITION_FRACTION", 0.0)
    max_slippage_bps = get_env_float("MAX_SLIPPAGE_BPS", 30)
    risk_per_trade = get_env_float("RISK_PER_TRADE", 0.0)
    alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    alert_events = set(x.strip() for x in os.getenv("ALERT_EVENTS", "order_sent,filled,rejected,halted,resume").split(",") if x.strip())
    pending_order_timeout_sec = get_env_int("PENDING_ORDER_TIMEOUT_SEC", 180)

    # Ensure log paths exist. We also "touch" the trade journal so monitors
    # don't error before the first trade is written.
    for _p in [trade_log_path, runtime_state_path, market_risk_path]:
        _d = os.path.dirname(_p)
        if _d:
            os.makedirs(_d, exist_ok=True)
    try:
        open(trade_log_path, "a", encoding="utf-8").close()
    except Exception as e:
        logging.warning("failed to init trade journal | path=%s err=%s", trade_log_path, e)

    cfg = {
        "EMA_FAST": get_env_int("EMA_FAST", 20),
        "EMA_SLOW": get_env_int("EMA_SLOW", 50),
        "ADX_PERIOD": get_env_int("ADX_PERIOD", 14),
        "ADX_MIN": get_env_float("ADX_MIN", 20),
        "DONCHIAN_PERIOD": get_env_int("DONCHIAN_PERIOD", 20),
        "ATR_PERIOD": get_env_int("ATR_PERIOD", 14),
        "ATR_MULT": get_env_float("ATR_MULT", 2.5),
        "RSI_PERIOD": get_env_int("RSI_PERIOD", 14),
        "RSI_OVERBOUGHT": get_env_float("RSI_OVERBOUGHT", 70),
        "RSI_OVERSOLD": get_env_float("RSI_OVERSOLD", 30),
        "VOL_PERIOD": get_env_int("VOL_PERIOD", 20),
        "VOL_SPIKE_MULT": get_env_float("VOL_SPIKE_MULT", 1.8),
        "VWMA_PERIOD": get_env_int("VWMA_PERIOD", 100),
        "LOSS_CUT_PCT": get_env_float("LOSS_CUT_PCT", 0.02),
        "MA_SHORT": get_env_int("MA_SHORT", 20),
        "MA_LONG": get_env_int("MA_LONG", 200),
        "MA_SLOPE_LOOKBACK": get_env_int("MA_SLOPE_LOOKBACK", 5),
        "SQUEEZE_GAP_PCT": get_env_float("SQUEEZE_GAP_PCT", 0.01),
        "SQUEEZE_LOOKBACK": get_env_int("SQUEEZE_LOOKBACK", 5),
        "BREAKOUT_BODY_ATR_MULT": get_env_float("BREAKOUT_BODY_ATR_MULT", 1.2),
        "SIDEWAYS_LOOKBACK": get_env_int("SIDEWAYS_LOOKBACK", 30),
        "SIDEWAYS_CROSS_THRESHOLD": get_env_int("SIDEWAYS_CROSS_THRESHOLD", 3),
        "RISK_AGGRESSIVE_MAX": get_env_float("RISK_AGGRESSIVE_MAX", 30),
        "RISK_CONSERVATIVE_MIN": get_env_float("RISK_CONSERVATIVE_MIN", 61),
        "AGG_SIGNAL_SCORE_THRESHOLD": get_env_int("AGG_SIGNAL_SCORE_THRESHOLD", 2),
        "AGG_BUY_SCORE_THRESHOLD": get_env_int("AGG_BUY_SCORE_THRESHOLD", 2),
        "AGG_SELL_RATIO": get_env_float("AGG_SELL_RATIO", 0.20),
        "AGG_BUY_RATIO": get_env_float("AGG_BUY_RATIO", 0.30),
        "CONS_SIGNAL_SCORE_THRESHOLD": get_env_int("CONS_SIGNAL_SCORE_THRESHOLD", 2),
        "CONS_BUY_SCORE_THRESHOLD": get_env_int("CONS_BUY_SCORE_THRESHOLD", 4),
        "CONS_SELL_RATIO": get_env_float("CONS_SELL_RATIO", 0.35),
        "CONS_BUY_RATIO": get_env_float("CONS_BUY_RATIO", 0.10),
    }

    if not access or not secret:
        raise RuntimeError("UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY를 .env에 입력하세요.")

    upbit = pyupbit.Upbit(access, secret)
    news_state = NewsState()
    active_box_high = None
    addon_done = False
    runtime_state = load_runtime_state(runtime_state_path)
    last_action_ts = float(runtime_state.get("last_action_ts", 0.0) or 0.0)
    last_signal_hash = str(runtime_state.get("last_signal_hash", "") or "")
    last_block_hash = str(runtime_state.get("last_block_hash", "") or "")
    last_block_ts = float(runtime_state.get("last_block_ts", 0.0) or 0.0)
    pending_order = runtime_state.get("pending_order")
    last_buy_stop_price = runtime_state.get("last_buy_stop_price")
    if last_buy_stop_price is not None:
        try:
            last_buy_stop_price = float(last_buy_stop_price)
        except Exception:
            last_buy_stop_price = None
    last_buy_ts = float(runtime_state.get("last_buy_ts", 0.0) or 0.0)
    entry_stop_price = runtime_state.get("entry_stop_price")
    if entry_stop_price is not None:
        try:
            entry_stop_price = float(entry_stop_price)
        except Exception:
            entry_stop_price = None
    partial_tp_done = set(str(x) for x in (runtime_state.get("partial_tp_done") or []))
    early_fail_done = set(str(x) for x in (runtime_state.get("early_fail_done") or []))
    trailing_peak_price = float(runtime_state.get("trailing_peak_price", 0.0) or 0.0)
    prev_halted = bool(runtime_state.get("halted", False))

    # PID/heartbeat for watchdog (avoid fragile CommandLine matching)
    pid_path = os.getenv("PID_PATH", "logs/bot.pid")
    heartbeat_path = os.getenv("HEARTBEAT_PATH", "logs/bot.heartbeat")
    try:
        _atomic_write_text(pid_path, str(os.getpid()))
        _atomic_write_text(heartbeat_path, str(time.time()))
    except Exception as e:
        logging.warning("failed to write pid/heartbeat | err=%s", e)

    logging.info("bot started | market=%s dry_run=%s entry_mode=%s", market, dry_run, entry_mode)
    loop_error_count = 0

    while True:
        try:
            now = time.time()
            # heartbeat: updated at top of loop so watchdog can detect stalls
            try:
                _atomic_write_text(heartbeat_path, str(now))
            except Exception:
                pass

            if brave_key and (now - news_state.last_checked >= news_interval_seconds):
                score, headlines = fetch_negative_news_score(
                    brave_key,
                    news_query,
                    news_country,
                    news_lang,
                    news_result_count,
                )
                news_state.last_checked = now
                news_state.negative_score = score
                news_state.headlines = headlines
                logging.info("news refreshed | negative_score=%s", score)
                for h in headlines[:3]:
                    logging.info("news: %s", h[:180])

            coin_balance, avg_buy_price, current_price = get_account_state(upbit, market)
            krw_balance = get_krw_balance(upbit)
            coin_value_krw = coin_balance * current_price
            equity = krw_balance + coin_value_krw

            # 연도/일자 상태 관리
            now_year = time.localtime().tm_year
            day_key = time.strftime("%Y-%m-%d")
            if runtime_state.get("year") != now_year:
                runtime_state["year"] = now_year
                runtime_state["yearly_stop_count"] = 0
            if runtime_state.get("day_key") != day_key:
                runtime_state["day_key"] = day_key
                runtime_state["day_start_equity"] = equity
                runtime_state["trades_today"] = 0
                runtime_state["consecutive_losses_today"] = 0

            # 재시작 내구성: 대기 주문 상태 조회
            if pending_order and isinstance(pending_order, dict) and pending_order.get("uuid"):
                st, order_obj = get_order_state(upbit, pending_order.get("uuid"))
                age_sec = max(0, int(time.time() - float(pending_order.get("created_ts", time.time()))))
                if st in ("done", "cancel"):
                    notify_event(alert_webhook_url, alert_events, "filled" if st == "done" else "rejected", f"pending_order_resolved side={pending_order.get('side')} state={st}")
                    pending_order = None
                elif st == "error":
                    logging.warning("pending order check error: %s", order_obj)
                else:
                    logging.info("pending order wait | side=%s state=%s age=%ss", pending_order.get("side"), st, age_sec)
                    if age_sec >= pending_order_timeout_sec and not pending_order.get("timeout_notified", False):
                        msg = f"pending_order_timeout side={pending_order.get('side')} age={age_sec}s state={st}"
                        notify_event(alert_webhook_url, alert_events, "rejected", msg)
                        append_trade_log(trade_log_path, {
                            "ts": int(time.time()),
                            "side": "ops_timeout",
                            "market": market,
                            "price": current_price,
                            "score": 0,
                            "reasons": [msg],
                            "dry_run": dry_run,
                        })
                        pending_order["timeout_notified"] = True

            # 자본곡선/킬스위치 관리
            runtime_state["equity_peak"] = max(float(runtime_state.get("equity_peak", 0.0)), equity)
            peak = max(float(runtime_state.get("equity_peak", 0.0)), 1.0)
            drawdown_pct = (peak - equity) / peak
            day_start_equity = float(runtime_state.get("day_start_equity", 0.0) or 0.0)
            daily_loss_pct = ((day_start_equity - equity) / day_start_equity) if day_start_equity > 0 else 0.0

            if drawdown_pct >= max_drawdown_pct:
                runtime_state["halted"] = True
                runtime_state["halt_reason"] = f"max_drawdown_reached({drawdown_pct:.2%})"
            if daily_loss_pct >= max_daily_loss_pct:
                runtime_state["halted"] = True
                runtime_state["halt_reason"] = f"max_daily_loss_reached({daily_loss_pct:.2%})"
            if int(runtime_state.get("yearly_stop_count", 0)) >= max_yearly_stop_count:
                runtime_state["halted"] = True
                runtime_state["halt_reason"] = f"yearly_stop_count_reached({runtime_state.get('yearly_stop_count', 0)})"
            if max_consecutive_losses > 0 and int(runtime_state.get("consecutive_losses_today", 0)) >= max_consecutive_losses:
                runtime_state["halted"] = True
                runtime_state["halt_reason"] = f"max_consecutive_losses_reached({runtime_state.get('consecutive_losses_today', 0)})"

            # 상태 저장(복구용 핵심 필드)
            runtime_state["last_action_ts"] = float(last_action_ts)
            runtime_state["last_signal_hash"] = last_signal_hash
            runtime_state["pending_order"] = pending_order
            runtime_state["last_buy_stop_price"] = last_buy_stop_price
            runtime_state["last_buy_ts"] = float(last_buy_ts)
            runtime_state["partial_tp_done"] = sorted(list(partial_tp_done))
            runtime_state["trailing_peak_price"] = float(trailing_peak_price)
            runtime_state["last_block_hash"] = last_block_hash
            runtime_state["last_block_ts"] = float(last_block_ts)
            save_runtime_state(runtime_state_path, runtime_state)

            if runtime_state.get("halted", False):
                # Special case: after hard stop, allow auto-resume within 24h if minute consensus recovers.
                halt_reason = str(runtime_state.get("halt_reason", "unknown"))
                can_try_resume = (halt_reason.startswith("hard_stop"))
                hard_stop_ts = float(runtime_state.get("hard_stop_ts", 0.0) or 0.0)
                in_window = (hard_stop_ts > 0) and ((time.time() - hard_stop_ts) <= float(hard_stop_recovery_window_sec))

                if can_try_resume and in_window and enable_mtf_score_sizing:
                    try:
                        def _mtf_ok(_df):
                            if _df is None or len(_df) < 80:
                                return False
                            _close = _df["close"]
                            _ma50 = _close.rolling(50).mean()
                            _ma200 = _close.rolling(200).mean()
                            return bool(
                                pd.notna(_ma50.iloc[-1])
                                and pd.notna(_ma200.iloc[-1])
                                and float(_ma50.iloc[-1]) > float(_ma200.iloc[-1])
                                and float(_close.iloc[-1]) > float(_ma50.iloc[-1])
                            )

                        minute_flags = []
                        for itv in minute_consensus_intervals:
                            _df_itv = pyupbit.get_ohlcv(market, interval=itv, count=max(mtf_lookback, 220))
                            minute_flags.append(_mtf_ok(_df_itv))
                        minute_consensus = all(minute_flags) if minute_flags else False

                        hour_mtf_score = 0
                        for itv, w in hour_score_weights.items():
                            _df_itv = pyupbit.get_ohlcv(market, interval=itv, count=max(mtf_lookback, 220))
                            if _mtf_ok(_df_itv):
                                hour_mtf_score += int(w)

                        if minute_consensus and hour_mtf_score >= mtf_stage_thresholds.get("scout", 30):
                            runtime_state["halted"] = False
                            runtime_state["halt_reason"] = ""
                            runtime_state["hard_stop_ts"] = 0.0
                            save_runtime_state(runtime_state_path, runtime_state)
                            notify_event(alert_webhook_url, alert_events, "resume", f"hard-stop recovery signal | score={hour_mtf_score}")
                            prev_halted = False
                        else:
                            if not prev_halted:
                                notify_event(alert_webhook_url, alert_events, "halted", f"halted: {halt_reason}")
                            prev_halted = True
                            logging.error("TRADING HALTED | reason=%s | recovery_window_remaining=%.0fs | minute=%s hour_score=%s",
                                          halt_reason,
                                          max(0.0, float(hard_stop_recovery_window_sec) - (time.time() - hard_stop_ts)) if hard_stop_ts > 0 else 0.0,
                                          minute_consensus,
                                          hour_mtf_score)
                            time.sleep(check_seconds)
                            continue
                    except Exception as e:
                        logging.warning("hard-stop recovery check failed: %s", e)

                if not prev_halted:
                    notify_event(alert_webhook_url, alert_events, "halted", f"halted: {halt_reason}")
                prev_halted = True
                logging.error("TRADING HALTED | reason=%s", halt_reason)
                time.sleep(check_seconds)
                continue
            elif prev_halted:
                notify_event(alert_webhook_url, alert_events, "resume", "trading resumed")
                prev_halted = False

            if coin_balance <= 0:
                last_buy_stop_price = None
                entry_stop_price = None
                last_buy_ts = 0.0
                trailing_peak_price = 0.0
                partial_tp_done = set()
                early_fail_done = set()
                active_box_high = None
                addon_done = False

            df = pyupbit.get_ohlcv(market, interval="minute60", count=320)
            if df is None or len(df) < 80:
                logging.warning("ohlcv unavailable or too short")
                time.sleep(check_seconds)
                continue

            df_mtf = pyupbit.get_ohlcv(market, interval=mtf_interval, count=max(mtf_lookback, 220))
            df_mtf_2 = pyupbit.get_ohlcv(market, interval=mtf_interval_2, count=max(mtf_lookback, 220))

            def _mtf_ma(_df):
                if _df is None or len(_df) < 80:
                    return None
                _close = _df["close"]
                _ma50 = _close.rolling(50).mean()
                _ma200 = _close.rolling(200).mean()
                if pd.isna(_ma50.iloc[-1]) or pd.isna(_ma200.iloc[-1]):
                    return None
                return float(_close.iloc[-1]), float(_ma50.iloc[-1]), float(_ma200.iloc[-1])

            def _mtf_ok(_df):
                x = _mtf_ma(_df)
                if x is None:
                    return False
                _c, _ma50, _ma200 = x
                return bool(_ma50 > _ma200 and _c > _ma50)

            mtf_trend_ok = _mtf_ok(df_mtf) and _mtf_ok(df_mtf_2)

            # Regime: downtrend block (use 4h df by default: MTF_TREND_INTERVAL_2)
            downtrend_block = False
            _ma_info = _mtf_ma(df_mtf_2)
            if _ma_info is not None:
                _, ma50_4h, ma200_4h = _ma_info
                downtrend_block = (ma50_4h < ma200_4h)

            # Phase A: minute consensus + hour MTF score (optional)
            minute_consensus = True
            hour_mtf_score = None
            mtf_stage = None
            score_based_buy_krw = None
            if enable_mtf_score_sizing:
                # minute consensus (3m, 5m, 15m by default)
                minute_flags = []
                for itv in minute_consensus_intervals:
                    _df_itv = pyupbit.get_ohlcv(market, interval=itv, count=max(mtf_lookback, 220))
                    minute_flags.append(_mtf_ok(_df_itv))
                minute_consensus = all(minute_flags) if minute_flags else True

                # hour score (30m, 1h, 4h, 1d, 1w)
                hour_mtf_score = 0
                for itv, w in hour_score_weights.items():
                    _df_itv = pyupbit.get_ohlcv(market, interval=itv, count=max(mtf_lookback, 220))
                    if _mtf_ok(_df_itv):
                        hour_mtf_score += int(w)

                # stage mapping
                if hour_mtf_score >= mtf_stage_thresholds["heavy"]:
                    mtf_stage = "heavy"
                elif hour_mtf_score >= mtf_stage_thresholds["medium"]:
                    mtf_stage = "medium"
                elif hour_mtf_score >= mtf_stage_thresholds["light"]:
                    mtf_stage = "light"
                elif hour_mtf_score >= mtf_stage_thresholds["scout"]:
                    mtf_stage = "scout"
                else:
                    mtf_stage = "none"

                pct = float(mtf_stage_pcts.get(mtf_stage, 0.0)) if mtf_stage != "none" else 0.0
                score_based_buy_krw = max(0.0, equity * pct)

            signal_score, signal_reasons, metrics = evaluate_signals(df, avg_buy_price, cfg, mtf_trend_ok=mtf_trend_ok)
            buy_score_raw = int(metrics.get("buy_score_raw", 0))
            buy_reasons_raw = list(metrics.get("buy_reasons_raw", []))
            buy_score = int(metrics.get("buy_score", 0))
            buy_reasons = list(metrics.get("buy_reasons", []))

            # ?ㅼ쟾 猷? 吏곸쟾 怨좎젏 紐명넻 ?뚰뙆 + ?λ? ?묐큺 ?뺤씤
            ma_exit = df["close"].rolling(ma_exit_period).mean()
            last_open = float(df["open"].iloc[-1])
            last_close = float(df["close"].iloc[-1])
            last_low = float(df["low"].iloc[-1])
            last_body = abs(last_close - last_open)
            atr_now = float(compute_atr(df, int(cfg["ATR_PERIOD"])).iloc[-1])
            strong_bull = (last_close > last_open) and (atr_now > 0) and (last_body >= cfg["BREAKOUT_BODY_ATR_MULT"] * atr_now)
            recent_high_prev = float(df["high"].tail(breakout_lookback + 1).iloc[:-1].max()) if len(df) > breakout_lookback else float(df["high"].iloc[:-1].max())
            breakout_long = last_close > recent_high_prev and strong_bull

            # ?〓낫 諛뺤뒪 ?뚰뙆 + 由ы뀒?ㅽ듃 異붽?吏꾩엯
            box_window = df.tail(box_lookback + 1).iloc[:-1] if len(df) > box_lookback else df.iloc[:-1]
            box_high = float(box_window["high"].max())
            box_low = float(box_window["low"].min())
            box_breakout_long = strong_bull and last_close > box_high
            if box_breakout_long:
                active_box_high = box_high
                addon_done = False

            news_risk = news_state.negative_score >= negative_news_threshold
            if news_risk:
                signal_score += 1
                buy_score = max(0, buy_score - 1)
                signal_reasons.append(f"negative_news(score={news_state.negative_score})")
                buy_reasons.append(f"negative_news(score={news_state.negative_score})")

            # risk mode: external file(logs/market_risk.json) -> fallback to news-derived score
            external_risk = load_market_risk(market_risk_path)
            if external_risk["risk_score"] is None:
                risk_score = min(100.0, float(news_state.negative_score) * 25.0)
                risk_source = "news_fallback"
            else:
                risk_score = float(external_risk["risk_score"])
                risk_source = str(external_risk["source"])

            risk_mode = resolve_risk_mode(risk_score, cfg)
            tuned = apply_risk_mode(risk_mode, {
                "signal_score_threshold": float(signal_score_threshold),
                "buy_score_threshold": float(buy_score_threshold),
                "sell_ratio": float(sell_ratio),
                "buy_ratio": float(buy_ratio),
            }, cfg)
            active_signal_threshold = int(tuned["signal_score_threshold"])
            active_buy_threshold = int(tuned["buy_score_threshold"])
            active_sell_ratio = float(tuned["sell_ratio"])
            active_buy_ratio = float(tuned["buy_ratio"])

            should_sell = signal_score >= active_signal_threshold and coin_balance > 0
            should_buy = buy_score >= active_buy_threshold and krw_balance >= min_buy_krw
            raw_should_buy = bool(should_buy)
            raw_buy_reasons = list(buy_reasons)
            raw_buy_score = int(buy_score)
            raw_buy_score_pre_mtf = int(buy_score_raw)
            raw_buy_reasons_pre_mtf = list(buy_reasons_raw)

            # Phase A gate: 3m/5m/15m AND consensus + hour score stage
            if enable_mtf_score_sizing:
                if not minute_consensus:
                    should_buy = False
                    buy_reasons.append("minute_mtf_no_consensus")
                else:
                    buy_reasons.append(f"minute_mtf_consensus({sum(1 for x in minute_flags if x)}/{len(minute_flags)})")
                    if hour_mtf_score is not None:
                        buy_reasons.append(f"hour_mtf_score({hour_mtf_score})")

                    # Relaxation mode (for log accumulation):
                    # If minute consensus passes but hour score is below scout threshold, still allow SCOUT (min order).
                    if hour_mtf_score is not None and (score_based_buy_krw is None or score_based_buy_krw <= 0):
                        mtf_stage = "scout"
                        score_based_buy_krw = max(min_buy_krw, float(equity) * float(mtf_stage_pcts.get("scout", 0.01)))
                        buy_reasons.append(f"hour_mtf_score_low_allow_scout({hour_mtf_score})")

                    # Regime filter: block buys in downtrend (4h MA50 < MA200)
                    if downtrend_block_enabled and downtrend_block:
                        should_buy = False
                        buy_reasons.append("downtrend_block(ma50<ma200@4h)")

                    # Pullback filter: in uptrend only, require fib retracement zone for entries
                    if should_buy and fib_pullback_enabled and (not downtrend_block):
                        try:
                            if fib_lookback > 0 and len(df) > fib_lookback:
                                win = df.tail(fib_lookback)
                            else:
                                win = df
                            # swing confirmation: require that low occurs before high within window (up-move then pullback)
                            if fib_swing_confirm:
                                hi_i = int(win["high"].values.argmax())
                                lo_i = int(win["low"].values.argmin())
                                if not (lo_i < hi_i):
                                    should_buy = False
                                    buy_reasons.append("fib_swing_unconfirmed")
                                    raise RuntimeError("fib swing unconfirmed")

                            swing_high = float(win["high"].max())
                            swing_low = float(win["low"].min())
                            rng = max(1e-9, swing_high - swing_low)
                            zone_high = swing_high - (rng * float(fib_min))
                            zone_low = swing_high - (rng * float(fib_max))
                            in_zone = (current_price >= zone_low) and (current_price <= zone_high)
                            if not in_zone:
                                should_buy = False
                                buy_reasons.append(f"fib_pullback_block({fib_min:.3f}-{fib_max:.3f})")
                            else:
                                buy_reasons.append(f"fib_pullback_ok({fib_min:.3f}-{fib_max:.3f})")
                        except Exception as e:
                            logging.warning("fib pullback calc failed: %s", e)

                    if mtf_stage is not None:
                        buy_reasons.append(f"mtf_stage({mtf_stage})")

            intended_sell = should_sell
            intended_buy = should_buy

            # emergency sell flag (for bypass/idempotency and full liquidation)
            emergency_sell = False

            # 진입 확정: breakout 조건 결합 방식(and/or) 파라미터화
            if entry_mode == "confirm_breakout":
                gate_ok = (breakout_long and box_breakout_long) if breakout_gate_mode == "and" else (breakout_long or box_breakout_long)
                if not gate_ok:
                    should_buy = False
                    buy_reasons.append(f"breakout_gate_block(mode={breakout_gate_mode})")

            # 손익비(R:R) 필터
            rr_value = None
            stop_price = min(last_low, metrics["ma20"] if metrics["ma20"] > 0 else last_low)
            target_price = max(recent_high_prev, last_close + (atr_now * rr_target_atr_mult))
            risk_per_unit = max(0.0, last_close - stop_price)
            reward_per_unit = max(0.0, target_price - last_close)
            if risk_per_unit > 0:
                rr_value = reward_per_unit / risk_per_unit
            if should_buy and (rr_value is None or rr_value < rr_min):
                should_buy = False
                buy_reasons.append(f"rr_block(rr={0 if rr_value is None else rr_value:.2f}<{rr_min:.2f})")

            # ATR 변동성 레짐 필터
            atr_pct = (atr_now / max(current_price, 1e-9)) * 100.0
            if should_buy and (atr_pct < atr_regime_min_pct or atr_pct > atr_regime_max_pct):
                should_buy = False
                buy_reasons.append(f"atr_regime_block({atr_pct:.2f}%)")

            # Volatility spike block (avoid entries during shock candles)
            if should_buy and vol_spike_block_enabled and atr_now > 0 and vol_spike_block_atr_mult > 0:
                if last_body >= (atr_now * vol_spike_block_atr_mult):
                    should_buy = False
                    buy_reasons.append(f"vol_spike_block(body>=ATR*{vol_spike_block_atr_mult:.2f})")

            # 스프레드 필터 + 진입 후 봉 쿨다운
            spread_bps = get_spread_bps(market)
            if should_buy and spread_bps_max > 0 and spread_bps > spread_bps_max:
                should_buy = False
                buy_reasons.append(f"spread_block({spread_bps:.1f}>{spread_bps_max:.1f})")

            entry_cooldown_sec = max(0, post_entry_cooldown_bars) * 3600
            if should_buy and last_buy_ts > 0 and (time.time() - last_buy_ts) < entry_cooldown_sec:
                should_buy = False
                buy_reasons.append("post_entry_cooldown_block")

            # 진입봉 저점/ATR 기반 스탑(저점 이탈 시 청산)
            if stop_by_entry_candle and coin_balance > 0 and last_buy_stop_price is not None and current_price < last_buy_stop_price:
                should_sell = True
                signal_reasons.append(f"entry_candle_stop({last_buy_stop_price:.0f})")

            # Hard stop (최후 보험): avg_buy_price 대비 -HARD_STOP_PCT 하락 시 전량 청산
            if coin_balance > 0 and avg_buy_price > 0 and hard_stop_pct > 0:
                hard_stop_price = avg_buy_price * (1.0 - hard_stop_pct)
                if current_price < hard_stop_price:
                    should_sell = True
                    emergency_sell = True
                    active_sell_ratio = 1.0
                    signal_reasons.append(f"hard_stop({hard_stop_pct*100:.1f}%)")

            # 멱등성/중복 주문 방지
            now_ts = time.time()
            in_cooldown = (now_ts - last_action_ts) < trade_cooldown_seconds
            side_for_hash = "sell" if should_sell else ("buy" if should_buy else "none")
            reasons_for_hash = signal_reasons if should_sell else buy_reasons
            score_for_hash = signal_score if should_sell else buy_score
            signal_hash = make_signal_hash(side_for_hash, reasons_for_hash, int(score_for_hash), market)

            if pending_order:
                should_sell = False
                should_buy = False
            elif in_cooldown:
                should_sell = False
                should_buy = False
            elif side_for_hash != "none" and signal_hash == last_signal_hash:
                should_sell = False
                should_buy = False

            # MA22 하향 이탈 익절/청산
            ma_exit_now = float(ma_exit.iloc[-1]) if pd.notna(ma_exit.iloc[-1]) else 0.0
            if take_profit_by_ma22 and coin_balance > 0 and ma_exit_now > 0 and last_close < ma_exit_now:
                should_sell = True
                emergency_sell = True
                signal_reasons.append(f"ma{ma_exit_period}_exit")

            # 잔여 물량 트레일링 스탑
            if coin_balance > 0:
                trailing_peak_price = max(trailing_peak_price, current_price)
                apply_trailing = True
                if trailing_apply_to_remainder_only and len(partial_tp_done) == 0:
                    apply_trailing = False
                if apply_trailing and trailing_stop_mode == "atr":
                    trailing_stop_price = trailing_peak_price - (atr_now * trailing_atr_mult)
                    if current_price < trailing_stop_price:
                        should_sell = True
                        signal_reasons.append(f"trailing_atr_stop({trailing_stop_price:.0f})")

            # 최종 멱등성 게이트 재적용 (긴급 청산 bypass 옵션)
            if should_sell or should_buy:
                final_side = "sell" if should_sell else "buy"
                final_reasons = signal_reasons if should_sell else buy_reasons
                final_score = signal_score if should_sell else buy_score
                final_hash = make_signal_hash(final_side, final_reasons, int(final_score), market)
                bypass = should_sell and emergency_sell and emergency_sell_bypass_idempotency
                if not bypass:
                    if pending_order or ((time.time() - last_action_ts) < trade_cooldown_seconds) or (final_hash == last_signal_hash):
                        should_sell = False
                        should_buy = False
                signal_hash = final_hash

            # 슬리피지 가드 + 일일 트레이드 제한 + 포지션 한도
            slippage_bps = (abs(current_price - last_close) / max(last_close, 1e-9)) * 10000
            if (should_sell or should_buy) and slippage_bps > max_slippage_bps:
                should_sell = False
                should_buy = False
                logging.warning("signal blocked by slippage | bps=%.1f > max=%.1f", slippage_bps, max_slippage_bps)

            if int(runtime_state.get("trades_today", 0)) >= max_trades_per_day:
                if should_buy or (should_sell and not emergency_sell):
                    should_buy = False
                    if not emergency_sell:
                        should_sell = False
                    logging.warning("signal blocked by max_trades_per_day=%s", max_trades_per_day)

            effective_max_position_krw = max_position_krw if max_position_krw > 0 else 0.0
            if max_position_fraction > 0:
                frac_cap = equity * max_position_fraction
                effective_max_position_krw = frac_cap if effective_max_position_krw <= 0 else min(effective_max_position_krw, frac_cap)

            if should_buy and effective_max_position_krw > 0 and (coin_value_krw + (krw_balance * min(probe_entry_ratio if entry_mode == 'confirm_breakout' else active_buy_ratio, active_buy_ratio))) > effective_max_position_krw:
                should_buy = False
                buy_reasons.append(f"max_position_block({effective_max_position_krw:.0f})")

            # 차단된 신호 별도 기록(운영 KPI용)
            blocked_reasons = [r for r in (signal_reasons + buy_reasons) if ("block" in str(r))]
            if (intended_buy or intended_sell) and (not should_buy and not should_sell) and blocked_reasons:
                blocked_side = "buy" if intended_buy else "sell"
                block_hash = make_signal_hash(f"block_{blocked_side}", blocked_reasons, int(signal_score if blocked_side == "sell" else buy_score), market)
                if block_hash != last_block_hash or (time.time() - last_block_ts) > 60:
                    append_trade_log(trade_log_path, {
                        "ts": int(time.time()),
                        "side": "signal_block",
                        "market": market,
                        "price": current_price,
                        "score": int(signal_score if blocked_side == "sell" else buy_score),
                        "reasons": blocked_reasons,
                        "dry_run": dry_run,
                    })
                    last_block_hash = block_hash
                    last_block_ts = time.time()

            # Candidate logging (debug/analysis): raw vs final buy decision
            candidate_log_enabled = env_bool("CANDIDATE_LOG_ENABLED", True)
            if candidate_log_enabled:
                # log only when there is any buy signal strength or a buy was considered
                if raw_buy_score_pre_mtf > 0 or buy_score > 0 or raw_should_buy or intended_buy:
                    append_trade_log(trade_log_path, {
                        "ts": int(time.time()),
                        "side": "buy_candidate",
                        "market": market,
                        "price": current_price,
                        "buy_score_pre_mtf": int(raw_buy_score_pre_mtf),
                        "buy_reasons_pre_mtf": raw_buy_reasons_pre_mtf,
                        "buy_score_raw": int(raw_buy_score),
                        "buy_threshold": int(active_buy_threshold),
                        "raw_should_buy": bool(raw_should_buy),
                        "final_should_buy": bool(should_buy),
                        "raw_reasons": raw_buy_reasons,
                        "final_reasons": buy_reasons,
                        "dry_run": dry_run,
                    })

            logging.info(
                "tick | mode=%s risk=%.1f(%s) mtf=%s rr=%.2f atr=%.2f%% spread=%.1fbps price=%.0f coin=%.0fKRW krw=%.0f sell=%s/%s buy=%s/%s breakout=%s boxBreak=%s ma20=%.0f ma200=%.0f vwma100=%.0f gap=%.2f%% adx=%.1f rsi=%.1f news=%s",
                risk_mode,
                risk_score,
                risk_source,
                int(mtf_trend_ok),
                -1.0 if rr_value is None else rr_value,
                atr_pct,
                spread_bps,
                current_price,
                coin_value_krw,
                krw_balance,
                signal_score,
                active_signal_threshold,
                buy_score,
                active_buy_threshold,
                int(breakout_long),
                int(box_breakout_long),
                metrics["ma20"],
                metrics["ma200"],
                metrics["vwma100"],
                metrics["ma_gap_pct"],
                metrics["adx"],
                metrics["rsi"],
                news_state.negative_score,
            )

            # 분할 익절(1R,2R 등) + Early fail cut(손실 구간 부분 정리)
            partial_tp_executed = False
            if coin_balance > 0 and avg_buy_price > 0 and last_buy_stop_price is not None and last_buy_stop_price < avg_buy_price:
                # R basis should be stable: use entry_stop_price (captured at entry) if available.
                stop_basis = entry_stop_price if (entry_stop_price is not None and entry_stop_price < avg_buy_price) else last_buy_stop_price
                risk_unit = avg_buy_price - float(stop_basis)
                r_now = (current_price - avg_buy_price) / max(risk_unit, 1e-9)

                # Early fail cut ladder (negative R): cut portions as loss deepens
                weak_trend_ctx = (metrics.get("adx", 0) < float(cfg["ADX_MIN"])) or (last_close < float(metrics.get("ma20", 0) or 0))

                # hybrid policy: cut more aggressively in weak trend; preserve right-tail in strong trend
                if early_fail_mode == "hybrid" and (not (weak_trend_ctx or (not mtf_trend_ok) or downtrend_block)):
                    n = min(len(early_fail_strong_levels_r), len(early_fail_strong_cut_ratios))
                    cut_levels = early_fail_strong_levels_r[:n]
                    cut_ratios = early_fail_strong_cut_ratios[:n]
                else:
                    n = min(len(early_fail_levels_r), len(early_fail_cut_ratios))
                    cut_levels = early_fail_levels_r[:n]
                    cut_ratios = early_fail_cut_ratios[:n]

                early_fail_ctx_ok = (early_fail_mode == "always") or (early_fail_mode == "hybrid") or weak_trend_ctx or (not mtf_trend_ok) or downtrend_block
                if early_fail_cut_enabled and early_fail_ctx_ok and r_now < 0 and cut_levels and cut_ratios:
                    # clamp
                    max_cut = min(1.0, max(0.0, float(early_fail_max_cut_ratio)))
                    for lvl, ratio in zip(cut_levels, cut_ratios):
                        try:
                            lvl_f = float(lvl)
                            ratio_f = float(ratio)
                        except Exception:
                            continue
                        if lvl_f <= 0:
                            continue
                        key = f"{lvl_f:.4f}"
                        if key in early_fail_done:
                            continue
                        if r_now <= (-lvl_f) and ratio_f > 0:
                            cut_ratio = min(max(0.0, ratio_f), 1.0)
                            cut_ratio = min(cut_ratio, max_cut)
                            cut_coin = coin_balance * cut_ratio
                            cut_value = cut_coin * current_price
                            if cut_value >= min_sell_krw:
                                logging.warning("EARLY FAIL CUT | R=%.2f level=-%.2f ratio=%.2f", r_now, lvl_f, cut_ratio)
                                append_trade_log(trade_log_path, {
                                    "ts": int(time.time()),
                                    "side": "sell_partial",
                                    "market": market,
                                    "price": current_price,
                                    "score": signal_score,
                                    "reasons": [f"early_fail_cut_{lvl_f:.2f}R"],
                                    "dry_run": dry_run,
                                })
                                filled_cut = False
                                if dry_run:
                                    last_action_ts = now_ts
                                    last_signal_hash = make_signal_hash("sell_partial", [f"early_fail_cut_{lvl_f:.2f}R"], int(signal_score), market)
                                    filled_cut = True
                                else:
                                    before_coin, before_krw = coin_balance, krw_balance
                                    notify_event(alert_webhook_url, alert_events, "order_sent", f"early_fail_cut sent {market} ratio={cut_ratio:.2f}")
                                    result = place_order_with_retry("early_fail_cut_market_order", upbit.sell_market_order, market, cut_coin)
                                    logging.warning("early_fail_cut result: %s", result)
                                    order_uuid = result.get("uuid") if isinstance(result, dict) else None
                                    if order_uuid:
                                        pending_order = {"uuid": order_uuid, "side": "early_fail_cut", "created_ts": now_ts, "timeout_notified": False}
                                    time.sleep(1.0)
                                    after_coin, _, _ = get_account_state(upbit, market)
                                    after_krw = get_krw_balance(upbit)
                                    filled_cut = verify_position_change(before_coin, before_krw, after_coin, after_krw, "sell")
                                    logging.warning("early_fail_cut fill check | filled=%s before_coin=%.8f after_coin=%.8f", filled_cut, before_coin, after_coin)
                                    if filled_cut:
                                        pending_order = None
                                        notify_event(alert_webhook_url, alert_events, "filled", f"early_fail_cut filled {market} level=-{lvl_f:.2f}R")
                                if filled_cut:
                                    early_fail_done.add(key)
                                    runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                                    partial_tp_executed = True
                            break

                # Profit-side partial TP
                for lvl, ratio in zip(partial_tp_levels, partial_tp_ratios):
                    key = f"{lvl:.4f}"
                    if key in partial_tp_done:
                        continue
                    if r_now >= lvl and ratio > 0:
                        tp_coin = coin_balance * min(max(ratio, 0.0), 1.0)
                        tp_value = tp_coin * current_price
                        if tp_value >= min_sell_krw:
                            logging.warning("PARTIAL TP | R=%.2f level=%.2f ratio=%.2f", r_now, lvl, ratio)
                            append_trade_log(trade_log_path, {
                                "ts": int(time.time()),
                                "side": "sell_partial",
                                "market": market,
                                "price": current_price,
                                "score": signal_score,
                                "reasons": [f"partial_tp_{lvl:.2f}R"],
                                "dry_run": dry_run,
                            })
                            filled_tp = False
                            if dry_run:
                                last_action_ts = now_ts
                                last_signal_hash = make_signal_hash("sell_partial", [f"partial_tp_{lvl:.2f}R"], int(signal_score), market)
                                filled_tp = True
                            else:
                                before_coin, before_krw = coin_balance, krw_balance
                                notify_event(alert_webhook_url, alert_events, "order_sent", f"sell_partial sent {market} ratio={ratio:.2f}")
                                result = place_order_with_retry("sell_partial_market_order", upbit.sell_market_order, market, tp_coin)
                                logging.warning("sell_partial result: %s", result)
                                order_uuid = result.get("uuid") if isinstance(result, dict) else None
                                if order_uuid:
                                    pending_order = {"uuid": order_uuid, "side": "sell_partial", "created_ts": now_ts, "timeout_notified": False}
                                time.sleep(1.0)
                                after_coin, _, _ = get_account_state(upbit, market)
                                after_krw = get_krw_balance(upbit)
                                filled_tp = verify_position_change(before_coin, before_krw, after_coin, after_krw, "sell")
                                logging.warning("sell_partial fill check | filled=%s before_coin=%.8f after_coin=%.8f", filled_tp, before_coin, after_coin)
                                if filled_tp:
                                    pending_order = None
                                    notify_event(alert_webhook_url, alert_events, "filled", f"sell_partial filled {market} level={lvl:.2f}R")
                            if filled_tp:
                                partial_tp_done.add(key)
                                runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                                # Optional: after first partial TP, move stop to breakeven (+fees buffer)
                                if move_stop_to_be_after_tp1 and len(partial_tp_done) == 1 and avg_buy_price > 0:
                                    # Context-aware BE move: protect left-tail in weak trend, preserve right-tail in strong trend.
                                    strong_ctx = (mtf_trend_ok is True) and (metrics.get("adx", 0) >= float(cfg["ADX_MIN"])) and (hour_mtf_score is not None and int(hour_mtf_score) >= int(mtf_stage_thresholds.get("medium", 65)))
                                    weak_ctx = (not mtf_trend_ok) or downtrend_block or (metrics.get("adx", 0) < float(cfg["ADX_MIN"]))

                                    target_stop = None
                                    if be_move_mode == "always":
                                        target_stop = avg_buy_price * (1.0 + (be_offset_bps / 10000.0))
                                        signal_reasons.append(f"be_stop_after_tp1({be_offset_bps:.1f}bps)")
                                    elif be_move_mode == "weak_only":
                                        if weak_ctx:
                                            target_stop = avg_buy_price * (1.0 + (be_offset_bps / 10000.0))
                                            signal_reasons.append(f"be_stop_after_tp1_weak({be_offset_bps:.1f}bps)")
                                    else:  # hybrid default
                                        if weak_ctx and (not strong_ctx):
                                            target_stop = avg_buy_price * (1.0 + (be_offset_bps / 10000.0))
                                            signal_reasons.append(f"be_stop_after_tp1_weak({be_offset_bps:.1f}bps)")
                                        elif strong_ctx and entry_stop_price is not None and entry_stop_price < avg_buy_price:
                                            # in strong trend, lift stop closer but keep some room to avoid chopping winners
                                            risk0 = avg_buy_price - float(entry_stop_price)
                                            target_stop = avg_buy_price + (risk0 * float(be_strong_stop_r))
                                            signal_reasons.append(f"stop_lift_after_tp1_strong({be_strong_stop_r:.2f}R)")

                                    if target_stop is not None:
                                        if last_buy_stop_price is None or float(target_stop) > float(last_buy_stop_price):
                                            last_buy_stop_price = float(target_stop)
                                        # keep entry_stop_price unchanged; we want stable R basis
                                partial_tp_executed = True
                        break

            if partial_tp_executed:
                runtime_state["last_action_ts"] = float(last_action_ts)
                runtime_state["last_signal_hash"] = last_signal_hash
                runtime_state["pending_order"] = pending_order
                runtime_state["last_buy_stop_price"] = last_buy_stop_price
                runtime_state["last_buy_ts"] = float(last_buy_ts)
                runtime_state["partial_tp_done"] = sorted(list(partial_tp_done))
                runtime_state["early_fail_done"] = sorted(list(early_fail_done))
                runtime_state["trailing_peak_price"] = float(trailing_peak_price)
                runtime_state["last_block_hash"] = last_block_hash
                runtime_state["last_block_ts"] = float(last_block_ts)
                save_runtime_state(runtime_state_path, runtime_state)
                time.sleep(check_seconds)
                continue

            if should_sell:
                sell_amount_coin = coin_balance * active_sell_ratio
                sell_value_krw = sell_amount_coin * current_price

                if sell_value_krw < min_sell_krw:
                    logging.warning("sell skipped | below min_sell_krw: %.0f < %.0f", sell_value_krw, min_sell_krw)
                else:
                    logging.warning("SELL SIGNAL | score=%s reasons=%s", signal_score, ", ".join(signal_reasons))
                    feedback = build_stop_feedback(signal_reasons)

                    # Stop event reporting policy (Phase A): report ONLY on hard stop
                    stop_event = any(str(r).startswith("hard_stop") for r in signal_reasons)
                    if stop_event:
                        runtime_state["yearly_stop_count"] = int(runtime_state.get("yearly_stop_count", 0)) + 1
                        runtime_state["hard_stop_ts"] = float(time.time())
                        runtime_state["halted"] = True
                        runtime_state["halt_reason"] = f"hard_stop_triggered({hard_stop_pct*100:.1f}%)"
                        save_runtime_state(runtime_state_path, runtime_state)

                    append_trade_log(trade_log_path, {
                        "ts": int(time.time()),
                        "side": "sell",
                        "market": market,
                        "price": current_price,
                        "score": signal_score,
                        "reasons": signal_reasons,
                        "feedback": feedback,
                        "stop_event": stop_event,
                        "yearly_stop_count": int(runtime_state.get("yearly_stop_count", 0)),
                        "dry_run": dry_run,
                    })
                    logging.warning("SELL FEEDBACK | %s", feedback)
                    if dry_run:
                        logging.warning("DRY_RUN sell | market=%s amount=%.8f (~%.0fKRW)", market, sell_amount_coin, sell_value_krw)
                        last_action_ts = now_ts
                        last_signal_hash = signal_hash
                        runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                        loss_hit = current_price < (avg_buy_price * 0.998) if avg_buy_price > 0 else False
                        if loss_hit:
                            runtime_state["consecutive_losses_today"] = int(runtime_state.get("consecutive_losses_today", 0)) + 1
                        else:
                            runtime_state["consecutive_losses_today"] = 0
                    else:
                        before_coin, before_krw = coin_balance, krw_balance
                        notify_event(alert_webhook_url, alert_events, "order_sent", f"sell sent {market} amount={sell_amount_coin:.8f}")
                        result = place_order_with_retry("sell_market_order", upbit.sell_market_order, market, sell_amount_coin)
                        logging.warning("sell_market_order result: %s", result)
                        order_uuid = result.get("uuid") if isinstance(result, dict) else None
                        pending_order = {"uuid": order_uuid, "side": "sell", "created_ts": now_ts, "timeout_notified": False} if order_uuid else None
                        time.sleep(1.0)
                        after_coin, _, _ = get_account_state(upbit, market)
                        after_krw = get_krw_balance(upbit)
                        filled = verify_position_change(before_coin, before_krw, after_coin, after_krw, "sell")
                        logging.warning("sell fill check | filled=%s before_coin=%.8f after_coin=%.8f", filled, before_coin, after_coin)
                        if filled:
                            last_action_ts = now_ts
                            last_signal_hash = signal_hash
                            runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                            loss_hit = current_price < (avg_buy_price * 0.998) if avg_buy_price > 0 else False
                            if loss_hit:
                                runtime_state["consecutive_losses_today"] = int(runtime_state.get("consecutive_losses_today", 0)) + 1
                            else:
                                runtime_state["consecutive_losses_today"] = 0
                            notify_event(alert_webhook_url, alert_events, "filled", f"sell filled {market} price={current_price:.0f}")
                            pending_order = None
                        elif order_uuid is None:
                            notify_event(alert_webhook_url, alert_events, "rejected", f"sell rejected {market}")

            elif should_buy:
                base_ratio = probe_entry_ratio if entry_mode == "confirm_breakout" else active_buy_ratio
                buy_krw = krw_balance * min(base_ratio, active_buy_ratio)

                # Phase A: score-based sizing (equity fraction) with existing safety min() sizing
                if enable_mtf_score_sizing and score_based_buy_krw is not None and score_based_buy_krw > 0:
                    buy_krw = min(buy_krw, score_based_buy_krw)

                # Optional risk-based sizing (uses stop distance from RR filter block)
                if risk_per_trade > 0 and stop_price > 0 and last_close > stop_price:
                    risk_budget_krw = equity * risk_per_trade
                    stop_pct = max(1e-6, (last_close - stop_price) / last_close)
                    buy_krw_risk = risk_budget_krw / stop_pct
                    buy_krw = min(buy_krw, buy_krw_risk, krw_balance)

                if buy_krw < min_buy_krw:
                    logging.warning("buy skipped | below min_buy_krw: %.0f < %.0f", buy_krw, min_buy_krw)
                else:
                    logging.warning("BUY SIGNAL | score=%s reasons=%s", buy_score, ", ".join(buy_reasons))
                    append_trade_log(trade_log_path, {
                        "ts": int(time.time()),
                        "side": "buy",
                        "market": market,
                        "price": current_price,
                        "score": buy_score,
                        "reasons": buy_reasons,
                        "dry_run": dry_run,
                    })
                    if dry_run:
                        logging.warning("DRY_RUN buy | market=%s krw=%.0f", market, buy_krw)
                        last_action_ts = now_ts
                        last_signal_hash = signal_hash
                        last_buy_ts = now_ts
                        trailing_peak_price = current_price
                        partial_tp_done = set()
                        runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                        if stop_by_entry_candle:
                            structural_stop = last_low
                            atr_stop = (last_close - (atr_now * entry_stop_atr_mult)) if (entry_stop_atr_mult > 0 and atr_now > 0) else structural_stop
                            last_buy_stop_price = min(structural_stop, atr_stop)
                            entry_stop_price = float(last_buy_stop_price)
                            early_fail_done = set()
                    else:
                        before_coin, before_krw = coin_balance, krw_balance
                        notify_event(alert_webhook_url, alert_events, "order_sent", f"buy sent {market} krw={buy_krw:.0f}")
                        result = place_order_with_retry("buy_market_order", upbit.buy_market_order, market, buy_krw)
                        logging.warning("buy_market_order result: %s", result)
                        order_uuid = result.get("uuid") if isinstance(result, dict) else None
                        pending_order = {"uuid": order_uuid, "side": "buy", "created_ts": now_ts, "timeout_notified": False} if order_uuid else None
                        time.sleep(1.0)
                        after_coin, _, _ = get_account_state(upbit, market)
                        after_krw = get_krw_balance(upbit)
                        filled = verify_position_change(before_coin, before_krw, after_coin, after_krw, "buy")
                        logging.warning("buy fill check | filled=%s before_coin=%.8f after_coin=%.8f", filled, before_coin, after_coin)
                        if filled:
                            last_action_ts = now_ts
                            last_signal_hash = signal_hash
                            last_buy_ts = now_ts
                            trailing_peak_price = current_price
                            partial_tp_done = set()
                            runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                            notify_event(alert_webhook_url, alert_events, "filled", f"buy filled {market} price={current_price:.0f}")
                            if stop_by_entry_candle:
                                structural_stop = last_low
                                atr_stop = (last_close - (atr_now * entry_stop_atr_mult)) if (entry_stop_atr_mult > 0 and atr_now > 0) else structural_stop
                                last_buy_stop_price = min(structural_stop, atr_stop)
                                entry_stop_price = float(last_buy_stop_price)
                                early_fail_done = set()
                            pending_order = None
                        elif order_uuid is None:
                            notify_event(alert_webhook_url, alert_events, "rejected", f"buy rejected {market}")

            # 박스 돌파 후 리테스트(지지 확인) 추가진입
            if enable_retest_addon and coin_balance > 0 and active_box_high is not None and not addon_done:
                retest_band = active_box_high * box_buffer_pct
                retest_hit = abs(current_price - active_box_high) <= retest_band
                retest_hold = (last_close >= active_box_high) and (last_close >= metrics["ma20"])

                addon_allowed = True
                if spread_bps_max > 0 and spread_bps > spread_bps_max:
                    addon_allowed = False
                if atr_pct < atr_regime_min_pct or atr_pct > atr_regime_max_pct:
                    addon_allowed = False
                if last_buy_ts > 0 and (time.time() - last_buy_ts) < entry_cooldown_sec:
                    addon_allowed = False

                if retest_hit and retest_hold and addon_allowed and krw_balance >= min_buy_krw:
                    addon_krw = krw_balance * addon_ratio
                    if effective_max_position_krw > 0 and (coin_value_krw + addon_krw) > effective_max_position_krw:
                        addon_krw = max(0.0, effective_max_position_krw - coin_value_krw)
                    if addon_krw >= min_buy_krw:
                        logging.warning("ADD-ON BUY | box_retest level=%.0f krw=%.0f", active_box_high, addon_krw)
                        append_trade_log(trade_log_path, {
                            "ts": int(time.time()),
                            "side": "buy_addon",
                            "market": market,
                            "price": current_price,
                            "score": buy_score,
                            "reasons": ["box_retest_addon"],
                            "dry_run": dry_run,
                        })
                        if not dry_run:
                            notify_event(alert_webhook_url, alert_events, "order_sent", f"buy_addon sent {market} krw={addon_krw:.0f}")
                            result = place_order_with_retry("buy_addon_market_order", upbit.buy_market_order, market, addon_krw)
                            logging.warning("buy_addon result: %s", result)
                        last_buy_ts = now_ts
                        runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                        addon_done = True

            runtime_state["last_action_ts"] = float(last_action_ts)
            runtime_state["last_signal_hash"] = last_signal_hash
            runtime_state["pending_order"] = pending_order
            runtime_state["last_buy_stop_price"] = last_buy_stop_price
            runtime_state["entry_stop_price"] = entry_stop_price
            runtime_state["last_buy_ts"] = float(last_buy_ts)
            runtime_state["partial_tp_done"] = sorted(list(partial_tp_done))
            runtime_state["early_fail_done"] = sorted(list(early_fail_done))
            runtime_state["trailing_peak_price"] = float(trailing_peak_price)
            save_runtime_state(runtime_state_path, runtime_state)
            loop_error_count = 0

            time.sleep(check_seconds)

        except Exception as e:
            err = str(e).lower()
            loop_error_count += 1

            # pending_order를 즉시 초기화하지 않고 보존해서 다음 루프에서 재조회/타임아웃 처리
            runtime_state["pending_order"] = pending_order

            if any(k in err for k in ["unauthorized", "invalid access key", "jwt", "forbidden", "permission"]):
                runtime_state["halted"] = True
                runtime_state["halt_reason"] = "auth_error"
                notify_event(alert_webhook_url, alert_events, "halted", "auth_error: check API keys/permissions")

            save_runtime_state(runtime_state_path, runtime_state)
            logging.exception("loop error #%s: %s", loop_error_count, e)

            backoff_sec = min(60, max(5, check_seconds) * (2 ** min(loop_error_count, 3)))
            time.sleep(backoff_sec)


if __name__ == "__main__":
    run()
