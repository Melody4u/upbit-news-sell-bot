import os
import time
import json
import logging
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import requests
import pyupbit
import pandas as pd
from dotenv import load_dotenv


NEGATIVE_KEYWORDS = [
    "hack", "해킹", "제재", "금지", "소송", "파산", "상장폐지", "투자주의", "규제", "investigation"
]


@dataclass
class NewsState:
    last_checked: float = 0.0
    negative_score: int = 0
    headlines: List[str] = None

    def __post_init__(self):
        if self.headlines is None:
            self.headlines = []


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "y")


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def setup_logger() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def fetch_negative_news_score(
    brave_api_key: str,
    query: str,
    country: str,
    lang: str,
    count: int,
) -> Tuple[int, List[str]]:
    if not brave_api_key:
        return 0, []

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key,
    }
    params = {
        "q": query,
        "count": count,
        "country": country,
        "search_lang": lang,
        "freshness": "pd",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("web", {}).get("results", [])
    texts = []
    for item in items:
        title = item.get("title", "") or ""
        desc = item.get("description", "") or ""
        combined = f"{title} {desc}".strip()
        if combined:
            texts.append(combined)

    score = 0
    for text in texts:
        lowered = text.lower()
        if any(k.lower() in lowered for k in NEGATIVE_KEYWORDS):
            score += 1

    return score, texts[:5]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
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

    atr = compute_atr(df, period).replace(0, pd.NA)

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0)


def get_account_state(upbit, market: str):
    balances = upbit.get_balances()
    base = market.split("-")[1]

    coin_balance = 0.0
    avg_buy_price = 0.0
    for b in balances:
        if b.get("currency") == base:
            coin_balance = float(b.get("balance") or 0)
            avg_buy_price = float(b.get("avg_buy_price") or 0)
            break

    current_price = pyupbit.get_current_price(market)
    if current_price is None:
        raise RuntimeError("현재가 조회 실패")

    return coin_balance, avg_buy_price, float(current_price)


def get_krw_balance(upbit) -> float:
    krw = upbit.get_balance("KRW")
    return float(krw or 0)


def get_spread_bps(market: str) -> float:
    try:
        ob = pyupbit.get_orderbook(market)
        if isinstance(ob, list):
            ob = ob[0] if ob else None
        if not ob:
            return 0.0
        units = ob.get("orderbook_units", [])
        if not units:
            return 0.0
        ask = float(units[0].get("ask_price") or 0)
        bid = float(units[0].get("bid_price") or 0)
        mid = (ask + bid) / 2 if (ask > 0 and bid > 0) else 0
        if mid <= 0:
            return 0.0
        return ((ask - bid) / mid) * 10000
    except Exception:
        return 0.0


def parse_float_list_csv(s: str, default: List[float]) -> List[float]:
    try:
        vals = [float(x.strip()) for x in (s or "").split(",") if x.strip()]
        return vals if vals else default
    except Exception:
        return default


def build_stop_feedback(reasons: List[str]) -> str:
    text = " | ".join(reasons)
    tips = []
    if "entry_candle_stop" in text:
        tips.append("吏꾩엯遊?????먯젅 諛쒖깮: PROBE_ENTRY_RATIO瑜???텛嫄곕굹 BREAKOUT_BODY_ATR_MULT瑜??믪뿬 ??媛뺥븳 ?뚰뙆留?吏꾩엯")
    if "ma22_exit" in text:
        tips.append("MA22 ?댄깉 泥?궛: MA_EXIT_PERIOD瑜?22??4~30?쇰줈 ?섎젮 怨쇰? 諛섏쓳 ?꾪솕 ?щ? ?먭?")
    if "sideways_filter" in text:
        tips.append("?〓낫 ?몄씠利?媛?μ꽦: SIDEWAYS_CROSS_THRESHOLD ?곹뼢 ?먮뒗 BOX ?뚰뙆 ?뺤씤 ??吏꾩엯")
    if "negative_news" in text:
        tips.append("?낆옱 ?댁뒪 ?곹뼢: NEWS_INTERVAL_SECONDS ?⑥텞/?꾧퀎移?議곗젙?쇰줈 ?댁뒪 ?꾪꽣 誘쇨컧???ъ젏寃")
    if not tips:
        tips.append("理쒓렐 30~50媛??몃젅?대뱶 濡쒓렇瑜?紐⑥븘 媛????? ?먯젅 ?먯씤遺???뚮씪誘명꽣 1媛쒖뵫留?議곗젙")
    return " / ".join(tips)


def append_trade_log(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_market_risk(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"risk_score": None, "source": "none"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = data.get("risk_score")
        if score is None:
            return {"risk_score": None, "source": "file_missing_score"}
        score = max(0.0, min(100.0, float(score)))
        return {"risk_score": score, "source": data.get("source", "file")}
    except Exception:
        return {"risk_score": None, "source": "file_error"}


def resolve_risk_mode(risk_score: float, cfg: Dict[str, float]) -> str:
    if risk_score >= cfg["RISK_CONSERVATIVE_MIN"]:
        return "conservative"
    if risk_score <= cfg["RISK_AGGRESSIVE_MAX"]:
        return "aggressive"
    return "neutral"


def apply_risk_mode(mode: str, base: Dict[str, float], cfg: Dict[str, float]) -> Dict[str, float]:
    tuned = dict(base)
    if mode == "aggressive":
        tuned["signal_score_threshold"] = cfg["AGG_SIGNAL_SCORE_THRESHOLD"]
        tuned["buy_score_threshold"] = cfg["AGG_BUY_SCORE_THRESHOLD"]
        tuned["sell_ratio"] = cfg["AGG_SELL_RATIO"]
        tuned["buy_ratio"] = cfg["AGG_BUY_RATIO"]
    elif mode == "conservative":
        tuned["signal_score_threshold"] = cfg["CONS_SIGNAL_SCORE_THRESHOLD"]
        tuned["buy_score_threshold"] = cfg["CONS_BUY_SCORE_THRESHOLD"]
        tuned["sell_ratio"] = cfg["CONS_SELL_RATIO"]
        tuned["buy_ratio"] = cfg["CONS_BUY_RATIO"]
    return tuned


def load_runtime_state(path: str) -> Dict:
    base = {
        "equity_peak": 0.0,
        "year": time.localtime().tm_year,
        "yearly_stop_count": 0,
        "halted": False,
        "halt_reason": "",
        "last_action_ts": 0.0,
        "last_signal_hash": "",
        "pending_order": None,
        "last_buy_stop_price": None,
        "last_buy_ts": 0.0,
        "partial_tp_done": [],
        "trailing_peak_price": 0.0,
        "day_key": time.strftime("%Y-%m-%d"),
        "day_start_equity": 0.0,
        "trades_today": 0,
    }
    if not os.path.exists(path):
        return base
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            base.update(loaded)
        return base
    except Exception:
        return base


def save_runtime_state(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_stop_loss_event(reasons: List[str]) -> bool:
    text = " | ".join(reasons)
    keys = ["entry_candle_stop", "loss_cut", "atr_trailing_break", "donchian_break"]
    return any(k in text for k in keys)


def place_order_with_retry(action_name: str, fn, *args, retries: int = 3, delay_sec: float = 1.0):
    last_err = None
    for i in range(retries):
        try:
            return fn(*args)
        except Exception as e:
            last_err = e
            logging.warning("%s failed (%s/%s): %s", action_name, i + 1, retries, e)
            time.sleep(delay_sec * (2 ** i))
    raise RuntimeError(f"{action_name} failed after retries: {last_err}")


def make_signal_hash(side: str, reasons: List[str], score: int, market: str) -> str:
    payload = f"{side}|{market}|{score}|{'|'.join(sorted(reasons))}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def verify_position_change(before_coin: float, before_krw: float, after_coin: float, after_krw: float, side: str) -> bool:
    if side == "buy":
        return (after_coin > before_coin) or (after_krw < before_krw)
    if side == "sell":
        return (after_coin < before_coin) or (after_krw > before_krw)
    return False


def notify_event(webhook_url: str, enabled_events: set, event_name: str, message: str) -> None:
    if not webhook_url or event_name not in enabled_events:
        return
    for attempt in range(2):
        try:
            requests.post(webhook_url, json={"event": event_name, "text": message}, timeout=5)
            return
        except Exception as e:
            if attempt == 1:
                logging.warning("notify failed(%s): %s", event_name, e)
            else:
                time.sleep(0.4)


def get_order_state(upbit, uuid: str):
    try:
        o = upbit.get_order(uuid)
        if isinstance(o, dict):
            return o.get("state", "unknown"), o
        if isinstance(o, list) and o:
            return o[0].get("state", "unknown"), o[0]
        return "unknown", o
    except Exception as e:
        return "error", {"error": str(e)}


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

    # ?〓낫???꾪꽣
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
    rr_min = get_env_float("MIN_RR", 2.5)
    rr_target_atr_mult = get_env_float("RR_TARGET_ATR_MULT", 2.0)
    spread_bps_max = get_env_float("SPREAD_BPS_MAX", 12)
    post_entry_cooldown_bars = get_env_int("POST_ENTRY_COOLDOWN_BARS", 3)
    partial_tp_levels = parse_float_list_csv(os.getenv("PARTIAL_TP_LEVELS", "1.0,2.0"), [1.0, 2.0])
    partial_tp_ratios = parse_float_list_csv(os.getenv("PARTIAL_TP_RATIOS", "0.3,0.3"), [0.3, 0.3])
    if len(partial_tp_levels) != len(partial_tp_ratios):
        logging.warning("PARTIAL_TP length mismatch | levels=%s ratios=%s -> truncating to min length", len(partial_tp_levels), len(partial_tp_ratios))
    n_tp = min(len(partial_tp_levels), len(partial_tp_ratios))
    partial_tp_levels = partial_tp_levels[:n_tp]
    partial_tp_ratios = partial_tp_ratios[:n_tp]
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
    max_trades_per_day = get_env_int("MAX_TRADES_PER_DAY", 20)
    max_position_krw = get_env_float("MAX_POSITION_KRW", 0)
    max_slippage_bps = get_env_float("MAX_SLIPPAGE_BPS", 30)
    alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    alert_events = set(x.strip() for x in os.getenv("ALERT_EVENTS", "order_sent,filled,rejected,halted,resume").split(",") if x.strip())
    pending_order_timeout_sec = get_env_int("PENDING_ORDER_TIMEOUT_SEC", 180)

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
    pending_order = runtime_state.get("pending_order")
    last_buy_stop_price = runtime_state.get("last_buy_stop_price")
    if last_buy_stop_price is not None:
        try:
            last_buy_stop_price = float(last_buy_stop_price)
        except Exception:
            last_buy_stop_price = None
    last_buy_ts = float(runtime_state.get("last_buy_ts", 0.0) or 0.0)
    partial_tp_done = set(str(x) for x in (runtime_state.get("partial_tp_done") or []))
    trailing_peak_price = float(runtime_state.get("trailing_peak_price", 0.0) or 0.0)
    prev_halted = bool(runtime_state.get("halted", False))

    logging.info("bot started | market=%s dry_run=%s entry_mode=%s", market, dry_run, entry_mode)

    while True:
        try:
            now = time.time()

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
                    if age_sec >= pending_order_timeout_sec:
                        notify_event(alert_webhook_url, alert_events, "rejected", f"pending_order_timeout side={pending_order.get('side')} age={age_sec}s state={st}")

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

            # 상태 저장(복구용 핵심 필드)
            runtime_state["last_action_ts"] = float(last_action_ts)
            runtime_state["last_signal_hash"] = last_signal_hash
            runtime_state["pending_order"] = pending_order
            runtime_state["last_buy_stop_price"] = last_buy_stop_price
            runtime_state["last_buy_ts"] = float(last_buy_ts)
            runtime_state["partial_tp_done"] = sorted(list(partial_tp_done))
            runtime_state["trailing_peak_price"] = float(trailing_peak_price)
            save_runtime_state(runtime_state_path, runtime_state)

            if runtime_state.get("halted", False):
                if not prev_halted:
                    notify_event(alert_webhook_url, alert_events, "halted", f"halted: {runtime_state.get('halt_reason', 'unknown')}")
                prev_halted = True
                logging.error("TRADING HALTED | reason=%s", runtime_state.get("halt_reason", "unknown"))
                time.sleep(check_seconds)
                continue
            elif prev_halted:
                notify_event(alert_webhook_url, alert_events, "resume", "trading resumed")
                prev_halted = False

            if coin_balance <= 0:
                last_buy_stop_price = None
                last_buy_ts = 0.0
                trailing_peak_price = 0.0
                partial_tp_done = set()
                active_box_high = None
                addon_done = False

            df = pyupbit.get_ohlcv(market, interval="minute60", count=320)
            if df is None or len(df) < 80:
                logging.warning("ohlcv unavailable or too short")
                time.sleep(check_seconds)
                continue

            df_mtf = pyupbit.get_ohlcv(market, interval=mtf_interval, count=max(mtf_lookback, 220))
            df_mtf_2 = pyupbit.get_ohlcv(market, interval=mtf_interval_2, count=max(mtf_lookback, 220))
            mtf_trend_ok = True

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

            mtf_trend_ok = _mtf_ok(df_mtf) and _mtf_ok(df_mtf_2)

            signal_score, signal_reasons, metrics = evaluate_signals(df, avg_buy_price, cfg, mtf_trend_ok=mtf_trend_ok)
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

            # 스프레드 필터 + 진입 후 봉 쿨다운
            spread_bps = get_spread_bps(market)
            if should_buy and spread_bps_max > 0 and spread_bps > spread_bps_max:
                should_buy = False
                buy_reasons.append(f"spread_block({spread_bps:.1f}>{spread_bps_max:.1f})")

            entry_cooldown_sec = max(0, post_entry_cooldown_bars) * 3600
            if should_buy and last_buy_ts > 0 and (time.time() - last_buy_ts) < entry_cooldown_sec:
                should_buy = False
                buy_reasons.append("post_entry_cooldown_block")

            # 진입봉 저점 이탈 손절
            if stop_by_entry_candle and coin_balance > 0 and last_buy_stop_price is not None and current_price < last_buy_stop_price:
                should_sell = True
                signal_reasons.append(f"entry_candle_stop({last_buy_stop_price:.0f})")

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
            emergency_sell = False
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

            if should_buy and max_position_krw > 0 and (coin_value_krw + (krw_balance * min(probe_entry_ratio if entry_mode == 'confirm_breakout' else active_buy_ratio, active_buy_ratio))) > max_position_krw:
                should_buy = False
                buy_reasons.append(f"max_position_block({max_position_krw:.0f})")

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

            # 분할 익절(1R,2R 등)
            partial_tp_executed = False
            if coin_balance > 0 and avg_buy_price > 0 and last_buy_stop_price is not None and last_buy_stop_price < avg_buy_price:
                risk_unit = avg_buy_price - last_buy_stop_price
                r_now = (current_price - avg_buy_price) / max(risk_unit, 1e-9)
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
                                    pending_order = {"uuid": order_uuid, "side": "sell_partial", "created_ts": now_ts}
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
                                partial_tp_executed = True
                        break

            if partial_tp_executed:
                runtime_state["last_action_ts"] = float(last_action_ts)
                runtime_state["last_signal_hash"] = last_signal_hash
                runtime_state["pending_order"] = pending_order
                runtime_state["last_buy_stop_price"] = last_buy_stop_price
                runtime_state["last_buy_ts"] = float(last_buy_ts)
                runtime_state["partial_tp_done"] = sorted(list(partial_tp_done))
                runtime_state["trailing_peak_price"] = float(trailing_peak_price)
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
                    stop_event = is_stop_loss_event(signal_reasons)
                    if stop_event:
                        runtime_state["yearly_stop_count"] = int(runtime_state.get("yearly_stop_count", 0)) + 1
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
                    else:
                        before_coin, before_krw = coin_balance, krw_balance
                        notify_event(alert_webhook_url, alert_events, "order_sent", f"sell sent {market} amount={sell_amount_coin:.8f}")
                        result = place_order_with_retry("sell_market_order", upbit.sell_market_order, market, sell_amount_coin)
                        logging.warning("sell_market_order result: %s", result)
                        order_uuid = result.get("uuid") if isinstance(result, dict) else None
                        pending_order = {"uuid": order_uuid, "side": "sell", "created_ts": now_ts} if order_uuid else None
                        time.sleep(1.0)
                        after_coin, _, _ = get_account_state(upbit, market)
                        after_krw = get_krw_balance(upbit)
                        filled = verify_position_change(before_coin, before_krw, after_coin, after_krw, "sell")
                        logging.warning("sell fill check | filled=%s before_coin=%.8f after_coin=%.8f", filled, before_coin, after_coin)
                        if filled:
                            last_action_ts = now_ts
                            last_signal_hash = signal_hash
                            runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                            notify_event(alert_webhook_url, alert_events, "filled", f"sell filled {market} price={current_price:.0f}")
                            pending_order = None
                        elif order_uuid is None:
                            notify_event(alert_webhook_url, alert_events, "rejected", f"sell rejected {market}")

            elif should_buy:
                base_ratio = probe_entry_ratio if entry_mode == "confirm_breakout" else active_buy_ratio
                buy_krw = krw_balance * min(base_ratio, active_buy_ratio)
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
                            last_buy_stop_price = last_low
                    else:
                        before_coin, before_krw = coin_balance, krw_balance
                        notify_event(alert_webhook_url, alert_events, "order_sent", f"buy sent {market} krw={buy_krw:.0f}")
                        result = place_order_with_retry("buy_market_order", upbit.buy_market_order, market, buy_krw)
                        logging.warning("buy_market_order result: %s", result)
                        order_uuid = result.get("uuid") if isinstance(result, dict) else None
                        pending_order = {"uuid": order_uuid, "side": "buy", "created_ts": now_ts} if order_uuid else None
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
                                last_buy_stop_price = last_low
                            pending_order = None
                        elif order_uuid is None:
                            notify_event(alert_webhook_url, alert_events, "rejected", f"buy rejected {market}")

            # 박스 돌파 후 리테스트(지지 확인) 추가진입
            if enable_retest_addon and coin_balance > 0 and active_box_high is not None and not addon_done:
                retest_band = active_box_high * box_buffer_pct
                retest_hit = abs(current_price - active_box_high) <= retest_band
                retest_hold = (last_close >= active_box_high) and (last_close >= metrics["ma20"])
                if retest_hit and retest_hold and krw_balance >= min_buy_krw:
                    addon_krw = krw_balance * addon_ratio
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
                        runtime_state["trades_today"] = int(runtime_state.get("trades_today", 0)) + 1
                        addon_done = True

            runtime_state["last_action_ts"] = float(last_action_ts)
            runtime_state["last_signal_hash"] = last_signal_hash
            runtime_state["pending_order"] = pending_order
            runtime_state["last_buy_stop_price"] = last_buy_stop_price
            runtime_state["last_buy_ts"] = float(last_buy_ts)
            runtime_state["partial_tp_done"] = sorted(list(partial_tp_done))
            runtime_state["trailing_peak_price"] = float(trailing_peak_price)
            save_runtime_state(runtime_state_path, runtime_state)

            time.sleep(check_seconds)

        except Exception as e:
            pending_order = None
            runtime_state["pending_order"] = None
            save_runtime_state(runtime_state_path, runtime_state)
            logging.exception("loop error: %s", e)
            time.sleep(max(5, check_seconds))


if __name__ == "__main__":
    run()
