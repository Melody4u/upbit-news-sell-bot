import os
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import requests
import pyupbit
import pandas as pd
from dotenv import load_dotenv


NEGATIVE_KEYWORDS = [
    "hack", "해킹", "제재", "금지", "폐쇄", "소송", "파산", "유의종목", "상장폐지", "악재", "규제", "investigation"
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


def evaluate_signals(df: pd.DataFrame, avg_buy_price: float, cfg: Dict[str, float]) -> Tuple[int, List[str], Dict[str, float]]:
    close = df["close"]
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

    score = 0
    reasons: List[str] = []

    # 1) 추세 약화: EMA 데드크로스 + ADX 강추세
    trend_break = bool(ema_fast.iloc[-1] < ema_slow.iloc[-1] and adx.iloc[-1] >= cfg["ADX_MIN"])
    if trend_break:
        score += 2
        reasons.append("trend_break(ema_fast<ema_slow & adx_high)")

    # 2) 채널 하단 이탈
    donchian_break = bool(current < donchian_low.iloc[-1]) if pd.notna(donchian_low.iloc[-1]) else False
    if donchian_break:
        score += 2
        reasons.append("donchian_break")

    # 3) ATR 기반 트레일링 스탑
    recent_peak = float(high.tail(int(cfg["DONCHIAN_PERIOD"])).max())
    atr_stop = recent_peak - (cfg["ATR_MULT"] * float(atr.iloc[-1]))
    atr_break = bool(current < atr_stop)
    if atr_break:
        score += 1
        reasons.append("atr_trailing_break")

    # 4) 과열 후 꺾임 (RSI 하향 + 거래량 스파이크)
    rsi_rollover = bool(rsi.iloc[-2] > cfg["RSI_OVERBOUGHT"] and rsi.iloc[-1] < rsi.iloc[-2])
    vol_spike = bool(vol.iloc[-1] > (vol_ma.iloc[-1] * cfg["VOL_SPIKE_MULT"])) if pd.notna(vol_ma.iloc[-1]) else False
    if rsi_rollover and vol_spike:
        score += 1
        reasons.append("rsi_rollover_with_volume_spike")

    # 5) 평균단가 기반 리스크 컷
    loss_cut = avg_buy_price > 0 and current < avg_buy_price * (1 - cfg["LOSS_CUT_PCT"])
    if loss_cut:
        score += 2
        reasons.append("loss_cut")

    metrics = {
        "price": current,
        "ema_fast": float(ema_fast.iloc[-1]),
        "ema_slow": float(ema_slow.iloc[-1]),
        "adx": float(adx.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "atr_stop": float(atr_stop),
        "donchian_low": float(donchian_low.iloc[-1]) if pd.notna(donchian_low.iloc[-1]) else 0.0,
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
    min_sell_krw = get_env_float("MIN_SELL_KRW", 5000)
    check_seconds = get_env_int("CHECK_SECONDS", 30)

    brave_key = os.getenv("BRAVE_API_KEY", "")
    news_query = os.getenv("NEWS_QUERY", "bitcoin OR btc")
    news_country = os.getenv("NEWS_COUNTRY", "KR")
    news_lang = os.getenv("NEWS_LANG", "ko")
    news_result_count = get_env_int("NEWS_RESULT_COUNT", 5)
    news_interval_seconds = get_env_int("NEWS_INTERVAL_SECONDS", 3600)
    negative_news_threshold = get_env_int("NEGATIVE_NEWS_THRESHOLD", 2)

    signal_score_threshold = get_env_int("SIGNAL_SCORE_THRESHOLD", 3)

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
        "VOL_PERIOD": get_env_int("VOL_PERIOD", 20),
        "VOL_SPIKE_MULT": get_env_float("VOL_SPIKE_MULT", 1.8),
        "LOSS_CUT_PCT": get_env_float("LOSS_CUT_PCT", 0.02),
    }

    if not access or not secret:
        raise RuntimeError("UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY를 .env에 입력하세요.")

    upbit = pyupbit.Upbit(access, secret)
    news_state = NewsState()

    logging.info("bot started | market=%s dry_run=%s", market, dry_run)

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
            coin_value_krw = coin_balance * current_price

            if coin_balance <= 0:
                logging.info("no position | market=%s", market)
                time.sleep(check_seconds)
                continue

            df = pyupbit.get_ohlcv(market, interval="minute60", count=220)
            if df is None or len(df) < 80:
                logging.warning("ohlcv unavailable or too short")
                time.sleep(check_seconds)
                continue

            signal_score, signal_reasons, metrics = evaluate_signals(df, avg_buy_price, cfg)

            news_sell = news_state.negative_score >= negative_news_threshold
            if news_sell:
                signal_score += 1
                signal_reasons.append(f"negative_news(score={news_state.negative_score})")

            should_sell = signal_score >= signal_score_threshold

            logging.info(
                "tick | price=%.0f value=%.0fKRW score=%s/%s adx=%.1f rsi=%.1f news=%s",
                current_price,
                coin_value_krw,
                signal_score,
                signal_score_threshold,
                metrics["adx"],
                metrics["rsi"],
                news_state.negative_score,
            )

            if should_sell:
                sell_amount_coin = coin_balance * sell_ratio
                sell_value_krw = sell_amount_coin * current_price

                if sell_value_krw < min_sell_krw:
                    logging.warning("sell skipped | below min_sell_krw: %.0f < %.0f", sell_value_krw, min_sell_krw)
                else:
                    logging.warning("SELL SIGNAL | score=%s reasons=%s", signal_score, ", ".join(signal_reasons))
                    if dry_run:
                        logging.warning("DRY_RUN sell | market=%s amount=%.8f (~%.0fKRW)", market, sell_amount_coin, sell_value_krw)
                    else:
                        result = upbit.sell_market_order(market, sell_amount_coin)
                        logging.warning("sell_market_order result: %s", result)

            time.sleep(check_seconds)

        except Exception as e:
            logging.exception("loop error: %s", e)
            time.sleep(max(5, check_seconds))


if __name__ == "__main__":
    run()
