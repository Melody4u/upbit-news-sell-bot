import time
import logging
import math

import pyupbit


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


def floor_dec(x: float, decimals: int) -> float:
    p = 10 ** int(decimals)
    return math.floor(float(x) * p) / p


def get_best_ask(market: str) -> float:
    ob = pyupbit.get_orderbook(market)
    ob0 = ob[0] if isinstance(ob, list) and ob else ob
    return float(ob0["orderbook_units"][0]["ask_price"])


def place_buy_entry_order(
    upbit,
    market: str,
    buy_krw: float,
    *,
    mode: str,
    offset_bps: float,
    timeout_sec: int,
    fee_rate: float,
    vol_decimals: int,
    price_tick_method: str,
    min_buy_krw: float,
):
    """Place buy entry order.

    mode:
      - market: buy_market_order(KRW)
      - limit: buy_limit_order(price, volume)

    Returns: (result, pending_order)
    """
    mode = (mode or "market").lower()

    if float(buy_krw) < float(min_buy_krw):
        raise RuntimeError(f"buy_krw below min_buy_krw ({buy_krw:.0f} < {min_buy_krw:.0f})")

    if mode == "market":
        result = upbit.buy_market_order(market, buy_krw)
        uuid = result.get("uuid") if isinstance(result, dict) else None
        pending = {"uuid": uuid, "side": "buy", "kind": "buy_market", "created_ts": time.time(), "timeout_notified": False} if uuid else None
        return result, pending

    if mode != "limit":
        raise RuntimeError(f"unsupported BUY_ENTRY_ORDER_MODE={mode}")

    ask = get_best_ask(market)
    raw_price = float(ask) * (1.0 + (float(offset_bps) / 10000.0))
    limit_price = float(pyupbit.get_tick_size(raw_price, method=str(price_tick_method or "ceil")))

    budget = float(buy_krw) * (1.0 - float(fee_rate))
    vol = floor_dec(budget / max(limit_price, 1e-12), int(vol_decimals))
    if vol <= 0:
        raise RuntimeError("computed volume <= 0")

    result = upbit.buy_limit_order(market, limit_price, vol)
    uuid = result.get("uuid") if isinstance(result, dict) else None
    pending = None
    if uuid:
        pending = {
            "uuid": uuid,
            "side": "buy",
            "kind": "buy_limit",
            "market": market,
            "created_ts": time.time(),
            "timeout_sec": int(timeout_sec),
            "timeout_notified": False,
            "limit_price": float(limit_price),
            "limit_volume": float(vol),
            "krw_budget": float(buy_krw),
            "cancel_requested": False,
        }

    return result, pending


def verify_position_change(before_coin: float, before_krw: float, after_coin: float, after_krw: float, side: str) -> bool:
    if side == "buy":
        return (after_coin > before_coin) or (after_krw < before_krw)
    if side == "sell":
        return (after_coin < before_coin) or (after_krw > before_krw)
    return False
