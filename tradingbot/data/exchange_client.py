import time
from typing import Tuple

import pyupbit


def get_account_state(upbit, market: str) -> Tuple[float, float, float]:
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
