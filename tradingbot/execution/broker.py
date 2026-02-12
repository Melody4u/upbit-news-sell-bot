import time
import logging


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


def verify_position_change(before_coin: float, before_krw: float, after_coin: float, after_krw: float, side: str) -> bool:
    if side == "buy":
        return (after_coin > before_coin) or (after_krw < before_krw)
    if side == "sell":
        return (after_coin < before_coin) or (after_krw > before_krw)
    return False
