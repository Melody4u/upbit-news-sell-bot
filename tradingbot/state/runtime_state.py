import os
import json
import time
from typing import Dict


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
