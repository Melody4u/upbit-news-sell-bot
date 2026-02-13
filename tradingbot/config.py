import os
from typing import List


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


def parse_float_list_csv(s: str, default: List[float]) -> List[float]:
    try:
        vals = [float(x.strip()) for x in (s or "").split(",") if x.strip()]
        return vals if vals else default
    except Exception:
        return default
