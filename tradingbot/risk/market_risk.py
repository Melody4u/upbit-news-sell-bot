import os
import json
from typing import Dict


def load_market_risk(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"risk_score": None, "source": "none"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"risk_score": None, "source": "file_invalid"}
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
