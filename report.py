import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def in_range(ts, start, end):
    dt = datetime.fromtimestamp(ts)
    return start <= dt < end


def summarize(rows):
    side = Counter(r.get("side", "") for r in rows)
    stop_rows = [r for r in rows if r.get("stop_event")]
    reason_counter = Counter()
    for r in stop_rows:
        for x in r.get("reasons", []):
            reason_counter[x] += 1

    tips = []
    top = [x for x, _ in reason_counter.most_common(3)]
    if any("entry_candle_stop" in t for t in top):
        tips.append("吏꾩엯 怨쇰? 媛?μ꽦: PROBE_ENTRY_RATIO瑜???텛怨?BREAKOUT_BODY_ATR_MULT瑜??믪씠湲?)
    if any("sideways_filter" in t for t in top):
        tips.append("?〓낫 怨쇰ℓ留?媛?μ꽦: SIDEWAYS_CROSS_THRESHOLD ?곹뼢, BOX_LOOKBACK ?뺣?")
    if any("ma22_exit" in t or "ma22" in t for t in top):
        tips.append("泥?궛 怨쇰? 媛?μ꽦: MA_EXIT_PERIOD 22->24~30 ?뚯뒪??)
    if not tips:
        tips.append("理쒓렐 濡쒓렇 湲곗? ?뚮씪誘명꽣 1媛쒖뵫留??뚰룺 議곗젙?섎ŉ 鍮꾧탳")

    return {
        "count": len(rows),
        "buy": side.get("buy", 0),
        "sell": side.get("sell", 0),
        "buy_addon": side.get("buy_addon", 0),
        "stop_count": len(stop_rows),
        "top_stop_reasons": reason_counter.most_common(5),
        "tips": tips,
    }


def fmt(title, s):
    lines = [f"[{title}]", f"trades={s['count']} buy={s['buy']} addon={s['buy_addon']} sell={s['sell']} stop={s['stop_count']}"]
    if s["top_stop_reasons"]:
        lines.append("top_stop_reasons: " + ", ".join(f"{k}({v})" for k, v in s["top_stop_reasons"]))
    lines.append("tuning_tips: " + " / ".join(s["tips"]))
    return "\n".join(lines)


if __name__ == "__main__":
    # usage: python report.py daily|weekly|monthly [jsonl_path]
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    path = sys.argv[2] if len(sys.argv) > 2 else "logs/trade_journal.jsonl"

    now = datetime.now()
    if mode == "daily":
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
    elif mode == "weekly":
        # monday-start week summary for previous 7d
        end = now
        start = now - timedelta(days=7)
    elif mode == "monthly":
        start = datetime(now.year, now.month, 1)
        end = now
    else:
        raise SystemExit("mode must be daily|weekly|monthly")

    rows = [r for r in load_jsonl(path) if in_range(r.get("ts", 0), start, end)]
    s = summarize(rows)
    print(fmt(mode, s))
