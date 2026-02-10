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
        tips.append("진입 직후 손절 빈도 높음: PROBE_ENTRY_RATIO를 낮추거나 BREAKOUT_BODY_ATR_MULT를 높여보세요")
    if any("sideways_filter" in t for t in top):
        tips.append("횡보 구간 거래 과다 가능성: SIDEWAYS_CROSS_THRESHOLD 상향, BOX_LOOKBACK 조정")
    if any("ma22_exit" in t or "ma22" in t for t in top):
        tips.append("청산이 너무 빠를 수 있음: MA_EXIT_PERIOD를 22 -> 24~30 범위로 테스트")
    if not tips:
        tips.append("최근 로그 기준으로 파라미터 1개씩만 조정하고 결과를 비교하세요")

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
