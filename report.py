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

    # 운영 KPI: 차단 사유/타임아웃/부분익절 비율
    reason_counter = Counter()
    block_counter = Counter()
    timeout_count = 0
    for r in rows:
        reasons = r.get("reasons", []) or []
        for x in reasons:
            reason_counter[x] += 1
            if isinstance(x, str) and (x.endswith("_block") or "_block(" in x or "blocked" in x):
                block_counter[x] += 1
            if isinstance(x, str) and "pending_order_timeout" in x:
                timeout_count += 1

    partial_tp_count = side.get("sell_partial", 0)

    tips = []
    top = [x for x, _ in reason_counter.most_common(5)]
    if any("entry_candle_stop" in t for t in top):
        tips.append("진입 직후 손절 빈도 높음: PROBE_ENTRY_RATIO↓ 또는 BREAKOUT_BODY_ATR_MULT↑")
    if any("sideways_filter" in t for t in top):
        tips.append("횡보 거래 과다 가능성: SIDEWAYS_CROSS_THRESHOLD↑")
    if any("rr_block" in t for t in top):
        tips.append("진입이 과도하게 막히면 MIN_RR/목표 ATR 배수 재점검")
    if not tips:
        tips.append("단일 변경 원칙으로 파라미터 1개씩만 조정하고 재검증하세요")

    return {
        "count": len(rows),
        "buy": side.get("buy", 0),
        "sell": side.get("sell", 0),
        "buy_addon": side.get("buy_addon", 0),
        "sell_partial": partial_tp_count,
        "stop_count": len(stop_rows),
        "pending_timeout_count": timeout_count,
        "top_reasons": reason_counter.most_common(8),
        "block_reasons": block_counter.most_common(8),
        "tips": tips,
    }


def fmt(title, s):
    lines = [
        f"[{title}]",
        f"trades={s['count']} buy={s['buy']} addon={s['buy_addon']} partial_tp={s['sell_partial']} sell={s['sell']} stop={s['stop_count']}",
        f"ops_kpi: pending_timeout={s['pending_timeout_count']}",
    ]
    if s["top_reasons"]:
        lines.append("top_reasons: " + ", ".join(f"{k}({v})" for k, v in s["top_reasons"]))
    if s["block_reasons"]:
        lines.append("block_reasons: " + ", ".join(f"{k}({v})" for k, v in s["block_reasons"]))
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
