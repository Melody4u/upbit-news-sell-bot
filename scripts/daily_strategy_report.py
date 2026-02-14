"""Daily strategy report (research-only).

Reads tmp/arch_loop/runs.jsonl (append-only) and summarizes the last 24h.
Also includes patch-note style git commit summary from the last 24h.

Output is designed to be posted to chat directly.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUMPROOT = REPO / "tmp" / "arch_loop"
RUNS_LOG = DUMPROOT / "runs.jsonl"


def now_kst() -> datetime:
    # System is already Asia/Seoul; keep naive local time.
    return datetime.now()


def load_runs(since: datetime) -> list[dict]:
    if not RUNS_LOG.exists():
        return []
    out: list[dict] = []
    for line in RUNS_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        try:
            ts = datetime.fromisoformat(obj.get("ts"))
        except Exception:
            continue
        if ts >= since:
            out.append(obj)
    return out


def git_last_24h(since: datetime) -> list[str]:
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    cmd = [
        "git",
        "log",
        f"--since={since_str}",
        "--pretty=format:%h %s",
        "--no-merges",
    ]
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    txt = (p.stdout or "").strip()
    if not txt:
        return []
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return lines[:12]


def fmt_pct(x) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)


def main():
    end = now_kst()
    since = end - timedelta(hours=24)

    runs = load_runs(since)

    accepts = [r for r in runs if r.get("applied")]
    patches = sum(int(r.get("runs", {}).get("patch", 0) or 0) for r in runs)
    rollbacks = sum(int(r.get("runs", {}).get("rollback", 0) or 0) for r in runs)
    backtests = sum(int(r.get("runs", {}).get("backtest", 0) or 0) for r in runs)

    last = runs[-1] if runs else {}
    focus = last.get("focus", "n/a")

    # Best improvement snapshot (by score delta if present)
    best = None
    best_key = -1e18
    for r in runs:
        d = float(r.get("delta", {}).get("score", 0.0) or 0.0)
        if d > best_key:
            best_key = d
            best = r

    # Step progression
    step_from = runs[0].get("step", {}).get("from") if runs else None
    step_to = last.get("step", {}).get("to") if runs else None

    print(f"[daily] {end.strftime('%Y-%m-%d')} 22:00 전략 리포트")
    print(f"- 요약: 24h runs={len(runs)} / backtest={backtests} / patch={patches} / rollback={rollbacks} / accept={len(accepts)}")
    if step_from is not None and step_to is not None:
        print(f"- 진행도: Step S{step_from} → S{step_to}")
    print(f"- 현재 병목(Focus): {focus}")

    if best:
        h2 = best.get("metrics", {}).get("H2", {})
        q4 = best.get("metrics", {}).get("Q4", {})
        print(
            "- 베스트 시도: "
            f"{best.get('action','')} | "
            f"Δscore={best.get('delta',{}).get('score',0):.2f} | "
            f"H2 {fmt_pct(h2.get('return_pct'))}, mdd {h2.get('mdd_pct')} | "
            f"Q4 {fmt_pct(q4.get('return_pct'))}, mdd {q4.get('mdd_pct')}"
        )

    # Strategy-mode suggestions (deterministic heuristics)
    print("- 전략 관점: rem 손실이 계속 TOP이면 ‘잔여(rem)=옵션’ 취급 → risk-off 잔여축소/시간청산/BE-정책을 우선 축으로 유지")
    print("- 워크플로우 관점: ACCEPT가 0이면 Step을 더 잘게(서브스텝) 쪼개서 ‘작은 승리 누적’부터 복구")

    commits = git_last_24h(since)
    if commits:
        print("- 패치노트(최근 커밋):")
        for ln in commits[:6]:
            print(f"  · {ln}")
    else:
        print("- 패치노트: (지난 24h 커밋 없음)")

    print("- 다음 24h 제안: 시시퍼스 candidates 생성이 실제로 파일에 반영되는지(빈 candidates.json 방지)부터 먼저 고정")


if __name__ == "__main__":
    main()
