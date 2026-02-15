"""3-hour patchnote report (research-only).

- Runs arch_autoloop once (baseline+candidate tests) to generate a fresh run log line.
- Summarizes the last 3 hours from tmp/arch_loop/runs.jsonl.
- Includes git patch-notes from last 3 hours.
- Maintains a simple version scheme in tmp/arch_loop/state.json:
  v2_YY.M.DD.<major>.<minor>
  - minor increments every patchnote run
  - major increments only when a big event is detected (ACCEPT with jump>=2 or score delta >= threshold)

This script prints a single message suitable for chat.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
AUTOLOOP = REPO / "scripts" / "arch_autoloop.py"
GEN_CANDS = REPO / "scripts" / "generate_candidates.py"
DUMPROOT = REPO / "tmp" / "arch_loop"
RUNS_LOG = DUMPROOT / "runs.jsonl"
STATE = DUMPROOT / "state.json"


def now_kst() -> datetime:
    return datetime.now()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO), check=True)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_version(v: str) -> tuple[int, int, int, int, int]:
    # v2_26.2.15.1.1 -> (26,2,15,1,1)
    try:
        s = v.replace("v2_", "")
        a, b, c, major, minor = s.split(".")
        return int(a), int(b), int(c), int(major), int(minor)
    except Exception:
        return (0, 0, 0, 1, 1)


def fmt_version(y: int, m: int, d: int, major: int, minor: int) -> str:
    return f"v2_{y}.{m}.{d}.{major}.{minor}"


def load_runs(since: datetime) -> list[dict]:
    if not RUNS_LOG.exists():
        return []
    out = []
    for line in RUNS_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            ts = datetime.fromisoformat(obj.get("ts"))
        except Exception:
            continue
        if ts >= since:
            out.append(obj)
    return out


def git_notes(since: datetime) -> list[str]:
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    cmd = ["git", "log", f"--since={since_str}", "--pretty=format:%h %s", "--no-merges"]
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    txt = (p.stdout or "").strip()
    if not txt:
        return []
    return [ln.strip() for ln in txt.splitlines() if ln.strip()][:8]


def main():
    now = now_kst()
    since = now - timedelta(hours=3)

    # 1) Baseline run to refresh evidence.json
    env = dict(**{**dict(**subprocess.os.environ), "PYTHONIOENCODING": "utf-8"})
    subprocess.run([str(PY), str(AUTOLOOP), "--max-candidates", "0"], cwd=str(REPO), check=True, env=env)

    # 2) Generate candidates.json from evidence.json (pipeline proof; later replaced by Sisyphus)
    subprocess.run([str(PY), str(GEN_CANDS)], cwd=str(REPO), check=True, env=env)

    # 3) Run autoloop to test candidates and possibly accept
    subprocess.run([str(PY), str(AUTOLOOP)], cwd=str(REPO), check=True, env=env)

    # 2) Summarize last 3h
    runs = load_runs(since)
    st = load_state()

    # init version if missing
    v = st.get("version")
    if not v:
        v = fmt_version(int(str(now.year)[-2:]), now.month, now.day, 1, 1)

    y, mo, da, major, minor = parse_version(v)

    accepts = [r for r in runs if r.get("applied")]
    patches = sum(int(r.get("runs", {}).get("patch", 0) or 0) for r in runs)
    rollbacks = sum(int(r.get("runs", {}).get("rollback", 0) or 0) for r in runs)
    backtests = sum(int(r.get("runs", {}).get("backtest", 0) or 0) for r in runs)

    # detect big event
    big = False
    for r in accepts:
        jump = int(r.get("step", {}).get("jump", 0) or 0)
        dscore = float(r.get("delta", {}).get("score", 0.0) or 0.0)
        if jump >= 2 or dscore >= 5.0:
            big = True
            break

    if big:
        major += 1
        minor = 1
    else:
        minor += 1

    # rollover date changes -> reset major/minor
    cur_y = int(str(now.year)[-2:])
    if (y, mo, da) != (cur_y, now.month, now.day):
        y, mo, da = cur_y, now.month, now.day
        major, minor = 1, 1

    v2 = fmt_version(y, mo, da, major, minor)
    st["version"] = v2
    st["version_last_ts"] = now.isoformat(timespec="seconds")
    save_state(st)

    last = runs[-1] if runs else {}
    focus = last.get("focus", "n/a")
    step = last.get("step", {})

    print(f"[{v2}] {now.strftime('%Y-%m-%d %H:%M')} Patchnote (3h)")
    print(f"- Runs: {len(runs)} / backtest {backtests} / patch {patches} / rollback {rollbacks} / accept {len(accepts)}")
    print(f"- Focus: {focus}")
    if step:
        print(f"- Step: S{step.get('from',0)} -> S{step.get('to',0)} (+{step.get('jump',0)})")

    # show last metrics
    m = last.get("metrics", {})
    if m:
        h2 = m.get("H2", {})
        q4 = m.get("Q4", {})
        print(f"- Metrics: H2 {h2.get('return_pct')}% / mdd {h2.get('mdd_pct')} / entries {h2.get('entries')} | Q4 {q4.get('return_pct')}% / mdd {q4.get('mdd_pct')} / entries {q4.get('entries')}")

    # patchnotes from git
    notes = git_notes(since)
    if notes:
        print("- Patch notes:")
        for ln in notes[:5]:
            print(f"  · {ln}")

    # next
    print("- Next: 시시퍼스 candidates.json 생성이 실제로 반영되는지(빈 파일 방지)부터 고정")

    # Kiwi one-liner (action-needed / permission-needed)
    need = []
    # memory_search currently needs Gemini embeddings key
    need.append("memory_search 복구용 Gemini API 키 발급(희찬) 대기")
    # candidates file empty -> Sisyphus generation not yet wired
    cand_path = DUMPROOT / "candidates.json"
    try:
        if cand_path.exists():
            obj = json.loads(cand_path.read_text(encoding="utf-8"))
            if not (obj.get("candidates") or []):
                need.append("candidates.json이 비어있음 → 시시퍼스 후보 생성 연결 필요")
    except Exception:
        need.append("candidates.json 파싱 실패")

    if need:
        print(f"- Kiwi 한줄평: {need[0]}")
    else:
        print("- Kiwi 한줄평: 오늘은 자동으로 굴러가고 있음 — 남은 건 ‘채택 1회’만 만들면 됨")


if __name__ == "__main__":
    main()
