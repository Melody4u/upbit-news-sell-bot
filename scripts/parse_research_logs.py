import ast
import json
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def _read_text(p: pathlib.Path) -> str:
    # handle utf-8 / utf-8-sig / utf-16 gracefully
    raw = p.read_bytes()
    # Try utf-8-sig first to strip BOM (U+FEFF) which breaks ast.literal_eval
    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp949"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def parse_out(path: pathlib.Path):
    txt = _read_text(path).strip()
    if not txt:
        return None
    # out logs are python dict repr
    try:
        return ast.literal_eval(txt)
    except Exception:
        # fallback: try json
        try:
            return json.loads(txt)
        except Exception:
            return {"_parse_error": True, "_path": str(path), "_head": txt[:200]}


def main():
    # Ensure Windows console can print UTF-8 (avoid cp949 UnicodeEncodeError)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    pattern = sys.argv[1] if len(sys.argv) > 1 else "research_KRW-BTC_"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    outs = sorted(LOGS.glob(f"{pattern}*.out.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    rows = []
    for p in outs:
        d = parse_out(p)
        if not d:
            continue
        d = dict(d)
        d["_file"] = p.name
        d["_mtime"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        rows.append(d)

    print(json.dumps({"count": len(rows), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
