import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_opencode() -> str | None:
    candidates = [
        "opencode",
        "opencode.cmd",
        os.path.expandvars(r"%APPDATA%\npm\opencode.cmd"),
    ]
    for c in candidates:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Safe wrapper for `opencode run` (avoids shell quoting issues)")
    p.add_argument("-m", "--model", required=True, help="Model id (e.g. opencode/claude-sonnet-4-5)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-p", "--prompt", help="Prompt text")
    src.add_argument("-f", "--file", help="Prompt file path (utf-8)")
    p.add_argument("--timeout", type=int, default=300, help="Timeout seconds (default: 300, use 0 for no timeout)")
    args = p.parse_args()

    prompt = args.prompt
    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")

    opencode_bin = resolve_opencode()
    if not opencode_bin:
        print("[opencode-run-safe] `opencode` command not found in PATH", file=sys.stderr)
        return 127

    try:
        cp = subprocess.run(
            [opencode_bin, "run", "-m", args.model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=(None if args.timeout <= 0 else args.timeout),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        print("[opencode-run-safe] timeout expired", file=sys.stderr)
        return 124
    except FileNotFoundError:
        print("[opencode-run-safe] `opencode` command not found in PATH", file=sys.stderr)
        return 127

    if cp.stdout:
        print(cp.stdout, end="")
    if cp.stderr:
        print(cp.stderr, end="", file=sys.stderr)
    return cp.returncode


if __name__ == "__main__":
    raise SystemExit(main())
