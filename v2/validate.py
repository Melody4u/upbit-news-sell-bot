from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    ok: bool
    msg: str


def require_keys(d: dict, keys: list[str]) -> CheckResult:
    missing = [k for k in keys if k not in d]
    if missing:
        return CheckResult(False, f"missing keys: {missing}")
    return CheckResult(True, "ok")


def schema_legs_min(columns: list[str]) -> CheckResult:
    need = ["pos_id", "entry_ts", "exit_ts", "exit_reason", "r"]
    miss = [c for c in need if c not in columns]
    if miss:
        return CheckResult(False, f"legs missing cols: {miss}")
    return CheckResult(True, "ok")
