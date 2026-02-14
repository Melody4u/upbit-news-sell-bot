"""V2 backtest entrypoint (research-only).

This is intentionally minimal for the first boot:
- Uses the existing data fetch + indicator computation from backtest_date_range_phaseA.py
- Adds v2 regime + structured dumps incrementally.

Next: move simulation logic into v2/sim.py and keep this as a thin CLI.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import backtest_date_range_phaseA as v1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", type=str, default="KRW-BTC")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--cache-dir", type=str, default="ohlcv_cache")
    ap.add_argument("--dump-trades-csv", type=str, default="")
    ap.add_argument("--preset", type=str, default=str(Path(__file__).resolve().parent / "v2" / "preset.json"))
    args = ap.parse_args()

    start = pd.Timestamp(datetime.fromisoformat(args.start))
    end = pd.Timestamp(datetime.fromisoformat(args.end))

    preset_path = Path(args.preset)
    preset = {}
    try:
        preset = json.loads(preset_path.read_text(encoding="utf-8"))
    except Exception:
        preset = {}

    sim_kwargs = dict(preset.get("simulate_kwargs", {}) or {})
    # CLI always wins
    sim_kwargs.update({
        "cache_dir": str(args.cache_dir),
        "dump_trades_csv": str(args.dump_trades_csv),
    })

    # For now call v1 simulate with caching + dumps. V2 will replace this gradually.
    res = v1.simulate(
        args.market,
        start,
        end,
        **sim_kwargs,
    )
    print(res)


if __name__ == "__main__":
    main()
