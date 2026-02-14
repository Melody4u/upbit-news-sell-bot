"""V2 backtest entrypoint (research-only).

This is intentionally minimal for the first boot:
- Uses the existing data fetch + indicator computation from backtest_date_range_phaseA.py
- Adds v2 regime + structured dumps incrementally.

Next: move simulation logic into v2/sim.py and keep this as a thin CLI.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import backtest_date_range_phaseA as v1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", type=str, default="KRW-BTC")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--cache-dir", type=str, default="ohlcv_cache")
    ap.add_argument("--dump-trades-csv", type=str, default="")
    args = ap.parse_args()

    start = pd.Timestamp(datetime.fromisoformat(args.start))
    end = pd.Timestamp(datetime.fromisoformat(args.end))

    # For now call v1 simulate with caching + dumps. V2 will replace this gradually.
    res = v1.simulate(
        args.market,
        start,
        end,
        cache_dir=str(args.cache_dir),
        dump_trades_csv=str(args.dump_trades_csv),
    )
    print(res)


if __name__ == "__main__":
    main()
