from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd


def dump_csv(base_path: str, legs: list[dict], positions: list[dict], debug: list[dict] | None = None) -> None:
    p = Path(base_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if legs:
        pd.DataFrame(legs).to_csv(str(p).replace(".csv", "_legs.csv"), index=False, encoding="utf-8-sig")
    if positions:
        pd.DataFrame(positions).to_csv(str(p).replace(".csv", "_pos.csv"), index=False, encoding="utf-8-sig")
    if debug:
        pd.DataFrame(debug).to_csv(str(p).replace(".csv", "_debug.csv"), index=False, encoding="utf-8-sig")
