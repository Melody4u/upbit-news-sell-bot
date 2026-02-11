import os
from dotenv import load_dotenv

from backtest_eth_iter import SimConfig, run_with_target_n


def summarize(name, r):
    return {
        "name": name,
        "days": r.get("days"),
        "trades": r.get("trades"),
        "win_rate_pct": r.get("win_rate_pct"),
        "profit_factor": r.get("profit_factor"),
        "expectancy_r": r.get("expectancy_r"),
        "avg_win_r": r.get("avg_win_r"),
        "avg_loss_r": r.get("avg_loss_r"),
        "median_hold_bars": r.get("median_hold_bars"),
        "mdd_pct": r.get("mdd_pct"),
        "integrity_ok": r.get("integrity_ok"),
        "block_ratio_pct": r.get("block_ratio_pct"),
    }


if __name__ == "__main__":
    load_dotenv()
    min_rr = float(os.getenv("MIN_RR", "3.0"))

    # 고정: trend loosen (현재 최우선 후보)
    base = SimConfig(min_rr=min_rr, trend_mode="loose", breakout_mode="strict", atr_min_pct=0.7, cooldown_bars=3)

    # 1변수 변경 A/B
    cand_atr_09 = SimConfig(min_rr=min_rr, trend_mode="loose", breakout_mode="strict", atr_min_pct=0.9, cooldown_bars=3)
    cand_atr_10 = SimConfig(min_rr=min_rr, trend_mode="loose", breakout_mode="strict", atr_min_pct=1.0, cooldown_bars=3)
    cand_cd_4 = SimConfig(min_rr=min_rr, trend_mode="loose", breakout_mode="strict", atr_min_pct=0.7, cooldown_bars=4)

    print("=== target N>=30 ===")
    r_base_30 = run_with_target_n(base, min_n=30)
    r_atr09_30 = run_with_target_n(cand_atr_09, min_n=30)
    r_atr10_30 = run_with_target_n(cand_atr_10, min_n=30)
    r_cd4_30 = run_with_target_n(cand_cd_4, min_n=30)

    print(summarize("base_trend_loose", r_base_30))
    print(summarize("atr_min_0.9", r_atr09_30))
    print(summarize("atr_min_1.0", r_atr10_30))
    print(summarize("cooldown_4", r_cd4_30))

    print("\n=== target N>=50 ===")
    r_base_50 = run_with_target_n(base, min_n=50)
    r_atr09_50 = run_with_target_n(cand_atr_09, min_n=50)
    r_atr10_50 = run_with_target_n(cand_atr_10, min_n=50)
    r_cd4_50 = run_with_target_n(cand_cd_4, min_n=50)

    print(summarize("base_trend_loose", r_base_50))
    print(summarize("atr_min_0.9", r_atr09_50))
    print(summarize("atr_min_1.0", r_atr10_50))
    print(summarize("cooldown_4", r_cd4_50))
