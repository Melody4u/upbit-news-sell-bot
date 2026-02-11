import os
from datetime import datetime, timedelta

import pandas as pd
import pyupbit
from dotenv import load_dotenv


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def run_market(market: str, min_rr: float = 3.0, relaxed: bool = False):
    # KST 기준 "어제"를 폭넓게 포함하도록 최근 300개 30분봉 사용 (~6일)
    df30 = pyupbit.get_ohlcv(market, interval="minute30", count=300)
    if df30 is None or len(df30) < 120:
        return {"market": market, "error": "ohlcv_fetch_failed"}

    df240 = pyupbit.get_ohlcv(market, interval="minute240", count=120)
    if df240 is None or len(df240) < 40:
        return {"market": market, "error": "ohlcv_fetch_failed_240"}

    df30 = df30.copy()
    df240 = df240.copy()

    # 지표
    df30["ma20"] = df30["close"].rolling(20).mean()
    df30["ma50"] = df30["close"].rolling(50).mean()
    df30["atr"] = compute_atr(df30, 14)

    df240["ma20_240"] = df240["close"].rolling(20).mean()

    # 30분봉에 240분 추세 매핑
    mtf = df240[["ma20_240", "close"]].reindex(df30.index, method="ffill")
    df30["ma20_240"] = mtf["ma20_240"]
    df30["close_240"] = mtf["close"]

    # 어제(Asia/Seoul) 필터
    now_kst = datetime.utcnow() + timedelta(hours=9)
    y = (now_kst - timedelta(days=1)).date()
    day_start = pd.Timestamp(datetime(y.year, y.month, y.day))
    day_end = day_start + pd.Timedelta(days=1)

    day_df = df30[(df30.index >= day_start) & (df30.index < day_end)].copy()
    if day_df.empty:
        return {"market": market, "error": "no_yesterday_bars"}

    in_pos = False
    entry = 0.0
    stop = 0.0
    tp = 0.0

    trades = []
    blocked = 0

    for i in range(1, len(day_df)):
        row = day_df.iloc[i]
        prev = day_df.iloc[i - 1]

        if pd.isna(row["ma20"]) or pd.isna(row["ma50"]) or pd.isna(row["atr"]) or pd.isna(row["ma20_240"]):
            continue

        if not in_pos:
            # 코어 룰(간이):
            # 1) 30분 추세: ma20 > ma50
            # 2) 240분 추세: close_240 > ma20_240
            # 3) 돌파: close가 직전 고가 상향 돌파
            cond30 = row["ma20"] > row["ma50"]
            cond240 = row["close_240"] > row["ma20_240"]
            breakout = (row["high"] > prev["high"]) if relaxed else (row["close"] > prev["high"])

            if cond30 and cond240 and breakout:
                entry = float(row["close"])
                atr = float(row["atr"])
                # 간이 손절/익절: stop=1ATR, tp=min_rr*1ATR
                stop = entry - atr
                tp = entry + atr * min_rr
                if entry <= stop:
                    blocked += 1
                    continue
                in_pos = True
            else:
                blocked += 1
                continue
        else:
            # 봉 내 stop/tp 터치 확인
            low = float(row["low"])
            high = float(row["high"])
            exit_price = None
            result = None

            if low <= stop:
                exit_price = stop
                result = "loss"
            if high >= tp:
                # 같은 봉에서 둘 다 터치면 보수적으로 loss 우선
                if exit_price is None:
                    exit_price = tp
                    result = "win"

            # 종료 시 강제 청산(어제 마지막 봉)
            if i == len(day_df) - 1 and exit_price is None:
                exit_price = float(row["close"])
                result = "win" if exit_price > entry else "loss"

            if exit_price is not None:
                pnl_r = (exit_price - entry) / (entry - stop) if (entry - stop) > 0 else 0
                trades.append({"result": result, "pnl_r": pnl_r})
                in_pos = False

    total = len(trades)
    wins = sum(1 for t in trades if t["result"] == "win")
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0.0

    gross_win = sum(max(0.0, t["pnl_r"]) for t in trades)
    gross_loss = abs(sum(min(0.0, t["pnl_r"]) for t in trades))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    return {
        "market": market,
        "date": str(y),
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "blocked_signals": blocked,
    }


if __name__ == "__main__":
    load_dotenv()
    min_rr = float(os.getenv("MIN_RR", "3.0"))

    markets = ["KRW-BTC", "KRW-ETH"]
    for m in markets:
        strict = run_market(m, min_rr=min_rr, relaxed=False)
        print({"mode": "strict", **strict})

        # 샘플이 너무 적으면 비교용 완화 모드도 같이 출력
        if strict.get("trades", 0) == 0:
            relaxed = run_market(m, min_rr=max(2.0, min_rr - 1.0), relaxed=True)
            print({"mode": "relaxed", **relaxed})
