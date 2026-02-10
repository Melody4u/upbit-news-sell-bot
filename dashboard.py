import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pyupbit
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
LOG_PATH = APP_ROOT / "logs" / "trade_journal.jsonl"
RISK_PATH = APP_ROOT / "logs" / "market_risk.json"
RUNTIME_PATH = APP_ROOT / "logs" / "runtime_state.json"

MARKET = os.getenv("MARKET", "KRW-BTC")

app = Flask(__name__, template_folder=str(APP_ROOT / "templates"), static_folder=str(APP_ROOT / "static"))


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _format_ts(ts: int):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%m-%d %H:%M")
    except Exception:
        return "-"


def _account_snapshot(market: str):
    access = os.getenv("UPBIT_ACCESS_KEY", "")
    secret = os.getenv("UPBIT_SECRET_KEY", "")
    if not access or not secret:
        return {"connected": False, "error": "UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY missing"}

    upbit = pyupbit.Upbit(access, secret)
    balances = upbit.get_balances() or []
    base = market.split("-")[1]

    coin_balance = 0.0
    avg_buy_price = 0.0
    krw_balance = 0.0

    for b in balances:
        cur = b.get("currency")
        if cur == base:
            coin_balance = float(b.get("balance") or 0)
            avg_buy_price = float(b.get("avg_buy_price") or 0)
        elif cur == "KRW":
            krw_balance = float(b.get("balance") or 0)

    price = pyupbit.get_current_price(market) or 0
    coin_value = coin_balance * float(price)
    equity = krw_balance + coin_value
    pnl_pct = 0.0
    if avg_buy_price > 0 and price > 0:
        pnl_pct = ((float(price) / avg_buy_price) - 1.0) * 100

    return {
        "connected": True,
        "market": market,
        "price": float(price),
        "krw_balance": krw_balance,
        "coin_balance": coin_balance,
        "avg_buy_price": avg_buy_price,
        "coin_value": coin_value,
        "equity": equity,
        "pnl_pct": pnl_pct,
    }


@app.route("/")
def index():
    return render_template("index.html", market=MARKET)


@app.route("/api/overview")
def api_overview():
    market = request.args.get("market", MARKET)
    rows = _read_jsonl(LOG_PATH)
    rows = rows[-200:]

    latest_trade = rows[-1] if rows else {}
    buy_count = sum(1 for r in rows if r.get("side") in ("buy", "buy_addon"))
    sell_count = sum(1 for r in rows if r.get("side") == "sell")

    risk_mode = "neutral"
    risk_score = None
    risk_source = "none"
    if RISK_PATH.exists():
        try:
            risk = json.loads(RISK_PATH.read_text(encoding="utf-8"))
            risk_score = risk.get("risk_score")
            risk_source = risk.get("source", "file")
            if isinstance(risk_score, (int, float)):
                if risk_score <= 30:
                    risk_mode = "aggressive"
                elif risk_score >= 61:
                    risk_mode = "conservative"
        except Exception:
            risk_source = "file_error"

    account = _account_snapshot(market)

    return jsonify(
        {
            "market": market,
            "risk_mode": risk_mode,
            "risk_score": risk_score,
            "risk_source": risk_source,
            "trade_count": len(rows),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "latest_trade": {
                "side": latest_trade.get("side", "-"),
                "price": latest_trade.get("price", 0),
                "score": latest_trade.get("score", 0),
                "reasons": latest_trade.get("reasons", []),
                "time": _format_ts(latest_trade.get("ts", 0)),
            },
            "account": account,
        }
    )


@app.route("/api/ohlcv")
def api_ohlcv():
    market = request.args.get("market", MARKET)
    interval = request.args.get("interval", "minute60")
    count = int(request.args.get("count", "220"))

    df = pyupbit.get_ohlcv(market, interval=interval, count=count)
    if df is None or len(df) == 0:
        return jsonify({"ok": False, "candles": []})

    close = df["close"]
    vol = df["volume"]
    ma20 = close.rolling(20).mean()
    ma200 = close.rolling(200).mean()
    vwma100 = (close * vol).rolling(100).sum() / vol.rolling(100).sum()

    candles = []
    for idx, row in df.iterrows():
        candles.append(
            {
                "time": int(idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "ma20": None if ma20.loc[idx] != ma20.loc[idx] else float(ma20.loc[idx]),
                "ma200": None if ma200.loc[idx] != ma200.loc[idx] else float(ma200.loc[idx]),
                "vwma100": None if vwma100.loc[idx] != vwma100.loc[idx] else float(vwma100.loc[idx]),
            }
        )

    return jsonify({"ok": True, "candles": candles})


@app.route("/api/trades")
def api_trades():
    rows = _read_jsonl(LOG_PATH)
    rows = rows[-120:]
    out = []
    for r in reversed(rows):
        out.append(
            {
                "time": _format_ts(r.get("ts", 0)),
                "side": r.get("side", "-"),
                "price": r.get("price", 0),
                "score": r.get("score", 0),
                "reasons": r.get("reasons", []),
            }
        )
    return jsonify({"rows": out})


def _report_summary(mode: str):
    rows = _read_jsonl(LOG_PATH)
    now = datetime.now()
    if mode == "daily":
        start = datetime(now.year, now.month, now.day)
    elif mode == "weekly":
        start = now - timedelta(days=7)
    else:
        start = datetime(now.year, now.month, 1)

    selected = [r for r in rows if datetime.fromtimestamp(int(r.get("ts", 0))) >= start]
    buy = sum(1 for r in selected if r.get("side") in ("buy", "buy_addon"))
    sell = sum(1 for r in selected if r.get("side") == "sell")
    stop = sum(1 for r in selected if r.get("stop_event"))

    reason_counts = {}
    for r in selected:
        for reason in r.get("reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "mode": mode,
        "count": len(selected),
        "buy": buy,
        "sell": sell,
        "stop": stop,
        "top_reasons": top_reasons,
    }


@app.route("/api/report")
def api_report():
    mode = request.args.get("mode", "daily")
    if mode not in ("daily", "weekly", "monthly"):
        mode = "daily"
    return jsonify(_report_summary(mode))


@app.route("/api/news")
def api_news():
    news = []
    risk_meta = {"risk_score": None, "source": "none", "updated_at": None}
    if RISK_PATH.exists():
        try:
            data = json.loads(RISK_PATH.read_text(encoding="utf-8"))
            risk_meta = {
                "risk_score": data.get("risk_score"),
                "source": data.get("source", "file"),
                "updated_at": data.get("updated_at"),
            }
            raw_news = data.get("news") or data.get("items") or []
            if isinstance(raw_news, list):
                for n in raw_news[:10]:
                    if isinstance(n, dict):
                        news.append({
                            "title": n.get("title", ""),
                            "url": n.get("url", ""),
                            "summary": n.get("summary", n.get("description", "")),
                        })
                    else:
                        news.append({"title": str(n), "url": "", "summary": ""})
        except Exception:
            pass
    return jsonify({"risk": risk_meta, "items": news})


@app.route("/api/account")
def api_account():
    market = request.args.get("market", MARKET)
    account = _account_snapshot(market)
    runtime = {}
    if RUNTIME_PATH.exists():
        try:
            runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
        except Exception:
            runtime = {}
    return jsonify({"account": account, "runtime": runtime})


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8787"))
    app.run(host="127.0.0.1", port=port, debug=False)
