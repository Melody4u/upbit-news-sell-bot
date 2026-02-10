# Upbit Auto Trading Bot (Starter)

업비트 보유 포지션을 대상으로,
- 차트 기반 다중 시그널(추세/브레이크아웃/변동성)
- Brave 뉴스 리스크 필터
를 결합해 매수/매도 판단을 보조하는 자동매매 봇입니다.

## 1) 설치

```bash
cd upbit-auto-trading-bot
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## 2) 환경변수(.env)

필수:
- `UPBIT_ACCESS_KEY`
- `UPBIT_SECRET_KEY`
- `BRAVE_API_KEY`

권장 기본값:
- `DRY_RUN=true` (실주문 없이 테스트)
- `SELL_RATIO=0.25`
- `BUY_RATIO=0.25`

## 3) 실행

```bash
python bot.py
```

## 3-1) 대시보드 실행 (로컬 웹)

```bash
python dashboard.py
# 실행 시 브라우저 자동 오픈 (기본값)
# 수동 접속: http://127.0.0.1:8787
```

## 4) 핵심 기능

- MA20/MA200 기반 추세 + 스퀴즈/브레이크아웃 판단
- VWMA100(거래량 가중 이동평균) 기반 지지/저항 보조 필터
- 멀티 타임프레임 추세 확인(기본: 4시간봉 MA50>MA200)
- 손익비(R:R) 필터(`MIN_RR`) 미충족 시 매수 차단
- 중복 주문 방지: 쿨다운(`TRADE_COOLDOWN_SECONDS`) + 신호 해시 기반 멱등성
- 재시작 내구성: 핵심 상태(last_action/signal/pending_order/stop_price)를 runtime_state에 저장/복구
- 주문 추적: UUID 기반 pending 상태 확인 후 다음 주문 허용
- 긴급 청산(MA 이탈) 시 멱등성 우회 여부 옵션(`EMERGENCY_SELL_BYPASS_IDEMPOTENCY`)
- 슬리피지 가드(`MAX_SLIPPAGE_BPS`) 및 추가 Kill-switch(`MAX_DAILY_LOSS_PCT`, `MAX_TRADES_PER_DAY`, `MAX_POSITION_KRW`)
- 운영 알림 웹훅(`ALERT_WEBHOOK_URL`, `ALERT_EVENTS`)
- 브레이크아웃 게이트 결합 방식 파라미터화(`BREAKOUT_GATE_MODE=and|or`)
- EMA/ADX, Donchian, ATR, RSI, 거래량 보조 시그널
- 뉴스 리스크 반영(네거티브 뉴스 점수)
- 손절/청산 규칙, 리트레이스 추가진입, 리스크 중단(DD/연간 손절 횟수)
- 거래 저널(JSONL) + `report.py`로 일/주/월 리포트 생성
- Risk mode 자동 전환(aggressive/neutral/conservative)
  - `MARKET_RISK_PATH`의 `risk_score(0~100)` 우선
  - 없으면 내부 뉴스 점수 기반 fallback

## 5) 주의

- 본 코드는 투자 조언이 아닙니다.
- 반드시 소액/테스트로 먼저 검증하세요.
- API 키는 `.env`로만 관리하고 커밋하지 마세요.
