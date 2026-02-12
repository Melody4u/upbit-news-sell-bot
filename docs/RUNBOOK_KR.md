# Upbit Auto Trading Bot RUNBOOK (A to Z)

이 문서는 "끄기 전/다음에 켰을 때" 바로 따라할 수 있는 운영 가이드입니다.

---

## 0) 현재 상태 요약 (핵심)

- 프로젝트명: **Upbit Auto Trading Bot**
- 원격 레포: `https://github.com/Melody4u/upbit-auto-trading-bot`
- 전략 성향: **스윙에 가까운 단기추세 추종** (1h 기반 + 4h MTF)
- 대상 자산: 우선 **BTC/ETH**
- 대시보드: `python dashboard.py` (기본 자동 브라우저 오픈)

주요 안정화 반영:
- 리스크 모드 자동전환(aggressive/neutral/conservative)
- 멱등성/중복 주문 방지(쿨다운 + signal hash + pending_order)
- 주문 UUID pending 추적 + 체결 검증
- 슬리피지/스프레드 필터
- ATR 레짐 필터
- 분할 익절 + 잔여 물량 트레일링
- Kill-switch 확장(DD/일일손실/일일횟수/포지션한도)

---

## 1) 폴더/파일 구조 (중요)

- `bot.py` : 메인 트레이딩 엔진
- `dashboard.py` : 로컬 대시보드 서버
- `report.py` : 일/주/월 리포트
- `.env.btc` : BTC 권장 프리셋
- `.env.eth` : ETH 권장 프리셋
- `.env` : 실제 실행에 사용되는 활성 설정
- `logs/trade_journal.jsonl` : 거래 로그
- `logs/runtime_state.json` : 런타임 상태(복구용)
- `logs/market_risk.json` : 시장 리스크 입력(외부/아침 브리핑 연동)
- `docs/EXPERIMENT_TEMPLATE.md` : 실험 템플릿
- `docs/ROLLBACK_POLICY.md` : 롤백 규칙

---

## 2) 최초/재시작 전 준비

PowerShell 기준:

```powershell
cd C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 가상환경이 없으면:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 3) 마켓 선택 (BTC/ETH)

### BTC로 실행
```powershell
copy .env.btc .env
```

### ETH로 실행
```powershell
copy .env.eth .env
```

> 키는 `.env`에서 반드시 확인:
- `UPBIT_ACCESS_KEY`
- `UPBIT_SECRET_KEY`
- `BRAVE_API_KEY`

---

## 4) 실행 방법

### A. 봇 실행
```powershell
python bot.py
```

### B. 대시보드 실행
```powershell
python dashboard.py
```
- 기본 접속: `http://127.0.0.1:8787`
- 기본 브라우저 자동 오픈: `DASHBOARD_AUTO_OPEN=true`

### C. 리포트 실행
```powershell
python report.py daily
python report.py weekly
python report.py monthly
```

---

## 5) 실전 전 체크리스트 (짧고 필수)

1. `DRY_RUN=true`로 먼저 테스트
2. 대시보드 탭 동작 확인
3. `report.py daily` 정상 출력 확인
4. `.env` 키 누락 없는지 확인
5. `logs/runtime_state.json` 생성/갱신 확인
6. `pending_order_timeout` 알림 과다 여부 확인

---

## 6) 핵심 파라미터 설명 (자주 만지는 것)

### 진입 품질
- `MIN_RR=3.0`
- `SPREAD_BPS_MAX` (BTC 12, ETH 18 권장)
- `ATR_REGIME_MIN_PCT`, `ATR_REGIME_MAX_PCT`
- `POST_ENTRY_COOLDOWN_BARS=3`

### 분할 익절 / 잔여 관리
- `PARTIAL_TP_LEVELS=1.0,2.0`
- `PARTIAL_TP_RATIOS=0.3,0.3`
- `TRAILING_STOP_MODE=atr`
- `TRAILING_ATR_MULT=2.5`
- `TRAILING_APPLY_TO_REMAINDER_ONLY=true`

### 안전장치
- `MAX_DRAWDOWN_PCT`
- `MAX_DAILY_LOSS_PCT`
- `MAX_TRADES_PER_DAY`
- `MAX_POSITION_KRW`
- `PENDING_ORDER_TIMEOUT_SEC`

### 운영 알림
- `ALERT_WEBHOOK_URL`
- `ALERT_EVENTS`

---

## 7) 변경 원칙 (중요)

- 한 번에 **하나만** 바꾼다.
- 바꾸기 전에 가설을 숫자로 적는다.
- 동일 조건으로 재검증한다.
- 기준 미달이면 즉시 롤백한다.

문서 사용:
- 실험 기록: `docs/EXPERIMENT_TEMPLATE.md`
- 롤백 조건: `docs/ROLLBACK_POLICY.md`

---

## 8) 자주 하는 실수 방지

1. `.env` 안 바꾸고 실행 (BTC/ETH 혼동)
2. `DRY_RUN=false` 너무 빨리 전환
3. 여러 파라미터 동시 변경
4. timeout 알림 과다 무시
5. 로그 파일을 git에 실수 커밋

---

## 9) 종료 전 루틴 (오늘 마무리용)

1. `report.py daily` 실행
2. 오늘 변경점 요약 3줄 기록
3. 현재 사용 `.env`를 확인(BTC/ETH)
4. 이상 로그(오류/timeout) 유무 점검
5. 프로세스 종료

---

## 10) 내일 재개 루틴 (3분 버전)

1. 가상환경 활성화
2. `.env` 선택(BTC 또는 ETH)
3. `python bot.py`
4. `python dashboard.py`
5. 대시보드에서 모드/계좌/로그 확인

끝. 이 문서만 보면 다시 바로 붙을 수 있음.
