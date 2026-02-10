# Upbit News Sell Bot (Starter)

업비트 보유 포지션을 대상으로,
- 유명 트레이더들이 자주 쓰는 계열의 차트 규칙(추세/돌파/변동성/과열)
- Brave 뉴스 악재 점수
를 결합해 매도 신호를 내는 **스타터 코드**입니다.

## 1) 설치

```bash
cd upbit-news-sell-bot
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## 2) 설정(.env)

필수:
- `UPBIT_ACCESS_KEY`
- `UPBIT_SECRET_KEY`
- `BRAVE_API_KEY`

안전 기본값:
- `DRY_RUN=true` (실주문 안 나감)
- `SELL_RATIO=0.25` (신호 시 25%만 매도)

## 3) 실행

```bash
python bot.py
```

## 4) 동작 개요

- `CHECK_SECONDS`마다 현재 포지션/가격 확인
- `NEWS_INTERVAL_SECONDS`마다 Brave 검색으로 뉴스 점수 갱신
- 차트 신호 점수화(점수형 엔진):
  - EMA 추세 약화(데드크로스 + ADX 필터)
  - Donchian 하단 이탈
  - ATR 트레일링 스탑 이탈
  - RSI 과열 후 꺾임 + 거래량 스파이크
  - 평균단가 기준 손절
- 뉴스 악재 점수가 임계치 이상이면 추가 감점(매도 쪽 가중)
- 최종 점수가 `SIGNAL_SCORE_THRESHOLD` 이상이면 매도 신호

## 5) 주의

- 이 코드는 투자 조언이 아니라 **자동화 예시**입니다.
- 실거래 전 반드시 소액/테스트로 검증하세요.
- 키는 `.env`로만 관리하고 절대 깃허브에 올리지 마세요.
