# DRY_RUN 데일리 리포트 템플릿 (Phase A)

## 0) 오늘 한 줄 결론
- (예) 진입 신호는 있었으나 minute consensus 부족으로 대부분 차단 / heavy stage 0회

## 1) 실행 요약
- Market: KRW-ETH / KRW-BTC
- Mode: DRY_RUN=true
- CHECK_SECONDS: 180
- ENABLE_MTF_SCORE_SIZING: true
- Minute consensus: minute3+minute5+minute15 AND
- Hour score TF: 30m/1h/4h/1d/1w (0~100)

## 2) 체결(시뮬) 통계
- 매수(건수/총액):
- 매도(건수/총액):
- net 포지션 변화(대략):

## 3) Stage 분포 (score 기반)
- scout(30~49):
- light(50~64):
- medium(65~79):
- heavy(80~100):

## 4) Signal 차단(블락) Top 이유
- minute_mtf_no_consensus:
- hour_mtf_score_low:
- rr_block:
- atr_regime_block:
- spread_block:
- post_entry_cooldown_block:
- breakout_gate_block:

## 5) 하드스탑 이벤트 (있을 때만)
- 발생 여부: YES/NO
- 시간/가격/사유:
- 이후 24h 복구 규칙: minute consensus + score>=scout 시 자동 재개
- 24h 내 복구 신호: YES/NO
- (복구 신호 없으면) 오늘 시장 복기 메모:

## 6) 내일 1변수 실험 제안(선택)
- 변경 변수 1개:
- 기대 효과:
- 실패 시 롤백 기준:
