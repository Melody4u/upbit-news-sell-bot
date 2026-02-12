# DM Monitoring (Sisyphus)

시시퍼스 DM으로 들어온 명령을 제한적으로 실행하는 모니터링 스크립트.

## 보안 원칙
- 토큰 하드코딩 금지 (`SISYPHUS_BOT_TOKEN` 환경변수만 사용)
- 허용 사용자/DM 채널 고정 (`DM_ALLOWED_USER_ID`, `DM_ALLOWED_CHANNEL_ID`)
- 임의 쉘 명령 실행 금지 (화이트리스트 명령만 처리)

## 준비
```powershell
cd C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot

# 토큰/허용ID 설정
$env:SISYPHUS_BOT_TOKEN = "<YOUR_NEW_TOKEN>"
$env:DM_ALLOWED_USER_ID = "403807381231894528"
$env:DM_ALLOWED_CHANNEL_ID = "1471523142181716000"

# 실행
.\.venv\Scripts\python.exe .\scripts\dm_monitor.py
```

> ⚠️ 채널에 노출된 기존 토큰은 폐기하고 Discord Developer Portal에서 **재발급** 후 사용 권장.

## 지원 명령
- `help`
- `status`
- `git status`
- `ps bot`
- `log tail [N]`
- `dry-run start`
- `dry-run stop`

## 확장 포인트
- 명령별 role 기반 권한 분기
- 실행 결과를 JSON 로그로 남기기
- `dry-run start` 중복 실행 방지 락파일
