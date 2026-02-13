$ErrorActionPreference = 'Stop'

$root = 'C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot'
Set-Location $root

$flag = Join-Path $root 'AUTOSTART_RESEARCH.enabled'
$worker = Join-Path $root 'scripts\research_worker.ps1'

if (-not (Test-Path -LiteralPath $flag)) { exit 0 }

while ($true) {
  try {
    if (-not (Test-Path -LiteralPath $flag)) { exit 0 }
    powershell -NoProfile -ExecutionPolicy Bypass -File $worker | Out-Null
  } catch {
    # swallow; retry later
  }
  # Avoid burning CPU: rerun every 3 hours
  Start-Sleep -Seconds 10800
}
