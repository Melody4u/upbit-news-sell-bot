$ErrorActionPreference = 'Stop'

$root = 'C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot'
Set-Location $root

$flag = Join-Path $root 'AUTOSTART_BOT.enabled'
$watchdog = Join-Path $root 'watchdog_dryrun.ps1'

# Safety: only run if the flag file exists.
if (-not (Test-Path -LiteralPath $flag)) {
  exit 0
}

while ($true) {
  try {
    if (-not (Test-Path -LiteralPath $flag)) { exit 0 }
    powershell -NoProfile -ExecutionPolicy Bypass -File $watchdog | Out-Null
  } catch {
    # swallow; next loop will retry
  }
  Start-Sleep -Seconds 60
}
