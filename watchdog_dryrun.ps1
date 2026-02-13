$ErrorActionPreference = 'Stop'

Set-Location -LiteralPath 'C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot'

$pidPath = Join-Path (Get-Location) 'logs\bot.pid'
$hbPath  = Join-Path (Get-Location) 'logs\bot.heartbeat'
$timeout = 180

function Get-NowEpoch {
  [double][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}

function Test-Health {
  param(
    [ref]$PidOut,
    [ref]$ReasonOut,
    [ref]$DebugOut
  )

  $PidOut.Value = $null
  $ReasonOut.Value = ''
  $DebugOut.Value = [ordered]@{
    pidTextLen = $null
    pidTextHead = $null
    pidDigits = $null
    pidParseMode = $null  # strict | digits_fallback
    pidParseError = $null
    getProcessOk = $null

    hbTextLen = $null
    hbTextHead = $null
    hbParseMode = $null   # strict
    hbParseError = $null

    nowEpoch = $null
    hbEpoch = $null
    ageSeconds = $null
  }

  if (-not (Test-Path -LiteralPath $pidPath)) {
    $ReasonOut.Value = 'pid_missing'
    return $false
  }

  $pidText = $null
  try {
    $pidText = (Get-Content -LiteralPath $pidPath -Raw -ErrorAction Stop).Trim()
    $DebugOut.Value.pidTextLen = $pidText.Length
    $DebugOut.Value.pidTextHead = ($pidText.Substring(0, [Math]::Min(50, $pidText.Length)))

    try {
      $PidOut.Value = [int]::Parse($pidText)
      $DebugOut.Value.pidParseMode = 'strict'
    } catch {
      $DebugOut.Value.pidParseError = $_.Exception.Message

      # fallback (only on failure): digits-only to tolerate BOM / stray chars
      $digits = ($pidText -replace '[^0-9]', '')
      $DebugOut.Value.pidDigits = $digits
      if ([string]::IsNullOrWhiteSpace($digits)) {
        throw
      }

      $PidOut.Value = [int]::Parse($digits)
      $DebugOut.Value.pidParseMode = 'digits_fallback'
    }
  } catch {
    if ($null -eq $DebugOut.Value.pidParseMode) { $DebugOut.Value.pidParseMode = 'failed' }
    if ($null -eq $DebugOut.Value.pidParseError) { $DebugOut.Value.pidParseError = $_.Exception.Message }
    $ReasonOut.Value = 'pid_parse_fail'
    return $false
  }

  try {
    Get-Process -Id $PidOut.Value -ErrorAction Stop | Out-Null
    $DebugOut.Value.getProcessOk = $true
  } catch {
    $DebugOut.Value.getProcessOk = $false
    $ReasonOut.Value = 'process_not_running'
    return $false
  }

  if (-not (Test-Path -LiteralPath $hbPath)) {
    $ReasonOut.Value = 'heartbeat_missing'
    return $false
  }

  $hb = $null
  try {
    $hbText = (Get-Content -LiteralPath $hbPath -Raw -ErrorAction Stop).Trim()
    $DebugOut.Value.hbTextLen = $hbText.Length
    $DebugOut.Value.hbTextHead = ($hbText.Substring(0, [Math]::Min(50, $hbText.Length)))

    $hb = [double]::Parse($hbText)
    $DebugOut.Value.hbParseMode = 'strict'
  } catch {
    $DebugOut.Value.hbParseMode = 'failed'
    $DebugOut.Value.hbParseError = $_.Exception.Message
    $ReasonOut.Value = 'heartbeat_parse_fail'
    return $false
  }

  $now = Get-NowEpoch
  $age = ($now - $hb)
  $DebugOut.Value.nowEpoch = $now
  $DebugOut.Value.hbEpoch = $hb
  $DebugOut.Value.ageSeconds = $age

  if ($age -gt $timeout) {
    $ReasonOut.Value = ('stalled_age_seconds=' + [math]::Round($age, 0))
    return $false
  }

  return $true
}

$result = [ordered]@{
  healthyBefore = $false
  restarted     = $false
  healthyAfter  = $false
  reasonBefore  = ''
  reasonAfter   = ''
  oldPid        = $null
  newPid        = $null
  outLog        = $null
  errLog        = $null
  startError    = $null
}

$pidRef = [ref]$null
$reasonRef = [ref]''
$debugRef = [ref]$null
$healthy = Test-Health -PidOut $pidRef -ReasonOut $reasonRef -DebugOut $debugRef

$result.healthyBefore = $healthy
$result.reasonBefore = $reasonRef.Value
$result.oldPid = $pidRef.Value
$result.debugBefore = $debugRef.Value

if ($healthy) {
  $result | ConvertTo-Json -Compress
  exit 0
}

# NOT healthy: restart
$result.restarted = $true

$oldPid = $pidRef.Value
if ($null -ne $oldPid) {
  try { Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue } catch {}
}

New-Item -ItemType Directory -Force -Path (Join-Path (Get-Location) 'logs') | Out-Null

$ts = (Get-Date).ToString('yyyyMMdd_HHmmss')
$outLog = 'logs\dryrun_bot_watchdog_' + $ts + '.out.log'
$errLog = 'logs\dryrun_bot_watchdog_' + $ts + '.err.log'
$result.outLog = $outLog
$result.errLog = $errLog

$py  = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$bot = Join-Path (Get-Location) 'bot.py'

try {
  Start-Process -FilePath $py -ArgumentList @($bot) -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden | Out-Null
} catch {
  $result.startError = $_.Exception.Message
  $result | ConvertTo-Json -Compress
  exit 2
}

Start-Sleep -Seconds 2

$pidRef2 = [ref]$null
$reasonRef2 = [ref]''
$debugRef2 = [ref]$null
$healthy2 = Test-Health -PidOut $pidRef2 -ReasonOut $reasonRef2 -DebugOut $debugRef2

$result.healthyAfter = $healthy2
$result.reasonAfter = $reasonRef2.Value
$result.newPid = $pidRef2.Value
$result.debugAfter = $debugRef2.Value

$result | ConvertTo-Json -Compress
if ($healthy2) { exit 10 } else { exit 11 }
