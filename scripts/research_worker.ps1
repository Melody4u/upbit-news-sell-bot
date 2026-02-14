$ErrorActionPreference = 'Stop'

$root = 'C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot'
Set-Location $root

$flag = Join-Path $root 'AUTOSTART_RESEARCH.enabled'
if (-not (Test-Path -LiteralPath $flag)) { exit 0 }

$logsDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$py = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'backtest_date_range_phaseA.py'

$hbPath = Join-Path $logsDir 'research.heartbeat'
$statePath = Join-Path $logsDir 'research_state.json'

function NowEpoch {
  return [double]([DateTimeOffset]::Now.ToUnixTimeSeconds())
}

function Run-Backtest([string]$market, [string]$start, [string]$end, [string]$tag, [string]$profile) {
  $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
  $outPath = Join-Path $logsDir "research_${market}_${tag}_${profile}_${ts}.out.log"
  $errPath = Join-Path $logsDir "research_${market}_${tag}_${profile}_${ts}.err.log"

  # Base profile = conservative but without extra filters.
  # Improved profile = sisyphus-guided tweaks (confirm=3 + vol-spike block).
  $confirmBars = if ($profile -eq 'IMP') { '3' } else { '2' }

  $args = @(
    $script,
    '--market', $market,
    '--start', $start,
    '--end', $end,
    '--wallst-v1',
    '--wallst-soft',
    '--highwr-v1',
    '--early-fail',
    '--early-fail-mode', 'hybrid',
    '--fib-lookback', '240',
    '--scout-min-score', '20',
    '--good-gate-mode', 'none',
    '--pyramiding',
    '--pos-cap-total', '0.90',
    '--mdd-limit-pct', '0.30',
    '--risk-per-trade', '0.015',
    '--downtrend-mode', 'h4',
    '--downtrend-early-fail-min-r', '1.2',
    '--swing-stop',
    '--swing-stop-lookback', '20',
    '--swing-stop-confirm-bars', $confirmBars
  )

  if ($profile -eq 'IMP') {
    $args += @('--vol-spike-block', '--vol-spike-atr-mult', '2.5', '--vol-spike-hold-bars', '2')
  }

  # Run synchronously so we always have a complete output file.
  $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root -RedirectStandardOutput $outPath -RedirectStandardError $errPath -NoNewWindow -PassThru
  $p.WaitForExit()

  # heartbeat for watchdog-style monitoring
  Set-Content -LiteralPath $hbPath -Value (NowEpoch) -Encoding ascii

  return [pscustomobject]@{
    tag = $tag
    profile = $profile
    market = $market
    start = $start
    end = $end
    exitCode = $p.ExitCode
    outLog = Split-Path -Leaf $outPath
    errLog = Split-Path -Leaf $errPath
    finishedAt = (Get-Date).ToString('s')
  }
}

$run = [ordered]@{
  startedAt = (Get-Date).ToString('s')
  note = 'Research worker: baseline vs improved (sisyphus-guided). Backtests only. Safe to run on boot.'
  runs = @()
}

try {
  # Half-year slices + 2025Q4 (stress)
  $slices = @(
    @{ tag = '2023H1'; start = '2023-01-01'; end = '2023-07-01' },
    @{ tag = '2023H2'; start = '2023-07-01'; end = '2024-01-01' },
    @{ tag = '2024H1'; start = '2024-01-01'; end = '2024-07-01' },
    @{ tag = '2024H2'; start = '2024-07-01'; end = '2025-01-01' },
    @{ tag = '2025H1'; start = '2025-01-01'; end = '2025-07-01' },
    @{ tag = '2025H2'; start = '2025-07-01'; end = '2026-01-01' },
    @{ tag = '2025Q4'; start = '2025-10-01'; end = '2026-01-01' }
  )

  foreach ($sl in $slices) {
    $run.runs += (Run-Backtest -market 'KRW-BTC' -start $sl.start -end $sl.end -tag $sl.tag -profile 'BASE')
    $run.runs += (Run-Backtest -market 'KRW-BTC' -start $sl.start -end $sl.end -tag $sl.tag -profile 'IMP')
  }
} catch {
  $run.error = $_.Exception.Message
}

$run.endedAt = (Get-Date).ToString('s')
$run | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding utf8
