@echo off
setlocal

REM Usage:
REM   opencode_run_safe.bat <model> "<prompt text>"
REM   opencode_run_safe.bat <model> --file <prompt.txt>

if "%~1"=="" goto :usage
set "MODEL=%~1"
shift

if "%~1"=="" goto :usage

set "SCRIPT_DIR=%~dp0"
set "PY_SCRIPT=%SCRIPT_DIR%opencode_run_safe.py"

if /I "%~1"=="--file" (
  if "%~2"=="" goto :usage
  python "%PY_SCRIPT%" -m "%MODEL%" -f "%~2"
  exit /b %ERRORLEVEL%
)

python "%PY_SCRIPT%" -m "%MODEL%" -p "%*"
exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   opencode_run_safe.bat ^<model^> "^<prompt text^>"
echo   opencode_run_safe.bat ^<model^> --file ^<prompt.txt^>
exit /b 2
