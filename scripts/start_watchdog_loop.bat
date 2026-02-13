@echo off
REM Starts watchdog loop (checks every 60s). Disable by deleting AUTOSTART_BOT.enabled
set ROOT=C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot
set PS1=%ROOT%\scripts\watchdog_loop.ps1
start "Upbit watchdog loop" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
