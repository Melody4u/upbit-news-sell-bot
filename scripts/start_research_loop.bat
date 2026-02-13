@echo off
REM Starts research loop (runs baseline backtests, then sleeps 3h; repeats)
set ROOT=C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot
set PS1=%ROOT%\scripts\research_loop.ps1
start "Upbit research loop" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
