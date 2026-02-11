@echo off
cd /d C:\Users\Home\.openclaw\workspace\upbit-news-sell-bot
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe dashboard.py
) else (
  python dashboard.py
)
