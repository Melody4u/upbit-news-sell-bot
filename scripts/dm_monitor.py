import asyncio
import os
import subprocess
from datetime import datetime
from pathlib import Path

import discord

ALLOWED_USER_ID = int(os.getenv("DM_ALLOWED_USER_ID", "403807381231894528"))
ALLOWED_CHANNEL_ID = int(os.getenv("DM_ALLOWED_CHANNEL_ID", "1471523142181716000"))
BOT_TOKEN = os.getenv("SISYPHUS_BOT_TOKEN", "").strip()
PROJECT_DIR = Path(os.getenv("DM_PROJECT_DIR", Path(__file__).resolve().parents[1]))


intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
client = discord.Client(intents=intents)


def _run_cmd(command: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(
            command,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        out = out.strip() or "(no output)"
        return f"[exit={proc.returncode}]\n{out}"
    except Exception as e:
        return f"[error] {e}"


def _tail_log(lines: int = 40) -> str:
    log_path = PROJECT_DIR / "logs" / "bot.log"
    if not log_path.exists():
        return f"log file not found: {log_path}"
    text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(text[-lines:]) if text else "(empty log)"


def _status() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_head = _run_cmd(["git", "rev-parse", "--short", "HEAD"], timeout=10)
    return f"time={ts}\nproject={PROJECT_DIR}\n{git_head}"


def _help() -> str:
    return (
        "지원 명령:\n"
        "- help\n"
        "- status\n"
        "- git status\n"
        "- ps bot\n"
        "- log tail [N]\n"
        "- dry-run start\n"
        "- dry-run stop\n"
    )


def _handle_command(content: str) -> str:
    c = content.strip().lower()
    if c in {"help", "명령", "도움"}:
        return _help()
    if c == "status":
        return _status()
    if c == "git status":
        return _run_cmd(["git", "status", "--short"]) 
    if c in {"ps bot", "process", "프로세스"}:
        return _run_cmd(["tasklist", "/FI", "IMAGENAME eq python.exe"]) 
    if c.startswith("log tail"):
        parts = c.split()
        n = 40
        if len(parts) >= 3 and parts[2].isdigit():
            n = max(10, min(200, int(parts[2])))
        return _tail_log(n)
    if c == "dry-run start":
        # Start detached process (Windows)
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Start-Process -FilePath .\\.venv\\Scripts\\python.exe -ArgumentList 'bot.py' -WorkingDirectory '.'",
        ]
        return _run_cmd(cmd)
    if c == "dry-run stop":
        # Conservative stop: kill python processes running bot.py commandline
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'bot.py'} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }",
        ]
        return _run_cmd(cmd)

    return "알 수 없는 명령이야. 'help' 입력해줘."


@client.event
async def on_ready():
    print(f"[dm-monitor] logged in as {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not isinstance(message.channel, discord.DMChannel):
        return
    if message.author.id != ALLOWED_USER_ID:
        return
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    response = _handle_command(message.content)
    # Discord message length safety
    if len(response) <= 1800:
        await message.channel.send(response)
    else:
        chunks = [response[i:i + 1800] for i in range(0, len(response), 1800)]
        for ch in chunks[:4]:
            await message.channel.send(ch)


def main():
    if not BOT_TOKEN:
        raise RuntimeError("SISYPHUS_BOT_TOKEN is missing. Set env var first.")
    client.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
